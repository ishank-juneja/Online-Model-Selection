"""
Adapted from script from the binary mask to COCO format tutorial:
https://github.com/waspinator/pycococreator/blob/master/examples/shapes/shapes_to_coco.py
"""

import argparse
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycocotools import mask
from pycococreatortools.pycococreatortools import binary_mask_to_rle, binary_mask_to_polygon
from src.utils.results_dir_manager import ResultDirManager

INFO = {
    "description": "Simple Models Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2022,
    "contributor": "ishank",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'supercategory': 'simple-model'
    }
]


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_npy(root, files):
    file_types = ['*.npy']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    # file_types = ['*.png', '*.npy']
    file_types = ['*.npy']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '_.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def create_annotation_info(annotation_id, image_id, category_info, binary_mask,
                           image_size=None, tolerance=2, bounding_box=None):

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    if category_info["is_crowd"]:
        # is_crowd is Unused
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else:
        is_crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentation:
            return None

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }

    return annotation_info


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[1],
            "height": image_size[0],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info


def write_cococreator_fmt_json(cococreator_fmt_dir: str, dir_manager: ResultDirManager):
    image_dir = os.path.join(cococreator_fmt_dir, "images")
    annotation_dir = os.path.join(cococreator_fmt_dir, "annotations")
    # Initialize dictionary that is to be saved as the coco format json output
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    image_id = 1
    segmentation_id = 1
    # filter for npy format images in dataset folder
    for root, _, files in os.walk(image_dir):
        image_files = filter_for_npy(root, files)
        image_files = dir_manager.natural_sort(image_files)
        # go through each image
        for image_filename in image_files:
            print("Now processing {0}".format(image_filename))
            image = np.load(image_filename)
            image_info = create_image_info(image_id, os.path.basename(image_filename), image.shape[:2])
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(annotation_dir):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    # print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.load(annotation_filename)

                    annotation_info = create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.shape[:2], tolerance=0)

                    # Annotation info can come out None if a chance empty frame is found but verify manually
                    # to be sure that any underlying countour detection from seg mask has not failed
                    if annotation_info is not None:
                        # Check if there are more than 1 polygonal contours indicating failure
                        if len(annotation_info['segmentation']) != 1:
                            print("{0} polygons returned for annotation {1} in image {2}".format(len(annotation_info['segmentation']),
                                                                                                 annotation_filename,
                                                                                                 image_filename))
                        coco_output["annotations"].append(annotation_info)
                    else:
                        print("Found no annotations for {0}".format(image_filename))

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    # split dir path to extract relevant strings needed for naming json file
    with open('{0}/coco_dict.json'.format(cococreator_fmt_dir), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


def make_segmentation_coco(dir_name: str, simple_model: str):
    # Assign the name to the categoary as the name of the simple model
    #  used as metadata for segmentation
    CATEGORIES[0]['name'] = simple_model

    dir_manager = ResultDirManager()

    dir = 'data/{0}'.format(dir_name)

    # List all dirs within cur_dir: example: ['train', 'test']
    dir_lst = dir_manager.list_dirs_in_dir(dir)

    # Write the json file for each
    for cococreator_fmt_dir in dir_lst:
        write_cococreator_fmt_json(cococreator_fmt_dir, dir_manager)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Follow dir format: https://patrickwasp.com/create-your-own-coco-style-dataset/
    parser.add_argument("--dir",
                        action='store',
                        nargs='+',
                        default="segmentation",
                        type=str,
                        help="training data dir for segmentation",
                        metavar=None)

    args = parser.parse_args()

    dir_manager = ResultDirManager()

    make_segmentation_coco(args.dir)
