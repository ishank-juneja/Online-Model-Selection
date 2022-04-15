"""
Cleanup a directory full of images of non-standard images
"""
import argparse
import numpy as np
import os
from PIL import Image
from src.utils import ResultDirManager


def main(args):
    dir_manager = ResultDirManager()
    img_list = dir_manager.list_dir_objects("data/" + args.folder, args.ext)

    for file_name in img_list:
        problem = False
        # First try opening the image and converting to a numpy array
        try:
            with Image.open(file_name) as pil_image:
                new_bg_img = np.array(pil_image)
        except OSError:
            print("Exception, unable to convert texture {0} to np array".format(file_name))
            problem = True
        finally:
            if len(new_bg_img.shape) < 3:
                print("Single channel image {0} encountered".format(file_name))
                problem = True
            elif new_bg_img.shape[2] != 3:
                print("Multi channel image {1} with {0} channels encountered".format(new_bg_img.shape[2], file_name))
                problem = True
            if args.purge and problem:
                os.remove(file_name)
                print("File with name {0} purged".format(file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder",
                        action='store',
                        type=str,
                        help="Folder within data which to cleanup",
                        metavar=None,
                        dest="folder")

    parser.add_argument("--extension",
                        action='store',
                        choices=["*.jpg", "*.jpeg", "*.JPEG", "*.bmp", "*.png"],
                        type=str,
                        help="Extension for the kinds of files to cleanup",
                        metavar=None,
                        dest="ext")

    parser.add_argument("--purge",
                        action='store_true',
                        help="Whether to purge the problematic images",
                        dest="purge")

    args = parser.parse_args()

    main(args)


