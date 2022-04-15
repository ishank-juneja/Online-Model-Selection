"""
Derived from Detectron 2 tutorial:
https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5?usp=sharing
"""
import argparse
from datetime import datetime
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.projects import point_rend
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from src.training.seg_training_utils import DatasetMapperForNPY
from src.plotting.plot_detectron_overlays import MySimplifiedVisualizer
import torch


def main(args):
    empty_dict = {}
    register_coco_instances(name="seg_test",
                            metadata=empty_dict,
                            json_file="data/{0}/test/coco_dict.json".format(args.folder),
                            image_root="data/{0}/test/images".format(args.folder))

    # metadata for class labels
    metadata = MetadataCatalog.get("seg_test")

    cfg = get_cfg()
    # Add PointRend-specific config
    point_rend.add_pointrend_config(cfg)
    cfg.OUTPUT_DIR = "runs/segmentation"
    load_path = "models/segmentation/{0}.pt".format(args.model_load)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.base_cnn_model))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    if args.arch == 'MRCNN':
        cfg.merge_from_file(model_zoo.get_config_file(args.base_cnn_model))
        if args.pretrained:
            cfg.MODEL.WEIGHTS = load_path
    elif 'PointRend' in args.arch:
        cfg.merge_from_file(
            "detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
        if args.pretrained:
            cfg.MODEL.WEIGHTS = "models/segmentation/pretrained_point_rend.pkl"
        # Point-Rend arch specific param
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # cfg now already contains everything we've set previously. We changed it a little-bit for inference:
    cfg.MODEL.WEIGHTS = load_path
    # Report instances on which more than 30% confident
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    segmenter = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get("seg_test")
    for d in random.sample(dataset_dicts, 20):
        im = np.load(d["file_name"])[..., ::-1]
        outputs = segmenter(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        # output_mask = outputs["instances"].get_fields()['pred_masks']
        # mask_np = np.array(output_mask.detach().cpu())
        #
        # plt.imshow(mask_np[0, :, :])
        # plt.show()
        #
        # # Viz downsampled version of a mask
        # plt.imshow(mask_np[0, ::8, ::8])
        # plt.show()

        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image())
        plt.show()

    evaluator = COCOEvaluator("seg_test", output_dir="./runs")
    test_loader = build_detection_test_loader(cfg, "seg_test", mapper=DatasetMapperForNPY(cfg, is_train=False))
    print(inference_on_dataset(segmenter.model, test_loader, evaluator))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch",
                        action='store',
                        type=str,
                        choices=['MRCNN', 'PointRend-Instance', 'PointRend-Semantic'],
                        help="Name of the architecture to use",
                        metavar="arch",
                        dest="arch")

    parser.add_argument("--base-cnn-model",
                        action='store',
                        type=str,
                        default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                        help="name of .pt file containing model state_dict in models/segmentation/",
                        metavar=None,
                        dest="base_cnn_model")

    parser.add_argument("--model-load",
                        action='store',
                        type=str,
                        help="name of .pt file containing model state_dict in models/segmentation/",
                        metavar=None,
                        dest="model_load")

    parser.add_argument("--use-pretrained",
                        action='store_true',
                        help="Whether or not to use pre-trained weights while training",
                        dest="pretrained")

    parser.add_argument("--folder",
                        action='store',
                        type=str,
                        help="Name of the folder into which to put dataset",
                        metavar="folder")

    args = parser.parse_args()

    main(args)
