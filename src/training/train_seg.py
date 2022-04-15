import argparse
from datetime import datetime
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from src.config import SegConfig
from src.utils import SegDataset
from src.training import SegTrainer
import torch


def main(args):
    # Setup dtc2 logger
    setup_logger()

    # Folder must follow naming convention
    # <SIMPLE_MODEL>seg_<NFRAMES>, example cartpole_seg_1frame
    folder = args.folder
    seg_dataset_obj = SegDataset(folder)
    simp_model = seg_dataset_obj.get_simp_model()
    # Number of frames this segmentation is trained to handle
    nframes = seg_dataset_obj.get_nframe()

    # Create a config object
    config = SegConfig(args, simp_model)

    # Register training data pre-prepared in coco-format as a detectron2 dataset
    empty_dict = {}
    register_coco_instances(name="seg_train",
                            metadata=empty_dict,
                            json_file="data/{0}/train/coco_dict.json".format(folder),
                            image_root="data/{0}/train/images".format(folder))
    register_coco_instances(name="seg_test",
                            metadata=empty_dict,
                            json_file="data/{0}/test/coco_dict.json".format(folder),
                            image_root="data/{0}/test/images".format(folder))

    # metadata for class labels
    metadata = MetadataCatalog.get("seg_train")

    if args.viz_input:
        # Verify data-loading is correct by visualizing loaded training data
        # Retrieve the detectron2 format dictionary associated with the registered dataset
        dataset_dicts = DatasetCatalog.get("seg_train")
        for d in random.sample(dataset_dicts, 10):
            print(d["file_name"])
            # Load in image in BGR format like openCV does
            img = np.load(d["file_name"])[..., ::-1]
            # Size of image frames assumed square
            img_size, _, _ = img.shape
            # scale is based on 0.7 being suitable for 800x800 images
            visualizer = Visualizer(img[..., ::-1], metadata=metadata, scale=1.0)
            out = visualizer.draw_dataset_dict(d)
            plt.imshow(out.get_image())
            plt.show()

    if args.arch == 'MRCNN':
        cfg = config.make_maskrcnn_config()
    elif args.arch == 'PointRend-Instance':
        cfg = config.make_pointrend_instance_config()
    elif args.arch == 'PointRend-Semantic':
        raise NotImplementedError
    else:
        raise NotImplementedError

    cfg.DATASETS.TRAIN = ("seg_train",)
    cfg.DATASETS.TEST = ("seg_test",)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    model_name = "model_{0}_{1}_{2}".format(folder, args.arch, current_time)
    cfg.OUTPUT_DIR = "runs/segmentation/{0}".format(model_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = SegTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    save_path = "models/segmentation/{0}.pt".format(model_name)
    torch.save(trainer.model.state_dict(), save_path)

    if args.evaluate or args.viz_output:
        cfg.MODEL.WEIGHTS = save_path
        # Confidence thresh above which objects segmented
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
        segmenter = DefaultPredictor(cfg)

    if args.viz_output:
        dataset_dicts = DatasetCatalog.get("seg_test")
        for d in random.sample(dataset_dicts, 20):
            im = np.load(d["file_name"])[..., ::-1]
            outputs = segmenter(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                           metadata=metadata,
                           scale=1.0,
                           instance_mode=ColorMode.IMAGE_BW
                           # remove the colors of unsegmented pixels. This option is only available for segmentation models
                           )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.imshow(out.get_image())
            plt.show()

    if args.evaluate:
        evaluator = COCOEvaluator("seg_test", output_dir="./runs/segmentation")
        test_loader = build_detection_test_loader(cfg, "seg_val")
        print(inference_on_dataset(segmenter.model, test_loader, evaluator))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder",
                        action='store',
                        type=str,
                        help="Folder within data that contains the dataset and the formatted version",
                        metavar=None,
                        dest="folder")

    parser.add_argument("--arch",
                        action='store',
                        choices=['MRCNN', 'PointRend-Instance', 'PointRend-Semantic'],
                        type=str,
                        help="Architecture to use for seg",
                        metavar=None,
                        dest="arch")

    parser.add_argument("--num-gpus",
                        action='store',
                        default=1,
                        type=int,
                        help="Number of GPUs to use for training",
                        metavar=None,
                        dest="num_gpus")

    parser.add_argument("--viz-input",
                        action='store_true',
                        help="whether visualize coco-format dataset being trained on",
                        dest="viz_input")

    parser.add_argument("--evaluate",
                        action='store_true',
                        help="whether run standard segmentation tests (AP metrics)",
                        dest="evaluate")

    parser.add_argument("--viz-output",
                        action='store_true',
                        help="whether run standard segmentation tests (AP metrics)",
                        dest="viz_output")

    parser.add_argument("--pretrained",
                        action='store_true',
                        help="Whether to use pretrained weights",
                        dest="pretrained")

    parser.add_argument("--augs",
                        action='store',
                        nargs='*',
                        choices=["no_fg_texture", "no_bg_simp_model", "no_bg_shape", "no_bg_imgnet"],
                        type=str,
                        help="Kinds of augmentations to be excluded from training: "
                             "fg_texture=Apply random FG textures"
                             "bg_simp_model=Have other simple models in background as distractors"
                             "bg_shape: Have simulator-esque random regular polygon shapes in the bg"
                             "bg_imgnet: Randomize the background with images from the imgnet dataset",
                        dest="excluded_augs",
                        metavar="excluded_augs")

    args = parser.parse_args()

    # Use launch for multi-gpu training
    launch(main, args.num_gpus, args=(args,),)
