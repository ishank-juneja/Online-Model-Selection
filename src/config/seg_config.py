from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.projects import point_rend


class SegConfig:
    """
    Config for all segmentation related things
    """
    def __init__(self, args=None, simp_model: str = None):
        # Minimum size of images to use is 64x64
        self.imsize = 512

        # Create a config object with dtc2 defaults
        self.cfg = get_cfg()

        #  - - - - MaskRCNN - - -
        # Base MaskRCNN specific model config to use
        # "yaml config file from options: "https://github.com/facebookresearch/detectron2/tree/main/configs",
        self.mrcnn_yaml = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

        # - - - - - PointRend with Mask RCNN trunk - - - -
        # PointRend instance seg pretrained weights to use
        self.pointrend_instance_yaml = "detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
        self.pointrend_instance_weights = "models/segmentation/pretrained_point_rend.pkl"

        if args is not None:
            # Whether to use pretrained weights
            self.pretrained = args.pretrained

            # Augmentations to be applied while training
            self.excluded_augs = args.excluded_augs

        if simp_model is not None:
            # Simple model for which this model is being trained
            self.simp_mod = simp_model

    def make_maskrcnn_config(self):
        """
        Takes in a config and adds MaskRCNN specific config
        :return:
        """
        self.cfg.merge_from_file(model_zoo.get_config_file(self.mrcnn_yaml))
        self.add_common_config()
        # If use pretrained weights from COCO segmentation dataset
        if self.pretrained:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.mrcnn_yaml)
        return self.cfg

    def make_pointrend_instance_config(self):
        """
        Config for doing instance segmentation with point rend
        :return:
        """
        # Add PointRend-specific config
        point_rend.add_pointrend_config(self.cfg)
        self.cfg.merge_from_file(self.pointrend_instance_yaml)
        self.add_common_config()
        # If use pretrained weights from COCO segmentation dataset
        if self.pretrained:
            self.cfg.MODEL.WEIGHTS = self.pointrend_instance_weights
        # Point-Rend arch specific param
        self.cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
        return self.cfg

    def make_pointrend_semantic_config(self):
        """
        Make pointrend config for semantic segmentation
        :return:
        """
        self.add_common_config()
        return self.cfg

    def add_common_config(self):
        """
        Adds in the common config related parameters
        Called after model specific parameters to override model specific defaults from their yaml
        :return:
        """
        # Default in dtc2 is BGR to comply with opencv format
        self.cfg.INPUT.FORMAT = "RGB"

        self.cfg.OUTPUT_DIR = "runs/segmentation"
        self.cfg.DATALOADER.NUM_WORKERS = 2

        # Images per bacth
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR

        # https://github.com/facebookresearch/Detectron/issues/447#:~:text=One%20iteration%20refers%20to%20one,mini%2Dbatch%20size%20of%20TRAIN.
        self.cfg.SOLVER.MAX_ITER = 10000
        self.cfg.SOLVER.STEPS = []  # do not decay learning rate
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        # By default we apply all augmentations
        self.cfg.FG_AUG = True
        self.cfg.SIMP_MOD_AUG = True
        self.cfg.SHAPE_AUG = True
        self.cfg.BG_IMGNET = True
        # Turn off excluded augmentations
        for excluded in self.excluded_augs:
            if excluded == 'no_fg_texture':
                self.cfg.FG_AUG = False
            elif excluded == 'no_bg_simp_model':
                self.cfg.SIMP_MOD_AUG = False
            elif excluded == 'no_bg_shape':
                self.cfg.SHAPE_AUG = False
            elif excluded == 'no_bg_imgnet':
                self.cfg.BG_IMGNET = False

        self.cfg.SIMP_MOD = self.simp_mod
