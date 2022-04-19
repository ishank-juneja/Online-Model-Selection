import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects import point_rend
import os
from src.config import SegConfig
from src.learned_models.masker import Masker
from src.utils import SegDataset
from typing import Tuple


class Segmenter:
    """
    Wrapper class around dtc2 lib default predictor
    """
    def __init__(self, model_name: str):
        # segmentation related config
        self.seg_config = SegConfig()
        self.model_name = model_name

        if model_name is not None:
            seg_dataset, self.arch = self.disect_model(model_name)

            # Simple model trained omn
            self.simp_model = seg_dataset.get_simp_model()
            # Number of frames trained on
            self.nframes = seg_dataset.get_nframe()
            # Number of masks per simple model image trained with
            #  If trained with 2 masks, we look for top result in LHP and RHP separately
            self.ninstances = seg_dataset.get_nmasks()

            # Confidence threshold above which we declare an instance of simp_model
            self.seg_thresh = 0.1

            # Construct minimal config from arch backbone for inference
            cfg = get_cfg()

            # Setup model config using the base model yaml defaults
            #  loading the trained model into memory with DefaultPredictor requires this yaml and
            if self.arch == 'MRCNN':
                cfg.merge_from_file(model_zoo.get_config_file(self.seg_config.mrcnn_yaml))
            elif self.arch == 'PointRend-Instance':
                # Add PointRend-specific config
                point_rend.add_pointrend_config(cfg)
                cfg.merge_from_file(self.seg_config.pointrend_instance_yaml)
                cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
            elif self.arch == 'PointRend-Semantic':
                raise NotImplementedError("Test for this seg arch not implemented")
            else:
                raise NotImplementedError("Test for this seg arch not implemented")

            # Each segmentation model is trained to recognize one class to keep performances of each model independent
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

            # cfg now already contains everything we've set previously. We changed it a little-bit for inference:
            cfg.MODEL.WEIGHTS = os.path.join("models/segmentation", self.model_name + '.pt')

            # Report instances on which more than 30% confident
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.seg_thresh

            # Default in dtc2 is BGR to comply with opencv format
            cfg.INPUT.FORMAT = "RGB"

            # Build segmenter using this loaded cfg
            self.segmenter = DefaultPredictor(cfg)

            # Object for masking unprocessed frames
            self.masker = Masker(input_res=self.seg_config.imsize, output_res=self.seg_config.imsize)

            # Determine the shape of images that should be considered in-distribution given the kinds of images we trained on
            if self.nframes == '1frame':
                self.in_shape = (self.seg_config.imsize, self.seg_config.imsize)
            elif self.nframes == '2frame1mask' or self.nframes == '2frame2mask':
                self.in_shape = (self.seg_config.imsize, 2*self.seg_config.imsize)

    @staticmethod
    def disect_model(model_name: str):
        """
        Infer the simple-model trained on, the arch used, and the number of frames this segmentation is
        trained to operate on (1 frame or 2 concatenated frames) from model_name
        :param model_name:
        :return:
        """
        # Split up the complete model name by _
        attrs = model_name.split('_')
        # Name of the dataset folder this model was created from
        folder_name = attrs[1] + '_' + attrs[2] + '_' + attrs[3]
        # Wrapper object around a segmentation dataset: Infers simple model and nframes trained on
        seg_dataset = SegDataset(folder_name)
        # Architecture this segmentation model was trained with
        arch = attrs[4]
        return seg_dataset, arch

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.model_name is None:
            raise NotImplementedError("Cannot call empty segmenter object")
        else:
            # Check if shape of image in-distribution
            if frame.shape[0] == self.in_shape[0] and frame.shape[1] == self.in_shape[1]:
                return self.__segment(frame)
            else:
                raise ValueError("Model supports image shape {0}, whereas found {1}".format(self.in_shape, frame.shape[:2]))

    def __segment(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Segment out the simple model segmenter has been trained to segment out from frame and ret
        :param frame:
        :return: Masked versions of the input frame
        """
        # Instances returned in case of instance segmentation
        if self.arch == 'MRCNN' or self.arch == 'PointRend-Instance':
            return self.__segment_instance(frame)

    def __segment_instance(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Segment out the simple model segmenter has been trained to segment out from frame and ret
        :param frame:
        :return: Masked versions of the input frame
        """
        # Returned value from instance segmentation arch. is a dictionary
        #  with a single key called instances whose value is a list of dtc2.structs.Instance objects
        found_instances = self.segmenter(frame)['instances']

        if len(found_instances) == 0:
            # Found nothing in image return unsegmented
            # When returning unprocessed frame, return confidence score as self.seg_thresh (The minimum score)
            return frame, self.seg_thresh

        # In the cases where we expect to segment out a single instance from the image
        #  which are the cases of single-frame and concatanted frame with a single disjoint mask
        if self.ninstances == 1:
            # Extract the top mask as a torch tensor
            top_mask = found_instances[0].get_fields()['pred_masks']
            top_score = found_instances[0].get_fields()['scores']
            # Convert CxHxW tensor to HxWxC numpy array
            top_mask_np = top_mask.permute((1, 2, 0)).cpu().detach().numpy()[:, :, 0]
            # Convert top score pt cuda tensor to float
            conf = top_score.cpu().detach().numpy()[0]
        elif self.ninstances == 2:
            # Extract top LHP mask and top RHP mask
            #  Note: White line visual cue gets masked out as well
            found_LHP = False
            found_RHP = False
            # Iterate over the sorted list of instances and stop once
            #  seen a mask with content in both LHP and in RHP
            for idx in range(len(found_instances)):
                instance = found_instances[idx]
                # Get xcoord of center of the bounding box associated with this instance
                box_x = instance.get_fields()['pred_boxes'].get_centers()[0][0]
                if box_x > self.seg_config.imsize and not found_RHP:
                    # Found RHP mask
                    # conf_right =
                    rhp_mask = instance.get_fields()['pred_masks']
                    rscore = instance.get_fields()['scores']
                    conf_right = rscore.cpu().detach().numpy()[0]
                    found_RHP = True
                elif not found_LHP:
                    # Found LHP mask
                    lhp_mask = instance.get_fields()['pred_masks']
                    lscore = instance.get_fields()['scores']
                    conf_left = lscore.cpu().detach().numpy()[0]
                    found_LHP = True
                elif found_LHP and found_RHP:
                    break
            # If we can't find something in both left half and right half, return unsegmented ...
            if not (found_LHP and found_RHP):
                return frame, self.seg_thresh
            conf = (conf_left + conf_right)/2
            # Combine LHP and RHP masks into a single mask
            #  - - - - - - Not Tested - - - - - - -
            lhp_mask_np = lhp_mask.permute((1, 2, 0)).cpu().detach().numpy()[:, :self.seg_config.imsize, 0]
            rhp_mask_np = rhp_mask.permute((1, 2, 0)).cpu().detach().numpy()[:, self.seg_config.imsize:, 0]
            top_mask_np = np.hstack((lhp_mask_np, rhp_mask_np))
            #  - - - - - - Not Tested - - - - - - -
        else:
            raise NotImplementedError("Unsupported number of masks per frame")

        masked = self.masker.apply_mask(frame, top_mask_np)
        return masked, conf
        # return masked
