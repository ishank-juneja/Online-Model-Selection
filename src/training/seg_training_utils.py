import copy
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as dtc_utils
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer
import numpy as np
from src.training import MyAugs
import torch


class DatasetMapperForNPY(DatasetMapper):
    def __call__(self, dataset_dict):
        """
        Identical to default DatasetMapper other than callable being configured to
        read .npy instead of .png/.jpeg files
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # # # # # # # # # # #  # # # # #
        image = np.load(dataset_dict["file_name"])
        # # # # # # # # # # #  # # # # #
        dtc_utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = dtc_utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            dtc_utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


class Trainer(DefaultTrainer):
    """
    A class identical to the default trainer except that it invokes build_detection_train_loader with custom arguments
    """
    @classmethod
    def build_train_loader(cls, cfg):
        augs = MyAugs(cfg)
        dataloader = build_detection_train_loader(cfg,
                                                  mapper=DatasetMapperForNPY(cfg, is_train=True, augmentations=augs.aug_list)
                                                  )
        return dataloader
