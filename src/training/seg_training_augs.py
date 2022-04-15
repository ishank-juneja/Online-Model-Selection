"""
A T.Augmentation object is passed to the train data loader
"""
from detectron2.data import transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from src.config import AugmentationDataDirs, ImageDataDir
from src.utils import ResultDirManager


def display_image_channel_wise(augmented: np.ndarray, original: np.ndarray, augmentation: np.ndarray):
    """
    For debugging data augmentations
    :return:
    """
    # Debug data-type compatibility issues between images being combined
    print("Original image data type".format(original.dtype))
    print("Image augmentation data type".format(augmentation.dtype))
    print("Augmented image data type".format(augmented.dtype))

    # Visually inspect the combined image
    fig = plt.figure(figsize=(20, 5))
    ax = fig.subplots(1, 4)
    ax[0].imshow(augmented)
    # Inspect all channels separately
    for idx in range(1, 4):
        ax[idx].imshow(augmented[:, :, idx - 1])
    plt.show()


class ReplaceForeground(T.Transform):
    """
    Replaces the foreground of a simple model simulated black BG image with a new texture

    IMP: This FG replacement only works if any images are stored as npy/bmp format
    with compressed formats bg replacement like this fails due to compression artifcats
    """
    def __init__(self, new_fg: np.ndarray):
        """
        Assumes new_fg is a compatible std RGB image
        :param new_fg:
        """
        self.new_fg = new_fg
        # For debugging augmentations
        self.display_output = False

    def apply_image(self, img: np.ndarray):
        # If any channel of a certain pixel is zero in original image
        # then we want to leave as is otherwise we use the RGB pixel from the new FG
        RGB_mask = np.repeat(np.all(img < 10, axis=2)[:, :, None], 3, axis=2)
        augmented = np.where(RGB_mask, img, self.new_fg)
        if self.display_output:
            display_image_channel_wise(augmented=augmented, original=img, augmentation=self.new_fg)
        return augmented

    # Do nothing to the segmentations
    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    # Apply no transform to coordinates
    def apply_coords(self, coords: np.ndarray):
        return coords


class ReplaceBlackBackground(T.Transform):
    """
    Replaces the background of a simple model simulated black BG image with a new_bg

    IMP: This BG replacement only works if any images with a black background are stored as npy/bmp format
    with compressed formats bg replacement like this fails due to compression artifcats
    """
    def __init__(self, new_bg: np.ndarray):
        self.new_bg = new_bg
        # For debugging augmentations
        self.display_output = False

    def apply_image(self, img: np.ndarray):
        # If any channel of a certain pixel is non-zero in original image
        # then we want to retain that pixel otherwise use the RGB pixel from the new bg
        RGB_mask = np.repeat(np.any(img > 10, axis=2)[:, :, None], 3, axis=2)
        augmented = np.where(RGB_mask, img, self.new_bg)
        if self.display_output:
            display_image_channel_wise(augmented=augmented, original=img, augmentation=self.new_bg)
        return augmented

    # Do nothing to the segmentations
    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    # Apply no transform to coordinates
    def apply_coords(self, coords: np.ndarray):
        return coords


class MyBaseAug(T.Augmentation):
    def __init__(self, aug_data: ImageDataDir):
        super().__init__()
        self.dir_manager = ResultDirManager()
        # List all file names of relevant extension in folder
        self.image_files = self.dir_manager.list_dir_objects(aug_data.folder, aug_data.extension)

    def get_transform(self, *args):
        return T.NoOpTransform


class FgTexture(MyBaseAug):
    """
    Replaces the non-black foreground simple model texture with an arbitrary texture from
    paint in numbers textures dataset
    """
    def get_transform(self, image):
        """
        :param image:
        :return: T.transform object that does the actual bg replacement
        """
        new_fg_path = random.choice(self.image_files)
        # Read in resized bg into RGB numpy array
        #  PIL size convention is (w, h)
        new_fg_img = np.array(Image.open(new_fg_path).resize((image.shape[1], image.shape[0])))

        # Toss a coin to decide whether or not to apply augmentation
        coin = random.randint(0, 1)
        if coin:
            return ReplaceForeground(new_fg_img)
        else:
            return T.NoOpTransform()


class ImagenetBgAugmentation(MyBaseAug):
    """
    Assumes input images have a black background and attempts to replace this background with an Imagenet img
    """
    def get_transform(self, image):
        """
        :param image:
        :return: T.transform object that does the actual bg replacement
        """
        new_bg_path = random.choice(self.image_files)
        # Read in resized bg into RGB numpy array
        new_bg_img = np.array(Image.open(new_bg_path).resize((image.shape[1], image.shape[0])))

        # Toss a coin to decide whether or not to apply augmentation
        coin = random.randint(0, 1)
        if coin:
            return ReplaceBlackBackground(new_bg_img)
        else:
            return T.NoOpTransform()


class SimGeomAugmentation(MyBaseAug):
    """
    Assumes input images have a black background and attempts to replace this background
    Background is replaced by arbitrary task agnostic geometries of the kind we would see in a simulated env
    Example: Other Simple Models, Arbitrary Outline Shapes, Arbitrary Filled Shapes
    """
    def get_transform(self, image):
        """
        :param image:
        :return: T.transform object that does the actual bg replacement
        """
        new_geom_path = random.choice(self.image_files)
        # Read in resized bg into RGB numpy array
        new_geom_img = np.load(new_geom_path)

        # Toss a coin to decide whether or not to apply augmentation
        coin = random.randint(0, 1)
        if coin:
            return ReplaceBlackBackground(new_geom_img)
        else:
            return T.NoOpTransform()


class AppliedAugmentations:
    """
    List of implemented data augmentations actually applied at run time
    """
    def __init__(self, cfg):
        augs_dirs = AugmentationDataDirs()
        self.aug_list = []

        # Applies a novel texture to the geometry of the simple model with p = 0.5
        self.fg_aug = FgTexture(augs_dirs.textures)

        # Simple model augmentations as distractors each with p=0.5
        self.ball_aug = SimGeomAugmentation(augs_dirs.ball)
        self.cartpole_aug = SimGeomAugmentation(augs_dirs.cartpole)
        self.dcartpole_aug = SimGeomAugmentation(augs_dirs.dcartpole)
        self.dubins_aug = SimGeomAugmentation(augs_dirs.dubins)

        # Add a filled shape to the bg with p = 0.5
        self.filled_shape_aug = SimGeomAugmentation(augs_dirs.filled_shapes)

        # Add an outline shape to the bg with p = 0.5
        self.outline_shape_aug = SimGeomAugmentation(augs_dirs.outline_shapes)

        # Apply a random bg from image net dataset
        self.image_net = ImagenetBgAugmentation(augs_dirs.image_net)

        # Sequence in which these augmentations are applied is crucial
        if cfg.FG_AUG:
            self.aug_list.append(self.fg_aug)
        if cfg.SIMP_MOD_AUG:
            if cfg.SIMP_MOD != 'ball':
                self.aug_list.append(self.ball_aug)
            if cfg.SIMP_MOD != 'cartpole':
                self.aug_list.append(self.cartpole_aug)
            if cfg.SIMP_MOD != 'dcartpole':
                self.aug_list.append(self.dcartpole_aug)
            if cfg.SIMP_MOD != 'dubins':
                self.aug_list.append(self.dubins_aug)
        if cfg.SHAPE_AUG:
            self.aug_list.append(self.filled_shape_aug)
            self.aug_list.append(self.outline_shape_aug)
        if cfg.BG_IMGNET:
            self.aug_list.append(self.image_net)
