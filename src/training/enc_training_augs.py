import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from random import randint
from src.config import AugmentationDataDirs, ImageDataDir
from src.utils.results_dir_manager import ResultDirManager
from typing import List


class BaseAugmentation:
    """Base class for all augs"""
    def __init__(self, aug_data: ImageDataDir):
        self.dir_manager = ResultDirManager()
        # List all file names of relevant extension in folder
        self.image_files = self.dir_manager.list_dir_objects(aug_data.folder, aug_data.extension)
        # Whether to display before and after of an aug for debugging
        self.display_output = False

    def replace_fg(self, img_traj: np.ndarray, new_fg: np.ndarray):
        """
        Replaces the foreground of a simple model simulated black BG image traj with a new texture
        :param img_traj: array of shape T x H x W x C (C=3 assumed here for augmentations)
        :param new_fg: array of shape H x W x 3
        :return:
        """
        # Make copies of the augmentation along T axis
        # Extract side length of square frame
        nframes, side, _, _ = img_traj.shape
        new_fg_repeated = np.repeat(new_fg[None], nframes, axis=0)

        # If any channel of a certain pixel is zero in original image
        # then we want to leave as is otherwise we use the RGB pixel from the new FG
        RGB_mask = np.repeat(np.all(img_traj < 1, axis=3)[:, :, :, None], repeats=3, axis=3)
        augmented = np.where(RGB_mask, img_traj, new_fg_repeated)
        return augmented

    def replace_bg(self, img_traj: np.ndarray, new_bg: np.ndarray, keep_front: bool = True):
        """
        Replaces the foreground of a simple model simulated black BG image traj with a new texture
        :param img_traj: Array of shape T x H x W x C (C=3 assumed here for augmentations)
        :param new_bg: Array of shape H x W x 3
        :param keep_front: If true keep image in front, else let it be occluded, only set to False with sim-esque geoms
        :return:
        """
        # Make copies of the augmentation along T axis
        # Extract side length of square frame
        nframes, side, _, _ = img_traj.shape
        new_bg_repeated = np.repeat(new_bg[None], nframes, axis=0)

        if keep_front:
            # If any channel of a certain pixel is zero in original image
            # then we want to leave as is otherwise we use the RGB pixel from the new BG
            RGB_mask = np.repeat(np.any(img_traj > 1, axis=3)[:, :, :, None], repeats=3, axis=3)
            augmented = np.where(RGB_mask, img_traj, new_bg_repeated)
        else:
            # If any channel of a certain pixel is zero in sim-esque geom bg
            # then we want to leave as is otherwise we use the RGB pixel from the actual image traj
            RGB_mask = np.repeat(np.any(new_bg_repeated > 1, axis=3)[:, :, :, None], repeats=3, axis=3)
            augmented = np.where(RGB_mask, new_bg_repeated, img_traj)
        return augmented

    @staticmethod
    def coin_toss():
        return randint(0, 1)


class FgTexture(BaseAugmentation):
    """
    Replaces the non-black foreground simple model texture with an arbitrary texture from
    paint in numbers textures dataset
    """
    def __call__(self, img_traj: np.ndarray) -> np.ndarray:
        if self.coin_toss():
            # Infer shape of single image in img_traj (assumed square)
            _, imsize, _, _ = img_traj.shape
            # Sample a new augmentation image
            new_fg_path = random.choice(self.image_files)
            # Read in resized bg into RGB numpy array
            #  PIL size convention is (w, h), we deal with only square images here
            new_fg_img = np.array(Image.open(new_fg_path).resize((imsize, imsize)))

            augmented = self.replace_fg(img_traj, new_fg_img)
        else:
            augmented = img_traj
        return augmented


class ImagenetBgAugmentation(BaseAugmentation):
    """
    Assumes input images have a black background and attempts to replace this background with an Imagenet img
    """
    def __call__(self, img_traj: np.ndarray) -> np.ndarray:
        if self.coin_toss():
            # Infer shape of single image in img_traj (assumed square)
            _, imsize, _, _ = img_traj.shape
            # Sample a new augmentation image
            new_bg_path = random.choice(self.image_files)
            # Read in resized bg into RGB numpy array
            #  PIL size convention is (w, h), we deal with only square images here
            new_bg_img = np.array(Image.open(new_bg_path).resize((imsize, imsize)))

            augmented = self.replace_bg(img_traj, new_bg_img)
        else:
            augmented = img_traj
        return augmented


class SimGeomAugmentation(BaseAugmentation):
    """
    Assumes input images have a black background and attempts to replace this background
    Background is replaced by arbitrary task agnostic geometries of the kind we would see in a simulated env
    Example: Other Simple Models, Arbitrary Outline Shapes, Arbitrary Filled Shapes
    """
    def __call__(self, img_traj: np.ndarray) -> np.ndarray:
        if self.coin_toss():
            # Infer shape of single image in img_traj (assumed square)
            _, imsize, _, _ = img_traj.shape
            # Sample a new augmentation image
            new_geom_path = random.choice(self.image_files)
            # Read in resized bg into RGB numpy array
            #  PIL size convention is (w, h), we deal with only square images here
            new_geom_img = np.load(new_geom_path)
            # Make the shape of distractor and training image the same
            # Assumed square image
            size, _, _ = new_geom_img.shape
            # Parameters setup so that size % imsize == 0
            if size % imsize == 0:
                # Get downsampled variant of image
                ds_ratio = size // imsize
                new_geom_img_ds = new_geom_img[::ds_ratio, ::ds_ratio, :]
            else:
                raise ValueError("Size {0} of distractor image and size {1} of "
                                 "training data are incompatible".format(size, imsize))
            if self.coin_toss():
                augmented = self.replace_bg(img_traj=img_traj, new_bg=new_geom_img_ds, keep_front=True)
            else:
                augmented = self.replace_bg(img_traj=img_traj, new_bg=new_geom_img_ds, keep_front=False)
        else:
            augmented = img_traj
        return augmented


class SaltPepperNoiseAug:
    """
    Randomly makes pixels one of Red, Green, Blue, Black, or White
    """
    def __call__(self, img_traj: np.ndarray) -> np.ndarray:
        if self.coin_toss():
            # Extract side length of square frame
            _, imsize, _, _ = img_traj.shape
            # colors to corrupt with: R, G, B, Bl, W
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)]
            # Pick a random number of pixels to currupt between imsize and 2imsize
            ncorruptions = randint(imsize, 2 * imsize)
            # Find ncorruptions number of random pixel locations (same for all frames in traj)
            sampled_colors_idx = np.random.randint(0, len(colors), ncorruptions)
            # Create a copy og trajectory to corrupt
            noisy = img_traj.copy()
            for idx in range(ncorruptions):
                # Sample a pixel
                rand_x, rand_y = np.random.randint(0, imsize, 2)
                noisy[:, rand_x, rand_y, :] = np.array(colors[sampled_colors_idx[idx]])
        else:
            noisy = img_traj
        return noisy

    @staticmethod
    def coin_toss():
        return randint(0, 1)


# Creates multiple task-agnostic domain-randomized versions of an image generated from a simple
# model image from simulation
class ImageTrajectoryAugmenter:
    def __init__(self, excluded_augs: List[str], data_dir: str):
        """
        Augments an entire trajectory of loaded images in one go
        :param excluded_augs: These are a list of implemented augmentations to be excluded from training
        :param data_dir: Data dir from which loading training data to infer simple model
        """
        # Config object for locations of augmentation datasets
        augs_dirs = AugmentationDataDirs()
        # Container for augmentation calllables
        self.aug_list = []

        # Infer simple model for which making data augmenter from datadir
        if 'ball' in data_dir:
            self.simp_model = 'ball'
        elif ('cartpole' in data_dir) and ('dcartpole' not in data_dir):
            self.simp_model = 'cartpole'
        elif 'dcartpole' in data_dir:
            self.simp_model = 'dcartpole'
        elif 'dubins' in data_dir:
            self.simp_model = 'dubins'
        else:
            raise NotImplementedError("simp model with data dir {0} is Not Impl.".format(data_dir))

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

        # Noise snp
        self.noisify = SaltPepperNoiseAug()

        # Sequence in which these augmentations are applied is crucial
        if 'no_fg_texture' not in excluded_augs:
            self.aug_list.append(self.fg_aug)
        if 'no_bg_simp_model' not in excluded_augs:
            if self.simp_model != 'ball':
                self.aug_list.append(self.ball_aug)
            if self.simp_model != 'cartpole':
                self.aug_list.append(self.cartpole_aug)
            if self.simp_model != 'dcartpole':
                self.aug_list.append(self.dcartpole_aug)
            if self.simp_model != 'dubins':
                self.aug_list.append(self.dubins_aug)
        if 'no_bg_shape' not in excluded_augs:
            self.aug_list.append(self.filled_shape_aug)
            self.aug_list.append(self.outline_shape_aug)
        if 'no_bg_imgnet' not in excluded_augs:
            self.aug_list.append(self.image_net)
        if 'no_noise' not in excluded_augs:
            self.aug_list.append(self.noisify)

    def __call__(self, image_trajectory: np.ndarray):
        """
        On every call of the augmenter, we toss a coin to determine whetehr or not to apply a certain augmentation
        Sequence of augmentations is important, FG augmentation can only come before a BG augmentation
        and bg augmentations that have black backgrounds in the augmenting image (simp model and shape)
        can only come before bg augmentations that change the bg (imgnet)
        :return: Image trajectory of same shape as input with augmentations applied
        """
        augmented = image_trajectory
        for apply_aug in self.aug_list:
            augmented = apply_aug(augmented)
        return augmented


if __name__ == '__main__':
    noisifier = SaltPepperNoiseAug()

    noisy = noisifier(np.zeros((20, 64, 64, 3), dtype=np.uint8))

    plt.imshow(noisy[0])
    plt.show()
    print(noisy.shape)
