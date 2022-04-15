"""
Methods for handling the frames generated from running the simple model environments
"""
import numpy as np
import math
import os
from PIL import Image
import random
from src.config import SegConfig
from src.utils.results_dir_manager import ResultDirManager
from typing import Tuple


class FramesHandler:
    def __init__(self, path_to_images: str = None, path_to_masks: str = None, simp_model: str = None, nframes: int = None):
        self.path_to_images = path_to_images
        self.path_to_masks = path_to_masks
        self.dir_manager = ResultDirManager()
        if self.path_to_masks is not None:
            self.dir_manager.add_location('masks', self.path_to_masks, make_dir_if_none=False)

        # Simple model for which handling these frames
        self.simp_model = simp_model

        # Seg config object for image size
        seg_config = SegConfig()
        # Be aware of the image size (single square frame)
        self.seg_size = seg_config.imsize

        # Blank black and white frames for making masks
        self.black = np.zeros((self.seg_size, self.seg_size), dtype=np.uint8)
        self.white = 255 * np.ones((self.seg_size, self.seg_size), dtype=np.uint8)

        self.PIL_nhstack = None
        self.PIL_nvstack = None

        # Whether test or train dataset
        self.dataset_type = None

        # Number of frames per observation. A single frame is a 3 channel RGB image
        self.nframes = nframes

    def set_dataset_type(self, dataset_type: str):
        self.dataset_type = dataset_type

    def set_PIL_stack_sizes(self, nhstack: int, nvstack: int):
        """
        Sets the number of indiviual frames to be stacked in horizontal and vertical direction
        when saving an np array as a PIL image
        :param nhstack:
        :param nvstack:
        :return:
        """
        self.PIL_nhstack = nhstack
        self.PIL_nvstack = nvstack

    def mask_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Takes in a single images and makes and returns a mask out of it
        :param image:
        :return: uint8 mask
        """
        mask = np.where(image.sum(axis=2) != 0, self.white, self.black)
        return mask

    def save_trajectory(self, traj_observations: list, traj_actions, traj_states, traj_idx: int) -> str:
        """
        Save handler for making encoder traning dataset as trajectories
        :param traj_observations:
        :return: traj_dir_path
        """
        # Create a dir to store observations from this trajectory
        traj_dir_path = self.dir_manager.make_dir_from_dict('cur_dataset', {'_traj': traj_idx + 1},
                                                        prefix=self.dataset_type)

        # Save trajectory, states, and actions to separate .npy files
        obs_path = traj_dir_path + '/traj_observations.npy'
        obs_path_jpeg = traj_dir_path + '/traj_observations.jpeg'
        actions_path = traj_dir_path + '/traj_actions.npy'
        states_path = traj_dir_path + '/traj_states.npy'
        if self.nframes == 1:
            # Convert lists of floats/np arrays to a single np array
            obs_npy = np.array(traj_observations)
        elif self.nframes == 2:
            # Find the shape of one side by side stacked rectangle frame
            test_obs = traj_observations[0]
            # Find the length the shorter side, longer will simply be 2x long
            imsize = test_obs.shape[0]
            # Convert side by side stacked RGB images to a square 6 channel image
            traj_observations_reshaped = []
            for traj_obs in traj_observations:
                # Left frame is view at t-1
                traj_obs_t_minus1 = traj_obs[:, :imsize, :]
                # Right frame is view at t
                traj_obs_t = traj_obs[:, imsize:, :]
                # Stack these 2 frames along channels dimension to make a 6 channel image
                traj_obs_reshaped = np.dstack((traj_obs_t_minus1, traj_obs_t))
                traj_observations_reshaped.append(traj_obs_reshaped)
            obs_npy = np.array(traj_observations_reshaped)
        else:
            raise ValueError("Invalid nframes {0}".format(self.nframes))
        # Use the original 2 channel version for cerating accompanying PIL/jpeg visualization=
        obs_PIL = self.traj_frame_list_to_PIL_image(traj_observations)
        actions_npy = np.array(traj_actions)
        states_npy = np.array(traj_states)
        # Save arrays and PIL to disk
        np.save(obs_path, obs_npy)
        obs_PIL.save(obs_path_jpeg)
        np.save(actions_path, actions_npy)
        np.save(states_path, states_npy)
        return traj_dir_path

    def save_frame(self, frame: np.ndarray, unique_id: int):
        """
        Saved an pbservation as an image
        :param frame:
        :param unique_id:
        :return:
        """
        # Save image at appropriate location
        obs_PIL = Image.fromarray(frame)
        img_path = self.path_to_images + "/{0}.npy".format(unique_id)
        img_path_jpeg = self.path_to_images + "/{0}.jpeg".format(unique_id)
        obs_PIL.save(img_path_jpeg)
        np.save(img_path, frame)
        # save success
        return True

    def save_1frame(self, cur_obs: np.ndarray, unique_id: int) -> bool:
        saved = False
        # Save mask
        mask_candidate = self.mask_from_image(cur_obs)
        # Check if mask is blank before saving
        if np.any(mask_candidate):
            self.save_mask_1frame(mask_candidate, unique_id)
            # Save mask and image at appropriate locations
            self.save_frame(cur_obs, unique_id)
            # Image saved, increment unique id
            saved = True

        return saved

    def save_2frame1mask(self, prev_obs: np.ndarray, cur_obs: np.ndarray, unique_id: int) -> bool:
        saved = False

        # concat prev and next frame if prev_obs is not None
        if prev_obs is not None:
            # Make masks seperately
            prev_mask = self.mask_from_image(cur_obs)
            cur_mask = self.mask_from_image(prev_obs)
            # Neither half of concatenated mask should be blank if saving mask as an annotation and an image
            if np.any(prev_mask) and np.any(cur_mask):
                self.save_mask_2frame1mask(prev_mask, cur_mask, unique_id)

                concat_obs = self.concat(prev_obs, cur_obs, white_line=True)
                self.save_frame(concat_obs, unique_id)
                saved = True
        return saved

    def save_2frame2mask(self, prev_obs: np.ndarray, cur_obs: np.ndarray, unique_id: int) -> bool:
        saved = False

        # concat prev and next frame if prev_obs is not None
        if prev_obs is not None:
            concat_obs = self.concat(prev_obs, cur_obs, white_line=True)

            # Make masks seperately
            prev_mask = self.mask_from_image(cur_obs)
            cur_mask = self.mask_from_image(prev_obs)
            # Neither half of concatenated mask should be blank if saving mask as an annotation and an image
            if np.any(prev_mask) and np.any(cur_mask):
                self.save_mask_2frame2mask(prev_mask, cur_mask, unique_id)

                self.save_frame(concat_obs, unique_id)
                saved = True
        return saved

    def save_mask_1frame(self, mask: np.ndarray, unique_id: int):
        """
        Save a single mask with a single annotation for the case when we save
        single pixel obs at time {t}
        """
        # Create PIL image out of np array mask
        mask_PIL = Image.fromarray(mask)
        # Annotation index remains fixed at 0 for single frames with single simple models
        anno_idx = 0
        # Save the mask(s) for this image
        mask_path_png = self.path_to_masks + "/{0}_{1}_{2}.png".format(unique_id, self.simp_model, anno_idx)
        mask_path = self.path_to_masks + "/{0}_{1}_{2}.npy".format(unique_id, self.simp_model, anno_idx)

        # Save mask
        np.save(mask_path, mask)
        mask_PIL.save(mask_path_png)

    def save_mask_2frame1mask(self, prev_mask: np.ndarray, mask: np.ndarray, unique_id: int):
        """
        Save a single mask with two disjoint regions from the concatentation of mask1 and mask2
        single pixel obs at time {t}
        """
        mask = self.concat(prev_mask, mask)
        # Create PIL image out of np array mask
        mask_PIL = Image.fromarray(mask)
        # Annotation index remains fixed at 0 for 2 frames with single disjoint mask of 2 pictured simp model
        anno_idx = 0
        # Save the mask(s) for this image
        mask_path_png = self.path_to_masks + "/{0}_{1}_{2}.png".format(unique_id, self.simp_model, anno_idx)
        mask_path = self.path_to_masks + "/{0}_{1}_{2}.npy".format(unique_id, self.simp_model, anno_idx)

        # Save mask
        np.save(mask_path, mask)
        mask_PIL.save(mask_path_png)

    def save_mask_2frame2mask(self, prev_mask: np.ndarray, mask: np.ndarray, unique_id: int):
        """
        Save a single mask with two disjoint regions from the concatentation of mask1 and mask2
        single pixel obs at time {t}
        """
        mask1 = self.concat(prev_mask, self.black)
        mask2 = self.concat(self.black, mask)
        # Create PIL image out of np array mask
        mask1_PIL = Image.fromarray(mask1)
        mask2_PIL = Image.fromarray(mask2)
        # @ Annotations for the single image with a left half mask and a right half mask
        anno_idx = 0

        # Save mask 1
        mask_path_png = self.path_to_masks + "/{0}_{1}_{2}.png".format(unique_id, self.simp_model, anno_idx)
        mask_path = self.path_to_masks + "/{0}_{1}_{2}.npy".format(unique_id, self.simp_model, anno_idx)
        np.save(mask_path, mask1)
        mask1_PIL.save(mask_path_png)

        anno_idx = 1
        # Save mask 2
        mask_path_png = self.path_to_masks + "/{0}_{1}_{2}.png".format(unique_id, self.simp_model, anno_idx)
        mask_path = self.path_to_masks + "/{0}_{1}_{2}.npy".format(unique_id, self.simp_model, anno_idx)
        np.save(mask_path, mask2)
        mask2_PIL.save(mask_path_png)

    def merge_images(self, image_fnames: list):
        img = np.load(os.path.join(self.path_to_images, image_fnames[0]))
        for idx in range(1, len(image_fnames)):
            next_image = np.load(os.path.join(self.path_to_images, image_fnames[idx]))
            # If any channel of a certain pixel is non-zero in original image
            # then we want to retain that pixel otherwise use the RGB pixel from next image
            RGB_mask = np.repeat(np.any(img > 10, axis=2)[:, :, None], 3, axis=2)
            img = np.where(RGB_mask, img, next_image)
        return img

    def get_mask_fnames(self, unique_id: str):
        # Create prefix for globbing
        pattern = unique_id + '_*.npy'
        # Return full path to masks
        return self.dir_manager.scrape_loc_for_glob('masks', pattern=pattern)

    def combine_images(self, max_combine: int, nnew_images: int, initial_unique_id: int):
        # List images available in images folder before adding new combined images
        images_list = self.dir_manager.list_dir_objects(self.path_to_images, '*.npy')

        # Check if max_combineis atleast 2
        if max_combine < 1:
            raise ValueError("max_combine should be an int greater than 1")

        # New unique ids allocated to combined images
        uid = initial_unique_id

        for idx in range(nnew_images):
            # Number of images to combine in this iteration
            ncombine = random.randint(2, max_combine)
            combine_us = []
            masks_dict = {}
            nadded = 0
            while nadded < ncombine:
                new_fname = random.choice(images_list)
                # Take out without replacement to prevent overlapping objects
                if new_fname not in combine_us:
                    combine_us.append(new_fname)
                    # Get the unique_id of this image as a string
                    unique_id = self.dir_manager.get_name_sans_extension_from_path(new_fname)
                    # Get the file_names of masks corresponding to this image
                    masks = self.get_mask_fnames(unique_id)
                    masks_dict[new_fname] = masks
                    nadded += 1

            # Get superimposed image by treating the images like they were in increasing
            #  order of depth
            combined_frame = self.merge_images(combine_us)
            combined_PIL = Image.fromarray(combined_frame)
            new_pth = self.path_to_images + "/{0}.npy".format(uid)
            new_pth_jpeg = self.path_to_images + "/{0}.jpeg".format(uid)
            print("Saving a combined frame and annotation ... uid {0}".format(uid))
            np.save(new_pth, combined_frame)
            combined_PIL.save(new_pth_jpeg)
            if uid % 50 == 0:
                print("Created a combined image with uid {0}".format(uid))
            # load the masks and save with name that reflects them being from new combined image
            instance_idx = 0
            for fname in combine_us:
                fname_masks = masks_dict[fname]
                for mask_pth in fname_masks:
                    mask_fname = self.dir_manager.get_name_sans_extension_from_path(mask_pth)
                    simple_model = mask_fname.split('_')[1]
                    mask_np = np.load(mask_pth)
                    mask_PIL = Image.fromarray(mask_np)
                    np_pth = self.path_to_masks + "/{0}_{1}_{2}.npy".format(uid, simple_model, instance_idx)
                    png_pth = self.path_to_masks + "/{0}_{1}_{2}.png".format(uid, simple_model, instance_idx)
                    np.save(np_pth, mask_np)
                    mask_PIL.save(png_pth)
                    instance_idx += 1
            uid += 1

    def concat(self, o1: np.ndarray, o2: np.ndarray, white_line: bool = False) -> np.ndarray:
        """
        Take in 2 H x W x C frames and return a single H x 2W x C image
        Adds a 1/64 pixel wide white line as a visual cue to aid accurate segmentation at test time
        :param o1: Frame 1
        :param o2: Frame 2
        :return:
        """
        stacked = np.hstack((o1, o2))
        # o1 and o2 assumed to be single square frames
        img_size = o1.shape[0]
        # White line procedure assumes dtype of stacked is np.uint8
        if white_line:
            # Number of pixels wide given 1 pixel when 64x64 image
            npixels = img_size // 64
            # Pixel range to set to 255
            stacked[:, img_size - math.ceil(npixels / 2):img_size + npixels // 2, :] = 255
        return stacked

    def traj_frame_list_to_PIL_image(self, frames_list: list):
        """
        Converts a list of frames frames_list into a single large PIL image of size hstack x vstack
        :param frames_list:
        :return:
        """
        # Split the list of frames into self.vstack distinct rows
        frames_list_split = [frames_list[idx:idx+self.PIL_nhstack] for idx in range(0, len(frames_list), self.PIL_nhstack)]
        # frames list to be vstacked after the hstacking of frames in a row is complete
        frames_list_hstacked = []
        for f_l in frames_list_split:
            frames_list_hstacked.append(np.hstack(f_l))
        stacked = np.vstack(frames_list_hstacked)
        return Image.fromarray(stacked)


def nsd(n: int) -> Tuple[int, int]:
    """
    Return the most square like divisors/factors of the number n
    src: https://stackoverflow.com/a/39248503/3642162
    :param n:
    :return:
    """
    val1 = math.ceil(math.sqrt(n))
    val2 = int(n / val1)
    while val2 * val1 != float(n):
        val1 -= 1
        val2 = int(n / val1)
    # Return sorted divisors, longer than shorter
    ret_val1 = max(val1, val2)
    ret_val2 = val1 + val2 - ret_val1
    return ret_val1, ret_val2
