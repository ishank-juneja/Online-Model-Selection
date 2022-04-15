from torch.utils.data import Dataset, DataLoader, RandomSampler
import os
import numpy as np
from src.simp_mod_datasets import nsd
from src.training.enc_training_augs import ImageTrajectoryAugmenter
import torch
from typing import List
import re
from matplotlib import pyplot as plt


# Wrapper class around ImageTrajectoryDataset class
class MyDatasetBuilder:
    def __init__(self, config, excluded_augs: List[str]):
        self.data_config = config.data_config
        self.excluded_augs = excluded_augs

    @staticmethod
    def sort_nicely(file_list: List[str]):
        """
        Sort the given list in the way that humans expect.
        """
        def tryint(s):
            try:
                return int(s)
            except:
                return s

        def alphanum_key(s):
            """
            Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
            """
            return [tryint(c) for c in re.split('([0-9]+)', s)]

        file_list.sort(key=alphanum_key)

    # Returns pytorch DataLoader compatible object
    def get_dataset(self, dataset_type: str):
        dataset_dir = self.data_config.data_dir
        # List out all the traj dirs in this folder
        dirnames = os.listdir(dataset_dir)
        self.sort_nicely(dirnames)
        # Lists to hold full datasets
        all_observations = []
        all_actions = []
        all_states = []
        # Iterate over the traj specific folder present inside data_dir
        for dirname in dirnames:
            # Ignore any stray .npy files in there
            if '.npy' in dirname:
                continue
            if dataset_type in dirname:
                cur_dir_path = os.path.join(dataset_dir, dirname)
                dir_filenames = os.listdir(cur_dir_path)
                for filename in dir_filenames:
                    # Ignore anything that is not a .npy file
                    if '.npy' not in filename:
                        continue
                    filepath = os.path.join(cur_dir_path, filename)
                    if 'observations' in filename:
                        all_observations.append(filepath)
                    elif 'actions' in filename:
                        all_actions.append(filepath)
                    elif 'states' in filename:
                        all_states.append(filepath)
        return ImageTrajectoryDataset(all_actions, all_states, all_observations, self.data_config,
                                      excluded_augs=self.excluded_augs)


def display_image_traj_channel_wise(augmented: np.ndarray, original: np.ndarray):
    """
    For debugging data augmentations
    :return:
    """
    # Debug data-type compatibility issues between images being combined
    print("Original image data type".format(original.dtype))
    print("Augmented image data type".format(augmented.dtype))

    # Get number of images and find nearest square divisors for viz
    n, _, _, _ = augmented.shape
    # a >= b ensured
    a, b = nsd(n)
    print(a, b)
    img_rows_lst = []
    for idx in range(b):
        img_row = np.hstack(augmented[a*idx:a*(idx+1)])
        print(img_row.shape)
        img_rows_lst.append(img_row)
    augmented = np.vstack(img_rows_lst)
    print(augmented.shape)

    # augmented = np.hstack(augmented[])
    original = np.hstack(original)

    # Visually inspect the combined image
    fig = plt.figure(figsize=(20, 5))
    ax = fig.subplots(1, 1)
    ax.imshow(augmented)
    # ax[0].imshow(augmented)
    # Inspect all channels separately
    # for idx in range(1, 4):
    #     ax[idx].imshow(augmented[:, :, idx - 1])
    plt.show()


class ImageTrajectoryDataset(Dataset):
    """
    Create an iterable dataset object in torch Dataset format
    """
    def __init__(self, action_filenames, state_filenames, obs_filenames, data_config, excluded_augs: List[str]):
        self.data_config = data_config
        self.image_filenames = obs_filenames
        self.state_filenames = state_filenames
        self.action_filenames = action_filenames
        self.augmenter = ImageTrajectoryAugmenter(excluded_augs=excluded_augs, data_dir=data_config.data_dir)
        # Determine parameters of the dataset
        # Assumption: Every trajectory is the same number of frames
        self.ntraj = len(self.image_filenames)
        # Load a single traj to get traj_len and image shape
        tmp_traj = np.load(self.image_filenames[0])
        self.traj_len = len(tmp_traj)
        self.imsize, _, nchannels = tmp_traj[0].shape
        # Enfore that self.nchannels be a multiple of 3
        if nchannels % 3 != 0:
            raise ValueError("Number of channels in training images must be a multiple of 3, found {0}".format(self.nchannels))
        else:
            self.nchannels = nchannels

        # Whether to display trajectories being loaded in
        self.display = False

    def preprocess_imgs(self, imgs):
        # # Preprocess grayscale images
        # preprocess_img = transforms.Compose([transforms.ToPILImage(),
        #                                  transforms.Resize((self.data_config.imsize, self.data_config.imsize)),
        #                                  transforms.ToTensor()
        #                                  ])
        # # make preprocess_img operate on individual frames
        # processed_imgs = torch.stack([preprocess_img(img) for img in imgs], 0)
        #
        # # The -1 means infer from other dimensions
        # processed_imgs = processed_imgs.view(self.traj_len, -1, self.imsize, self.imsize)
        # return processed_imgs
        return imgs

    def npy_loader(self, path):
        sample = torch.from_numpy(np.load(path))
        return sample

    # Return a single traj of observations, one traj is one datapoint in the training loop
    def __getitem__(self, item):
        # dtype of this tensor should be uint8 for transforms.ToPILImage() to work correctly
        loaded_obs = np.load(self.image_filenames[item])
        # Preallocate array to hold augmented version of image traj, zeros_like preserves dtype
        aug_obs = np.zeros_like(loaded_obs)
        # Augment continguos 3 channel trajectory of images seperately and then stitch back together
        for idx in range(self.nchannels // 3):
            aug_obs[..., 3*idx:3*(idx+1)] = self.augmenter(loaded_obs[..., 3*idx:3*(idx+1)])
            if self.display:
                display_image_traj_channel_wise(augmented=aug_obs[..., 3*idx:3*(idx+1)], original=loaded_obs[..., 3*idx:3*(idx+1)])
        aug_obs_pt = torch.from_numpy(aug_obs)
        # Change the sequence of tensor dims
        aug_obs_pt = aug_obs_pt.permute(0, 3, 2, 1)
        # Preprocess traj of images
        aug_obs_prepro = self.preprocess_imgs(aug_obs_pt)
        loaded_actions = self.npy_loader(self.action_filenames[item])
        loaded_states = self.npy_loader(self.state_filenames[item])
        return aug_obs_prepro.float(), loaded_states.float(), loaded_actions.float()

    # Return total number of trajectories, one traj is one datapoint in the training loop
    def __len__(self):
        return self.ntraj


def preprocess_identity(states):
    return states


# Test dataset_builder
if __name__ == '__main__':
    from src.config.cartpole_config import Config
    config = Config()
    data_config = config.data_config
    dataset_builder = MyDatasetBuilder()
    dataset = dataset_builder.get_dataset(data_config, 'train')

    train_sampler = RandomSampler(dataset)

    for data_pt in dataset:
        pass

    train_loader = DataLoader(dataset, sampler=train_sampler,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              drop_last=True)
    idx = 0
    for obs, state, action in train_loader:
        print(obs.shape)
        plt.imshow(obs[0, 5, 0, :, :])
        plt.show()
        print(obs[0, 5, 0, :, :])
        print(state.shape)
        print(action.shape)
        idx += 1
        print(idx)

    # img, states, actions = dataset[0]
    #
    # loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, drop_last=True)
    #
    # for i in range(len(dataset)):
    #     imgs, states, actions = dataset[i]
    #     print(imgs.size())
    #     print(actions.size())
    #     print(states.size())
    #
    # for i, batch in enumerate(loader):
    #     imgs = batch[0]
    #     states = batch[1]
    #     actions = batch[2]
    #     print(imgs.size())
    #     print(states.size())
    #     print(actions.size())
    #
    #
    # print(loader.batch_size)
    # print(len(loader.dataset))
    # print(len(loader.dataset) // loader.batch_size)
