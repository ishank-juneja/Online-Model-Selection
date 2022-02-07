from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import numpy as np
from torchvision import transforms
import torch
import re


# Wrapper class around ImageTrajectoryDataset class
class MyDatasetBuilder:
    def __init__(self):
        pass

    def tryint(self, s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(self, s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [self.tryint(c) for c in re.split('([0-9]+)', s)]

    def sort_nicely(self, l):
        """ Sort the given list in the way that humans expect.
        """
        l.sort(key=self.alphanum_key)

    # Returns pytorch DataLoader compatible object
    def get_dataset(self, data_config, data_identifier: str):
        dataset_dir = data_config.data_dir
        # List out all the traj dirs in this folder
        dirnames = os.listdir(dataset_dir)
        self.sort_nicely(dirnames)
        # Lists to hold full datasets
        all_obs_frame_files = []
        all_actions = []
        all_states = []
        # Iterate over the traj specific folder present inside
        for dirname in dirnames:
            if data_identifier in dirname:
                cur_dir_path = os.path.join(dataset_dir, dirname)
                dir_filenames = os.listdir(cur_dir_path)
                self.sort_nicely(dir_filenames)
                # List to hold file names of observations in sorted order
                obs_frame_files = []
                for filename in dir_filenames:
                    filepath = os.path.join(cur_dir_path, filename)
                    if 'observation' in filename:
                        obs_frame_files.append(filepath)
                    elif 'actions' in filename:
                        all_actions.append(filepath)
                    elif 'states' in filename:
                        all_states.append(filepath)
                all_obs_frame_files.append(obs_frame_files)
        return ImageTrajectoryDataset(all_actions, all_states, all_obs_frame_files, data_config)


# Class in the standard pytorch dataloader format overriding __getitem__ and __len__
class ImageTrajectoryDataset(Dataset):
    def __init__(self, action_filenames, state_filenames, obs_filenames, data_config):
        self.data_config = data_config
        self.image_filenames = obs_filenames
        self.state_filenames = state_filenames
        self.action_filenames = action_filenames
        # Determine parameters of the dataset
        # Assumption: Every trajectory is the same number of frames
        self.ntraj = len(self.image_filenames)
        self.traj_len = len(self.image_filenames[0])
        # Load a single image to get image shape
        tmp_frame = np.load(self.image_filenames[0][0])
        self.imsize, _, self.nchannels = tmp_frame.shape

    def preprocess_imgs(self, imgs):
        # Preprocess grayscale images
        if self.data_config.grey:
            preprocess_img = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Grayscale(num_output_channels=1),
                                             transforms.Resize((self.data_config.imsize, self.data_config.imsize)),
                                             transforms.ToTensor()
                                             ])
        # Preprocess RGB images
        else:
            preprocess_img = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((self.data_config.imsize, self.data_config.imsize)),
                                             transforms.ToTensor()
                                             ])
        # make preprocess_img operate on individual frames
        processed_imgs = torch.stack([preprocess_img(img) for img in imgs], 0)
        # The -1 means infer from other dimensions
        processed_imgs = processed_imgs.view(self.traj_len, -1, self.imsize, self.imsize)
        return processed_imgs

        # Get a batch of trajectories

    def npy_loader(self, path):
        sample = torch.from_numpy(np.load(path))
        return sample

    # Return a single traj of observations, one traj is one datapoint in the training loop
    # TODO: Can use torchvision.DatasetFolder later
    def __getitem__(self, item):
        # Allocate a tensor for sticking together loaded traj image frames
        loaded_frames = torch.zeros((self.traj_len, self.imsize, self.imsize, self.nchannels))
        traj_image_filenames = self.image_filenames[item]
        # Iterate over image filenames for this traj
        for frame_idx, filepath in enumerate(traj_image_filenames):
            loaded_frames[frame_idx, :, :, :] = self.npy_loader(filepath)
        loaded_actions = self.npy_loader(self.action_filenames[item])
        loaded_states = self.npy_loader(self.state_filenames[item])
        # Change the sequence of tensor dims
        loaded_frames = loaded_frames.permute(0, 3, 2, 1)
        # Preprocess traj of images
        loaded_frames = self.preprocess_imgs(loaded_frames)
        return loaded_frames.float(), loaded_states.float(), loaded_actions.float()

    # Return total number of trajectories, one traj is one datapoint in the training loop
    def __len__(self):
        return self.ntraj


def preprocess_identity(states):
    return states


# Test dataset_builder
if __name__ == '__main__':
    from src.config.cartpole import Config
    config = Config()
    data_config = config.data_config
    dataset_builder = MyDatasetBuilder()
    dataset = dataset_builder.get_dataset(data_config, 'train')

    img, states, actions = dataset[0]

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, drop_last=True)

    for i in range(len(dataset)):
        imgs, states, actions = dataset[i]
        print(imgs.size())
        print(actions.size())
        print(states.size())

    for i, batch in enumerate(loader):
        imgs = batch[0]
        states = batch[1]
        actions = batch[2]
        print(imgs.size())
        print(states.size())
        print(actions.size())


    print(loader.batch_size)
    print(len(loader.dataset))
    print(len(loader.dataset) // loader.batch_size)
