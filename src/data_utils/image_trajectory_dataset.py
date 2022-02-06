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
        dataset_dir = os.path.join(data_config.data_dir, data_identifier)
        filenames = os.listdir(dataset_dir)
        self.sort_nicely(filenames)
        observation_files = []
        for filename in filenames:
            if 'actions' in filename:
                action_file_path = os.path.join(data_config.data_dir, filename)
            elif 'states' in filename:
                state_file_path = os.path.join(data_config.data_dir, filename)
            if 'observations' in filename:
                observation_files.append(os.path.join(data_config.data_dir, filename))
        return ImageTrajectoryDataset(action_file_path, state_file_path, observation_files, data_config)


# Class in the standard pytorch dataloader format overriding __getitem__ and __len__
class ImageTrajectoryDataset(Dataset):
    def __init__(self, image_filenames, state_filename, action_filename, data_config):
        self.data_config = data_config
        self.image_filenames = image_filenames
        self.state_filename = state_filename
        self.action_filename = action_filename

        images = np.asarray(np.load(image_filename, allow_pickle=True))
        images = torch.from_numpy(images)
        images = self.data_config.preprocess_obs_fn(images)
        states = torch.from_numpy(np.asarray(np.load(state_filename, allow_pickle=True)))
        self.states = self.data_config.preprocess_state_fn(states)
        self.actions = torch.from_numpy(np.asarray(np.load(action_filename, allow_pickle=True))).float()
        images = images.permute(0, 1, 4, 3, 2)
        N, T, _ = self.actions.shape
        self.input_imsize = images.size()[3]
        self.N = images.size()[0]
        self.T = images.size()[1]
        self.images = self.preprocess_imgs(images)

    def preprocess_imgs(self, imgs):
        if self.data_config.depth:
            imgs = imgs[:, :, :-1]
            #depth += 0.2871
            #depth += torch.randn_like(depth) * 0.0
            # depth = depth.clamp(min=0.0, max=1.0)
            #depth += 0.01 * torch.randn_like(depth)

        if self.data_config.grey:
            preprocess_img = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Grayscale(num_output_channels=1),
                                             transforms.Resize((self.data_config.imsize, self.data_config.imsize)),
                                             transforms.ToTensor()
                                             ])
        else:
            preprocess_img = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((self.data_config.imsize, self.data_config.imsize)),
                                             transforms.ToTensor()
                                             ])

        imgs_flat = imgs.reshape(-1, 3, self.input_imsize, self.input_imsize)
        processed_imgs = torch.stack([preprocess_img(img) for img in imgs_flat], 0)
        # The -1 is probably an openCV thing
        processed_imgs = processed_imgs.view(self.N, self.T, -1, self.data_config.imsize, self.data_config.imsize)
        return processed_imgs

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        image = self.images[item]

        if self.states is None:
            return image, self.actions[item]

        return image.float(), self.states[item].float(), self.actions[item].float()


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
