from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import numpy as np
from torchvision import transforms
import torch
import re
from src.quaternion import qeuler, euler_to_quaternion
import cv2
import matplotlib.pyplot as plt


def preprocess_victor_observations(observations):
    N, T, w, h, c = observations.shape
    observations = observations.float()
    #observations[:, :, :, :, 3] /= 2.6 #torch.max(observations[:, :, :, :, 3])

    print(torch.max(observations[:, :, :, :, 3]))
    print(torch.min(observations[:, :, :, :, 3]))
    print(torch.max(observations[:, :, :, :, :3]))

    return observations

def preprocess_mujoco_cartpole_state(states, N):
    new_states = torch.zeros(states.shape[0], states.shape[1], 5)
    new_states[:, :, 0] = -states[:, :, 0]
    theta = torch.atan2(states[:, :, 1], states[:, :, 2]) - np.pi
    new_states[:, :, 1] = torch.sin(theta)
    new_states[:, :, 2] = torch.cos(theta)
    new_states[:, :, 3] = - states[:, :, 3]
    new_states[:, :, 4] = states[:, :, 4]
    return new_states

def preprocess_identity(states):
    return states

def preprocess_cartpole_states(states):

    # Non mujoco data
    new_states = torch.zeros(states.shape[0], states.shape[1], 5)
    new_states[:, :, 2] = states[:, :, 2]
    new_states[:, :, 2] = torch.cos(states[:, :, 2] + np.pi)
    new_states[:, :, 3] = torch.sin(states[:, :, 2] + np.pi)
    new_states[:, :, 4] = states[:, :, 3]

    return new_states


def preprocess_two_link_rope_states(states):

    return states.float()


def preprocess_one_link_rope_states(states):
    print(states[:, :, 5].min())
    return states
    '''
        zeros out rotation about x axis since rotationally invariant
    '''
    p = states[:, :, :3]
    q = states[:, :, 3:].contiguous()

    euler_angles = qeuler(q, 'zyx')
    # Set x axis rotation to be zero (invariant to this)
    euler_angles[:, :, 0] = 0.
    # Convert back to quaternions
    q = torch.from_numpy(euler_to_quaternion(euler_angles.numpy(), 'zyx'))

    return torch.cat((p, q), dim=2)


def image_augmentation(image):
    im = image[0].permute(2, 1, 0)
    numpy_im = im.numpy()
    numpy_rgb = numpy_im[:, :, :3]
    numpy_depth = numpy_im[:, :, 3]
    '''
    plt.imshow(numpy_rgb)
    plt.show()
    plt.imshow(numpy_depth)
    plt.show()
    '''


    # Get to btween 0 and 255
    #depth = numpy_depth + np.min(numpy_depth)
    synthetic_data = True
    if synthetic_data:
        #depth /= np.max(depth)
        depth = 255.0 * numpy_depth
        depth = depth.astype(np.uint8)
        edges = cv2.Canny(depth, 100, 200)
        ex, ey = np.where(edges == 255)
        num_edge = len(ex)
        min_edge = int(0.5 * num_edge)
        num_disturbances = np.random.randint(min_edge, num_edge)
        perm = np.random.permutation(np.arange(0, num_edge))
        pixels_to_disturb = perm[:num_disturbances]
        idx = (ex[pixels_to_disturb], ey[pixels_to_disturb])
        eones = np.ones(depth.shape)
        eones[idx] = 0.0

        # Add edge noise
        numpy_depth *= eones

    # Get area where depth beyond threshold (background)
    background_idx = np.where(np.logical_or(numpy_depth > 2/2.6, numpy_depth == 0))
    bones = np.ones(numpy_depth.shape)
    bones[background_idx] = 0.0

    # Salt and pepper noise to background
    bx, by = background_idx
    max_disturbances = int(0.2 * len(bx))
    min_disturbances = int(0.1 * len(bx))
    num_snp = np.random.randint(min_disturbances, max_disturbances)
    b_permutation = np.random.permutation(len(bx))
    snp_idx = (bx[b_permutation[:num_snp]], by[b_permutation[:num_snp]])
    snp_noise_rgb = 0.5 + 0.1 * np.random.randn(num_snp, 3)
    snp_noise_depth = 1.0 + 0.1 * np.random.randn(num_snp)

    # Subtract background
    numpy_depth *= bones
    numpy_rgb *= bones[:, :, None]

    # Add snp noise
    if synthetic_data:
        numpy_depth[snp_idx] = snp_noise_depth
        numpy_rgb[snp_idx] = snp_noise_rgb
    '''
    plt.imshow(numpy_rgb)
    plt.show()
    plt.imshow(numpy_depth)
    plt.show()
    '''
    ret_img = np.concatenate((numpy_rgb, numpy_depth[:, :, None]), axis=2)
    ret_img = torch.from_numpy(ret_img).permute(2, 0, 1)
    ret_img += 0.05 * torch.randn_like(ret_img)
    return ret_img.unsqueeze(0)


class ImageTrajectoryDataset(Dataset):
    def __init__(self, image_filename, state_filename, action_filename, data_config):
        self.data_config = data_config
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
            depth = imgs[:, :, -1].unsqueeze(2)
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


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def dataset_builder(data_config, data_identifier):
    filenames = os.listdir(data_config.data_dir)
    filenames.sort()

    action_files = []
    observation_files = []
    state_files = []
    for filename in filenames:
        if data_identifier in filename:
            if 'actions' in filename:
                action_files.append(os.path.join(data_config.data_dir, filename))
            if 'observations' in filename:
                observation_files.append(os.path.join(data_config.data_dir, filename))
            if 'states' in filename:
                state_files.append(os.path.join(data_config.data_dir, filename))

    datasets = []
    if len(state_files) == 0:
        state_files = [[]] * len(action_files)

    for observation_f, state_f, action_f in zip(observation_files, state_files, action_files):
        datasets.append(ImageTrajectoryDataset(observation_f, state_f, action_f, data_config))

    print(datasets)
    return ConcatDataset(datasets)


if __name__ == '__main__':

    ''' Just testing to see if it works '''
    dataset = dataset_builder('data/trajectories/damped_cartpole', 'npy')

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