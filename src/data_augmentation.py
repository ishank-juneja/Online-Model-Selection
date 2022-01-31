import numpy as np
import shutil
from math import sqrt
from random import randint
import argparse
from results_dir_manager import ResultDirManager


# Creates multiple task-agnostic domain-randomized versions of an image generated from a simple
# model image from simulation
class SimDomainRandomization:
    def __init__(self):
        pass

    # Function to apply a background augmentation, assumes everything 0 in sim frame is FG and rest is BG
    # Also assumes R=G=B i.e. Black and White images from sim
    # Frame is uint8 type image
    def apply_bag_augmentation(self, frame, new_bg):
        # Find shape of square frame
        frame_size, _, _ = frame.shape
        # Take the thresholded negative of the image
        thresholded_neg_frame = np.zeros_like(frame, dtype=np.float64)
        for idx in range(frame_size):
            for jdx in range(frame_size):
                if frame[idx, jdx, 0] == 0:
                    thresholded_neg_frame[idx, jdx, :] = 1.0
                else:
                    thresholded_neg_frame[idx, jdx, :] = 0.0
        # Get bg frame with fg masked out
        masked = new_bg * thresholded_neg_frame
        masked *= 255.0
        # Convert to unint8 format for additivity with original RGB frame
        masked = masked.astype(int)
        return self.add_salt_pepper(masked + frame)

    # Checkerboard pattern 1 with smaller squares
    def checkerboard_type1(self, frame):
        # Extract side length of square frame
        side, _, _ = frame.shape
        sqrt_side = int(sqrt(side))
        # Create a square of ones and with zeros with sqrt_side size
        one_mini_sq = np.ones((sqrt_side, sqrt_side))
        zero_mini_sq = np.zeros((sqrt_side, sqrt_side))
        # Create a combined 4x4 style block tile out of these minisquares to tile up the bigger square with
        row1 = np.hstack((one_mini_sq, zero_mini_sq))
        row2 = np.hstack((zero_mini_sq, one_mini_sq))
        mytile = np.vstack((row1, row2))
        checkerboard = np.tile(mytile, (sqrt_side // 2, sqrt_side // 2))
        RGB_checkerboard = np.repeat(checkerboard[:, :, np.newaxis], 3, axis=2)
        bg_augmented = self.apply_bag_augmentation(frame, RGB_checkerboard)
        return self.add_salt_pepper(bg_augmented)

    # Checkerboard pattern 2 with larger squares
    def checkerboard_type2(self, frame):
        # Extract side length of square frame
        side, _, _ = frame.shape
        sqrt_side = int(sqrt(side))
        # Create a square of ones and with zeros with sqrt_side size
        one_mini_sq = np.ones((2*sqrt_side, 2*sqrt_side))
        zero_mini_sq = np.zeros((2*sqrt_side, 2*sqrt_side))
        # Create a combined 4x4 style block tile out of these minisquares to tile up the bigger square with
        row1 = np.hstack((one_mini_sq, zero_mini_sq))
        row2 = np.hstack((zero_mini_sq, one_mini_sq))
        mytile = np.vstack((row1, row2))
        checkerboard = np.tile(mytile, (sqrt_side // 4, sqrt_side // 4))
        RGB_checkerboard = np.repeat(checkerboard[:, :, np.newaxis], 3, axis=2)
        bg_augmented = self.apply_bag_augmentation(frame, RGB_checkerboard)
        return self.add_salt_pepper(bg_augmented)

    # Reflect about the major diagonal aka the line y=-x and update state accordingly
    # Not to be used with cartpole
    # Assumes state is a 2D planar coordinate
    def reflect_about_diagonal(self, frame, state):
        transposed_frame = frame.transpose((1, 0, 2))
        new_state = np.array([-state[1], -state[0]])
        return self.add_salt_pepper(transposed_frame), new_state

    # Add salt and pepper noise
    def add_salt_pepper(self, frame):
        # Extract side length of square frame
        side, _, _ = frame.shape
        # Pre-decide that a random number of pixels between 2*side and 10*side are going to be randomly put to 0/255
        rand_255 = randint(side, 10*side)
        rand_0 = randint(side, 10*side)
        # Pre-allocate corrupted frame
        cur_frame = frame.copy()
        for idx in range(rand_255):
            # Get random x and y coordinates (upper limit incl in randint)
            randx = randint(0, side - 1)
            randy = randint(0, side - 1)
            cur_frame[randx, randy] = 255
        for idx in range(rand_0):
            # Get random x and y coordinates (upper limit incl in randint)
            randx = randint(0, side - 1)
            randy = randint(0, side - 1)
            cur_frame[randx, randy] = 0
        return cur_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Name of the simple model on which we do augmentation
    parser.add_argument("--model-name", type=str)
    args = parser.parse_args()
    name = args.model_name
    dir_manager = ResultDirManager()
    # Add location where raw data generated from sim is present
    dir_manager.add_location('raw', 'data/{0}'.format(name))
    # Path to states and observations
    states_pth = dir_manager.get_file_path('raw', 'all_train_states.npy')
    obs_path = dir_manager.get_file_path('raw', 'all_train_observations.npy')
    actions_pth = dir_manager.get_file_path('raw', 'all_train_actions.npy')
    raw_states = np.load(states_pth)
    raw_observations = np.load(obs_path)
    # Find ntraj and traj_len
    ntraj, traj_len, _ = raw_states.shape
    # Create domain randomizer object
    my_randomizer = SimDomainRandomization()
    # Lists to hold checker1, checker2, and Reflected versions of og
    all_checker1_obs = np.zeros_like(raw_observations)
    all_checker2_obs = np.zeros_like(raw_observations)
    # Reflection augmentation is only for Ball
    if 'Ball' in name:
        all_reflected_obs = np.zeros_like(raw_observations)
        all_reflected_state = np.zeros_like(raw_states)
    for idx in range(ntraj):
        print("Augmenting data from traj {0}".format(idx + 1))
        for jdx in range(traj_len):
            frame = raw_observations[idx, jdx, :, :, :]
            frame_state = raw_states[idx, jdx, :]
            all_checker1_obs[idx, jdx, :, :, :] = my_randomizer.checkerboard_type1(frame)
            all_checker2_obs[idx, jdx, :, :, :] = my_randomizer.checkerboard_type2(frame)
            if 'Ball' in name:
                all_reflected_obs[idx, jdx, :, :, :], all_reflected_state[idx, jdx, :] \
                    = my_randomizer.reflect_about_diagonal(frame, frame_state)
    checker1_path_obs = dir_manager.get_file_path('raw', 'all_train_observations_checker1.npy')
    checker1_path_states = dir_manager.get_file_path('raw', 'all_train_states_checker1.npy')
    checker1_path_actions = dir_manager.get_file_path('raw', 'all_train_actions_checker1.npy')
    # save checkers1 dataset and make a copy of states and actions for it
    np.save(checker1_path_obs, all_checker1_obs)
    shutil.copyfile(actions_pth, checker1_path_actions)
    shutil.copyfile(states_pth, checker1_path_states)
    # simi for checkers2
    checker2_path_obs = dir_manager.get_file_path('raw', 'all_train_observations_checker2.npy')
    checker2_path_states = dir_manager.get_file_path('raw', 'all_train_states_checker2.npy')
    checker2_path_actions = dir_manager.get_file_path('raw', 'all_train_actions_checker2.npy')
    np.save(checker2_path_obs, all_checker2_obs)
    shutil.copyfile(actions_pth, checker2_path_actions)
    shutil.copyfile(states_pth, checker2_path_states)
    # simi for reflected if in ball set
    if 'Ball' in name:
        reflected_path_obs = dir_manager.get_file_path('raw', 'all_train_observations_reflected.npy')
        reflected_path_states = dir_manager.get_file_path('raw', 'all_train_states_reflected.npy')
        reflected_path_actions = dir_manager.get_file_path('raw', 'all_train_actions_reflected.npy')
        np.save(reflected_path_obs, all_reflected_obs)
        # Ball state is updated under reflection
        np.save(reflected_path_states, all_reflected_state)
        shutil.copyfile(actions_pth, reflected_path_actions)
