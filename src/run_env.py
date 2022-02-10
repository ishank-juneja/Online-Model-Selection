"""
Running mujoco-gym environments
"""

# Some of these import are needed even they haven't been used explicitly here
import random
import gym
# pycharm may not highlight this one but it is needed
import gym_cenvs
import numpy as np
import argparse
import matplotlib.pyplot as plt
from results_dir_manager import ResultDirManager
from arm_pytorch_utilities.rand import seed
from data_augmentation import SimDomainRandomization

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Pass name of environment from within file gym_cenvs __init__.py
    parser.add_argument("--env", help="gym environment")
    # prefix for file names to be saved
    parser.add_argument("--dataset-type", help="either train or test", choices=['train', 'test'])
    # Number of trajectories being used to get the desired images
    parser.add_argument("--ntraj", default=100, help="Number of trajectories", type=int)
    # Actions in each traj
    parser.add_argument("--len", default=100, help="Trajectory length", type=int)
    # Display the images being produced
    parser.add_argument("--show", action="store_true", help="render env")
    # Random seed for torch, np, random, and gym
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    # Create a gym object for the environment
    seed(args.seed)
    env_name = args.env
    env = gym.make(env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.reset()
    # Get name of dataset type (test/train)
    dataset_name_prefix = args.dataset_type
    # Create dir manager object for saving results
    mydirmanager = ResultDirManager()
    mydirmanager.add_location('cur_dataset', 'data/', make_dir_if_none=True)


    # Trajectory index
    traj_idx = 0
    # Already printed this traj idx status?
    printed = False
    # Lists to hold all actions/states
    all_actions = []
    all_states = []
    all_observations = []
    # Collect observations from ntraj number of trajectories for a sequence of trajlen number of random actions
    while traj_idx < args.ntraj:
        if traj_idx % 10 == 0:
            if not printed:
                print('Done {} trajectories'.format(traj_idx))
                printed = True
        else:
            printed = False
        state = env.reset()
        # Var to hold the frames of a traj
        img = None
        # List to hold traj state labels
        traj_states = []
        # List to hold traj actions
        traj_actions = []
        # List to hold traj observations
        traj_observations = []
        for i in range(args.len):
            # Sample a random action
            action = env.action_space.sample()
            # Simulate a step
            observation, _, done, info = env.step(action)
            # info is a dict from a mujoco env, NA for gym in built envs
            state = info["state"]
            # Add state and action to list
            traj_states.append(state)
            traj_actions.append(action)
            # Add observation to tmp list
            traj_observations.append(observation)
            # Optional run-time matplotlib viz
            if args.show:
                # For the first time an image is created
                if img is None:
                    img = plt.imshow(observation[:, :, :3])
                else:
                    img.set_data(observation[:, :, :3])
                plt.pause(0.01)
                plt.draw()
            # Premature termination criteria
            if done:
                break
            # Note: We save
            # observation at t
            # state at t
            # action at t-1 (that took us from t-1 to t)
        # If traj was len long, then save to disk
        if len(traj_observations) == args.len:
            all_observations.append(traj_observations)
            # Add states and actions to consolidated data structure
            all_actions.append(traj_actions)
            all_states.append(traj_states)
            traj_idx += 1
        else:
            # Nothing to update
            continue
    env.close()
    all_observations_path = mydirmanager.get_file_path('cur_dataset', 'all_{0}_observations.npy'.format(dataset_name_prefix))
    all_states_path = mydirmanager.get_file_path('cur_dataset', 'all_{0}_states.npy'.format(dataset_name_prefix))
    all_actions_path = mydirmanager.get_file_path('cur_dataset', 'all_{0}_actions.npy'.format(dataset_name_prefix))
    np.save(all_states_path, np.array(all_states))
    np.save(all_actions_path, np.array(all_actions))
    np.save(all_observations_path, np.array(all_observations))
