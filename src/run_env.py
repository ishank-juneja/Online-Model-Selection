"""
Running mujoco-gym environments
"""

# Some of these import are needed even they haven't been used explicitly here
import gym
# pycharm may not highlight this one but it is needed
import gym_cenvs
import numpy as np
import argparse
import matplotlib.pyplot as plt
from results_dir_manager import ResultDirManager
from arm_pytorch_utilities.rand import seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Pass name of environment from within file gym_cenvs __init__.py
    parser.add_argument("--env", help="gym environment")
    # prefix for file names to be saved
    parser.add_argument("--name-traj", help="name of loaded/saved trajectories", default=None)
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
    dataset_name_prefix = args.name_traj
    # Create dir manager object for saving results
    mydirmanager = ResultDirManager()
    mydirmanager.add_location('trajs', 'data/{0}'.format(env_name), make_dir_if_none=True)

    # Trajectory index
    traj_idx = 0
    # Already printed this traj idx status?
    printed = False
    # Lists to hold all actions/states/observations
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
        # # Make a dir for storing frames from the current traj
        # dir_dict = {'{0}_traj_num'.format(dataset_name_prefix): traj_idx + 1}
        # traj_dir_path = mydirmanager.make_dir_from_dict('trajs', dir_dict)
        # # Add this location with full path as location name
        # mydirmanager.add_location(traj_dir_path, traj_dir_path)
        # List to hold state labels
        traj_states = []
        # List to hold actions
        traj_actions = []
        # List to hold traj observations
        traj_observations = []
        for i in range(args.len):
            # Sample a random action
            action = env.action_space.sample()
            # Simulate a step
            observation, _, done, info = env.step(action)
            # info is something we get when we use a mujoco env, NA for gym in built envs
            state = info["state"]
            # For conkers env we take out only the ball/sphere/conker coordinates from state
            # to make our difficult test dataset
            if 'Conkers' in env_name:
                state = state[-2:]
            # Add state and action to list
            traj_states.append(state)
            traj_actions.append(action)
            # Add observation to tmp list
            traj_observations.append(observation)
            # Optional run-time viz
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
            # observation at t
            # state at t
            # action at t-1 (that took us from t-1 to t)
        # If went all the way, then save to disk
        if len(traj_observations) == args.len:
            # for obs_np_array in traj_observations:
            #     # Save this observation frame to disk
            #     obs_path = mydirmanager.next_path(traj_dir_path, '{0}_observation_'.format(dataset_name_prefix), '%s.npy')
            #     np.save(obs_path, obs_np_array)
            # Save labels for this trajectory as an array in the same folder
            # traj_state_labels_path = mydirmanager.get_file_path(traj_dir_path, '{0}_traj_states.npy'.format(dataset_name_prefix))
            # np.save(traj_state_labels_path, np.array(traj_states))
            # traj_action_labels_path = mydirmanager.get_file_path(traj_dir_path, '{0}_traj_actions.npy'.format(dataset_name_prefix))
            # np.save(traj_action_labels_path, np.array(traj_actions))
            # Add to consolidated data structure
            all_observations.append(traj_observations)
            all_actions.append(traj_actions)
            all_states.append(traj_states)
            traj_idx += 1
        # Nothing to update
        else:
            continue
    env.close()
    all_states_path = mydirmanager.get_file_path('trajs', 'all_{0}_states.npy'.format(dataset_name_prefix))
    all_actions_path = mydirmanager.get_file_path('trajs', 'all_{0}_actions.npy'.format(dataset_name_prefix))
    all_observations_path = mydirmanager.get_file_path('trajs', 'all_{0}_observations.npy'.format(dataset_name_prefix))
    np.save(all_states_path, np.array(all_states))
    np.save(all_actions_path, np.array(all_actions))
    np.save(all_observations_path, np.array(all_observations))
