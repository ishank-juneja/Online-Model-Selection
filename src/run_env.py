"""
    Run mujoco-gym environments
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
import os
import subprocess
import glob


def cleanup_frames_from_disk(path_where_frames_dumped):
    # Get rid of temp png files
    for file_name in glob.glob(os.path.join(path_where_frames_dumped, '*.png')):
        os.remove(file_name)
    return


def save_gif(gif_path, frames_dir):
    print("Baking your GIF ... ")
    frames_path_pattern = os.path.join(frames_dir, '*.png')
    subprocess.call([
        'convert', '-delay', '20', '-loop', '0', frames_path_pattern, gif_path
    ])
    cleanup_frames_from_disk(frames_dir)

def save_traj_frames(frames_list, frames_dir):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.subplots(1, 1)
    # Plot a separate plot for every frame in the trajectory
    for idx in range(len(frames_list)):
        # The image frame goes in the first column
        ax.imshow(frames_list[idx])
        fig.suptitle('Problem Trajectory', size=16)
        # Save temporary png file frames in home folder
        fig.savefig(os.path.join(frames_dir, "file{0:02d}.png".format(idx + 1)))
    fig.clear()
    plt.close()


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
    parser.add_argument("--seed", default=0, help="Random Seed", type=int)
    parser.add_argument("--show", action="store_true")
    # Display the images being produced
    parser.add_argument("--terminate-at-done", action="store_true")
    parser.add_argument("--terminate-off-screen", action="store_true")
    args = parser.parse_args()

    # Create and seed a gym object for the environment
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
    mydirmanager.add_location('cur_dataset', 'data', make_dir_if_none=True)

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
        # Whether early term happened
        early_term = False
        for i in range(args.len):
            # Sample a random action
            action = env.action_space.sample()
            # Simulate a step
            observation, _, done, info = env.step(action)
            # info is a dict from a mujoco env, NA for gym in built envs
            state = info["state"]
            # We save
            # observation at t
            # state at t
            # action at t-1
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
            # If early termination criteria reached
            if done:
                early_term = True
                break
        # If traj was len long, then save to disk
        if len(traj_observations) == args.len:
            if early_term:
                frames_dir_path = mydirmanager.add_location('problem_trajs', 'results/problem_trajs')
                save_traj_frames(traj_observations, frames_dir_path)
                gif_path = mydirmanager.next_path('problem_trajs', 'cartpole_problem_traj', '-%s.gif')
                save_gif(gif_path, frames_dir_path)
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
