"""
Generating data for pendulum and cartpole using OpenAI gym
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
seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Pass name of environment from within file gym_cenvs __init__.py
    parser.add_argument("--env", help="gym environment", default="ContinuousCartpole-v0")
    # Number of trajectories being used to get the desired images
    parser.add_argument("--ntraj", default=100, help="Number of trajectories", type=int)
    # Actions in each traj
    parser.add_argument("--len", default=100, help="Trajectory length", type=int)
    # Save raw image observations as big .npy file: Shape ntrajs x len (traj-len) x 64 x 64 (Frame Shape) x 3 (RGB)
    parser.add_argument("--save-observations", action="store_true", help="store trajectories")
    # Save ground truth states and actions (ntrajs x len (traj-len) x dims in action/state space) as numpy arrays
    parser.add_argument("--save-states", action="store_true")
    parser.add_argument("--save-actions", action="store_true")
    # Display the images being produced
    parser.add_argument("--show", action="store_true", help="render env")
    # Terminate an episode when the underlying task of mujoco cartpole is completed
    parser.add_argument("--terminate-at-done", action="store_true")
    args = parser.parse_args()

    # Create a gym object for the environment
    env_name = args.env
    env = gym.make(env_name)
    env.reset()

    # Create dir manager object for saving results
    mydirmanager = ResultDirManager()
    mydirmanager.add_location('trajs', 'data/{0}/trajectories'.format(env_name), make_dir_if_none=True)
    mydirmanager.add_location('isolated', 'data/{0}/isolated_frames'.format(env_name), make_dir_if_none=True)

    # Trajectory index
    traj_idx = 0
    # list to hold state labels for all observations
    all_traj_states = []
    # Collect observations from ntraj number of trajectories for a sequence of trajlen number of random actions
    while traj_idx < args.ntraj:
        if traj_idx % 10 == 0:
            print('Done {} trajectories'.format(traj_idx))
        state = env.reset()
        # Var to hold the frames of a traj
        img = None
        # Make a dir for storing frames from the current traj
        dir_dict = {'traj_num': traj_idx + 1}
        traj_dir_path = mydirmanager.make_dir_from_dict('trajs', dir_dict)
        # Add this location with full path as location name
        mydirmanager.add_location(traj_dir_path, traj_dir_path)
        # List to hold state labels
        traj_states = []
        for i in range(args.len):
            # Sample a random action
            action = env.action_space.sample()
            # Simulate a step
            observation, _, done, info = env.step(action)
            # Save this observation frame to disk in two places (trajs and isolated)
            obs_path = mydirmanager.next_path(traj_dir_path, 'obs_', '%s.npy')
            obs_path_isolated = mydirmanager.next_path('isolated', 'obs_', '%s.npy')
            np.save(obs_path, observation)
            np.save(obs_path_isolated, observation)
            # info is something we get when we use a mujoco env, NA for gym in built envs
            state = info["state"]
            # Add state to list
            traj_states.append(state)
            all_traj_states.append(state)
            if args.show:
                # For the first time an image is created
                if img is None:
                    img = plt.imshow(observation[:, :, :3])
                else:
                    img.set_data(observation[:, :, :3])
                plt.pause(0.01)
                plt.draw()
                print(state)
                # breakpoint()
            if args.terminate_at_done and done:
                break
            # observation at t
            # state at t
            # action at t-1 (that took us from t-1 to t)
        # Save labels for this trajectory as an array in the same folder
        traj_labels_path = mydirmanager.get_file_path(traj_dir_path, 'traj_states.npy')
        np.save(traj_labels_path, np.array(traj_states))
        traj_idx += 1
    env.close()
    # Save all the labels in one big array for the isolated frames version
    traj_labels_path = mydirmanager.get_file_path('isolated', 'states.npy')
    np.save(traj_labels_path, np.array(all_traj_states))
