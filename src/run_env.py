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
    # prefix for file names to be saved
    parser.add_argument("--name-traj", help="name of loaded/saved trajectories", default=None)
    # Display the images being produced
    parser.add_argument("--show", action="store_true", help="render env")
    # Var to decide whether observation returned by step() should be an image (64 x 64 x 3) or symbolic
    parser.add_argument("--image-observations", action="store_true", help="Observations as images?")
    # Terminate an episode when the underlying task of mujoco cartpole is completed
    parser.add_argument("--terminate-at-done", action="store_true")
    args = parser.parse_args()

    # Create a gym object for the environment
    env = gym.make(args.env)
    env.reset()

    # Create dir manager object for saving results
    mydirmanager = ResultDirManager()
    mydirmanager.add_location('trajs', 'data/trajectories', make_dir_if_none=False)

    # Unwrapped property is for the behind the scenes dynamics of a specific environment
    env.unwrapped.symbolic = not args.image_observations
    # List to hold complete collection of all observations
    all_observations = []
    # var for printing status of the simulation
    printed = False
    # Trajectory index
    traj_idx = 0
    # Collect observations from ntraj number of trajectiories for a sequence of trajlen number of rnadom actions
    while traj_idx < args.ntraj:
        if (traj_idx % 10 == 0) and not printed:
            printed = True
            print('Done {} trajectories'.format(traj_idx))
        else:
            printed = False
        state = env.reset()
        # List to hold all the obs for the current traj
        observations = []
        # Var to hold the frames of a traj
        img = None
        # Make a dir for storing frames from the current traj
        dir_dict = {'traj_num': traj_idx + 1}
        traj_dir_path = mydirmanager.make_dir_from_dict('trajs', dir_dict)
        # Add this location with full path as location name
        mydirmanager.add_location(traj_dir_path, traj_dir_path)
        for i in range(args.len):
            # Sample a random action
            action = env.action_space.sample()
            # action = env.np_random.uniform(low=-0.1, high=0.1, size=6)
            observation, _, done, info = env.step(action)
            # Save this observation frame to disk
            obs_path = mydirmanager.next_path(traj_dir_path, 'state_', '%s.npy')
            np.save(obs_path, observation)
            # # info is something we get when we use a mujoco env, NA for gym in built envs
            state = info["state"]
            if args.show:
                # For the first time an image is created
                if img is None:
                    img = plt.imshow(observation[:, :, :3])
                else:
                    img.set_data(observation[:, :, :3])
                plt.pause(0.01)
                plt.draw()
            if args.terminate_at_done and done:
                break
            # observation at t
            # state at t
            # action at t-1
            observations.append(observation)
        # Once all the observations from a traj are ready we add them to list
        if len(observations) == args.len:
            all_observations.append(observations)
        traj_idx += 1
    env.close()
    all_observations = np.array(all_observations)
    if args.save_observations:
        np.save('data/trajectories/{}_observations.npy'.format(args.name_traj), all_observations)
