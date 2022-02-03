import gym
# pycharm may not highlight this one but it is needed
import gym_cenvs
import numpy as np
import matplotlib.pyplot as plt
from arm_pytorch_utilities.rand import seed


if __name__ == '__main__':

    # Create a gym object for the environment
    env_name = 'Kendama-v0'
    env = gym.make(env_name)
    env.reset()

    # Trajectory index
    traj_idx = 0
    # Already printed this traj idx status?
    printed = False
    # Lists to hold all actions/states/observations
    all_actions = []
    all_states = []
    all_observations = []
    # Collect observations from ntraj number of trajectories for a sequence of trajlen number of random actions
    while traj_idx < 100:
        state = env.reset()
        # Var to hold the frames of a traj
        img = None
        # List to hold state labels
        traj_states = []
        # List to hold actions
        traj_actions = []
        # List to hold traj observations
        traj_observations = []
        for i in range(100):
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
        traj_idx += 1
    env.close()
