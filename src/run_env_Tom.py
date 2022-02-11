"""
    Generating data for pendulum and cartpole using OpenAI gym
"""

# Some of these import are needed even they haven't been used explicitly here
import random
import gym
import gym_cenvs
import numpy as np
import argparse
from gym import wrappers
from torch import Tensor
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="gym environment", default="ContinuousCartpole-v0")
    parser.add_argument("--N", default=100, help="Number of trajectories", type=int)
    parser.add_argument("--len", default=100, help="Trajectory length", type=int)
    parser.add_argument("--name", help="model name", default=None)
    parser.add_argument("--save-observations", action="store_true", help="store trajectories")
    parser.add_argument("--save-states", action="store_true")
    parser.add_argument("--save-actions", action="store_true")
    parser.add_argument("--load", action="store_true", help="load trajectories")
    parser.add_argument("--name-traj", help="name of loaded/saved trajectories", default=None)
    parser.add_argument("--show", action="store_true", help="render env")
    parser.add_argument("--map", action="store_true", help="use mapping to map loaded trajectories")
    parser.add_argument("--map-path", help="path to map")
    parser.add_argument("--normalize-actions", action="store_true")
    parser.add_argument("--image-observations", action="store_true", help="Observations as images?")
    parser.add_argument("--use-policy", action="store_true")
    parser.add_argument("--policy-path", help="path to policy")
    parser.add_argument("--terminate-at-done", action="store_true")
    parser.add_argument("--terminate-off-screen", action="store_true")
    parser.add_argument("--save-fig", action="store_true")
    parser.add_argument("--depth", action="store_true")

    args = parser.parse_args()

    if args.map:
        map = np.load(args.map_path)

    if args.normalize_actions:
        env = NormalizedActions(gym.make(args.env))
    else:
        env = gym.make(args.env)

    if args.save_fig:
        env = wrappers.Monitor(env, '../data/image/{}'.format(args.name_traj), force=True)

    if args.use_policy:
        agent = REINFORCE(env.observation_space.shape[0], env.action_space, hidden=128)
        agent.load_policy(args.policy_path)

    if args.load:
        loaded_states = np.load('../data/trajectories/{}_states.npy'.format(args.name_traj))

        if args.map:
            loaded_states = loaded_states @ map

    env.seed(0)
    env.action_space.seed(0)
    env.reset()
    env.unwrapped.symbolic = not args.image_observations
    all_observations = []
    all_actions = []
    all_states = []

    printed=False
    while len(all_states) < args.N:

        if (len(all_states) % 10 == 0) and not printed:
            printed = True
            print('Done {} trajectories'.format(len(all_states)))
        else:
            printed = False

        state = env.reset()

        observations = []
        actions = []
        states = []
        total_reward = 0.0

        import time
        start = time.time()
        import copy

        img = None
        for i in range(args.len):
            if args.use_policy:
                action, _, _ = agent.select_action(Tensor([state]))
                action = action.detach().cpu().numpy()[0]
            else:
                action = env.action_space.sample()
                #action = env.np_random.uniform(low=-0.1, high=0.1, size=6)
            observation, reward, done, info = env.step(action)
            state = info["state"]

            if args.show:
                if img is None:
                    img = plt.imshow(observation[:, :, :3])
                else:
                    img.set_data(observation[:, :, :3])
                plt.pause(0.01)
                plt.draw()

            if args.load:
                env.unwrapped.state = loaded_states[j, i, :]

            total_reward += reward

            if args.terminate_at_done and done:
                break

            if args.terminate_off_screen and (np.abs(state[0]) > 1.7):
                break

            # observation at t
            # state at t
            # action at t-1

            states.append(state)
            actions.append(action)
            observations.append(observation)

        end = time.time()
        #print('t per step', (start - end) / args.len)

        if len(states) == args.len:
            #print('Total reward: {}'.format(total_reward))
            all_states.append(states)
            all_actions.append(actions)
            all_observations.append(observations)

    #recorder.close()
    env.close()

    all_states = np.array(all_states)
    all_observations = np.array(all_observations)
    all_actions = np.array(all_actions)

    print(all_observations.shape)
    print(np.min(all_observations))
    print(np.max(all_observations))
    if args.save_observations:
        np.save('../data/trajectories/{}_observations.npy'.format(args.name_traj), all_observations)
    if args.save_actions:
        np.save('../data/trajectories/{}_actions.npy'.format(args.name_traj), all_actions)
    if args.save_states:
        np.save('../data/trajectories/{}_states.npy'.format(args.name_traj), all_states)
