"""
Step through and get observations for a gym environment
"""
import argparse
from arm_pytorch_utilities.rand import seed
import gym
import gym_cenvs
import matplotlib.pyplot as plt
import numpy as np
from src.plotting import SimpleModViz
import time


def main(args):
    # Seed envs differently for every dataset
    env_seed = args.seed

    seed(env_seed)

    env = gym.make(args.env)
    env.seed(env_seed)
    env.action_space.seed(env_seed)
    prev_obs = env.reset()

    img = None
    trajs = 0
    # Total number of steps considered
    steps = 0
    error = 0
    prev_cart = None
    # Collect observations from an instance of an environment until it goes done/out of view
    while trajs < args.trajs:
        _ = env.reset()
        while True:
            # Sample a random action, ignore action space all simple models are 1D
            action = np.random.uniform(-1.0, 1.0)
            # action = -1.0
            # Simulate a step
            cur_obs, _, done, info = env.step(action)

            if args.show:
                if img is None:
                    img = plt.imshow(prev_obs[:, :, :3])
                else:
                    img.set_data(prev_obs[:, :, :3])
                plt.pause(1)
                plt.draw()

            prev_obs = cur_obs

            # info is a dict from a mujoco model, NA for gym in built envs
            state = info["state"]

            # Verify velocity labels of slider joint
            # if prev_cart is not None:
                # Check if we can backprop current qvel and qpos to match previous qpos
                # back_propped = state[0] - state[11] * env.dt
                # print("Back propagated position is {0}".format(back_propped))
                # print("Actual prev state is {0}".format(prev_cart))
                # steps += 1
                # error += np.abs(prev_cart - back_propped)
                # print("Per point error so far {0}".format(error/steps))

            # Update prev_cart with current qpos
            # prev_cart = state[0]
            # print(state[0])
            # print(state[11])
            # print(state)
            # time.sleep(1)

            # If early termination criteria reached, reject traj
            if done:
                break
        trajs += 1
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--env",
                        action='store',
                        type=str,
                        help="Name of registered custom mujoco-gym environment",
                        metavar="env",
                        dest="env")

    parser.add_argument("--ntraj",
                        action='store',
                        type=int,
                        default=1,
                        help="Maximum number of trajectories",
                        metavar="max-trajs",
                        dest="trajs")

    parser.add_argument("--seed",
                        action='store',
                        type=int,
                        default=1,
                        help="Integer random seed for gym/random/torch/numpy reprocuability",
                        metavar="seed",
                        dest="seed")

    parser.add_argument("--show",
                        action='store_true',
                        help="Whether to show image observations",
                        dest="show")

    args = parser.parse_args()

    main(args)
