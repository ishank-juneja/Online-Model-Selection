import argparse
import gym
import gym_cenvs
import numpy as np
from src.plotting import SMVOnline
from src.simp_mod_library.kinodynamic_funcs import BallDynamics, CartpoleDynamics
import torch


def main(args):
    # Create the simulated environment corresponding to this simple model
    if args.smodel == 'ball':
        # Catching environment has point-robot fixed to cup in
        #  addition to ball
        env = gym.make("Catching-v0")
        dyn = BallDynamics()
    elif args.model == 'cartpole':
        env = gym.make("MujocoCartpole-v0")
        dyn = CartpoleDynamics()
    else:
        raise NotImplementedError

    # Retrieve cam matrix for active environment
    cam_mat = env.get_cam_mat()

    # Visualization object
    viz = SMVOnline(simp_model=args.smodel, cam_mat=cam_mat)

    state = torch.zeros(1, 6, dtype=torch.float64, device="cuda:0")
    action = torch.tensor([1.0], dtype=torch.float64, device="cuda:0")

    next_state = state
    # Overlay trajectory of simple model state onto a trajectory of simple model frames
    for idx in range(10):
        next_state = dyn(next_state, action)
        print(next_state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--smodel",
                        action='store',
                        type=str,
                        choices=["cartpole", "ball", "dcartpole", "dubins"],
                        help="Short name of simple model for which testing "
                             "dynamics/kineamtics",
                        metavar="smodel",
                        dest="smodel",
                        required=True)

    args = parser.parse_args()

    main(args)
