import argparse
from arm_pytorch_utilities.rand import seed
from datetime import datetime
import gym
import gym_cenvs
import glob
import numpy as np
import os
from src.plotting import GIFMaker, SMVKDTest
from src.simp_mod_library.kinodynamic_funcs import BallDynamics, CartpoleDynamics
import torch
from src.utils import ResultDirManager
from typing import List


def sim_state_to_dyn_state(state_sim: torch.Tensor, indices_list: List[int]) -> torch.Tensor:
    """
    Convert a state received from simulator to state accepted by dynamics callable
    :return:
    """
    dyn_dims = len(indices_list)
    # unscrambled np array to be returned and used by dynamics callable
    ret = torch.zeros(dyn_dims, dtype=torch.float64)
    for idx in range(dyn_dims):
        ret[idx] = state_sim[indices_list[idx]]
    return ret


def cleanup_frames(tmp_frames_dir):
    """
    Cleanup the frames from the tmp location once done making GIF
    :param tmp_frames_dir: Location to clear frames for to prepare for next call
    :return:
    """
    # Get rid of temp png files
    for file_name in glob.glob(os.path.join(tmp_frames_dir, '*.png')):
        os.remove(file_name)
    return


def main(args):
    env_seed = args.seed

    seed(env_seed)

    # Create the simulated environment corresponding to this simple model
    # Create indices list to bring state into dyn_callable format
    if args.smodel == 'ball':
        # Catching environment has point-robot fixed to cup in
        #  addition to ball
        # Catching environment, state representation:
        # [cup_x, ball_x, ball_y, ball_z, ball_quat (4 dim), cup_vx, ball_vx, ball_vy, ball_vz, ball_angular_vels (3 dim)]
        env = gym.make("Catching-v0")
        # state format: [cup_x, cup_vx, ball_x, ball_y, ball_vx, balL_vy]
        indices_list = [0, 8, 1, 3, 9, 11]
        dyn = BallDynamics(device='cpu')
    elif args.smodel == 'cartpole':
        # Cartpole environment, state representation:
        # [x_cart, x_mass, y_mass, v_cart, vx_mass, vy_mass]
        env = gym.make("MujocoCartpole-v0")
        # state format: [x_cart, v_cart, x_mass, y_mass, vx_mass, vy_mass]
        indices_list = [0, 3, 1, 2, 4, 5]
        dyn = CartpoleDynamics(device='cpu')
    else:
        raise NotImplementedError

    # Seed gym environment and action space
    env.seed(env_seed)
    env.action_space.seed(env_seed)

    dir_manager = ResultDirManager()
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    test_folder_path = "runs/test_kinodynamic_funcs/{0}_{1}".format(args.smodel, current_time)
    dir_manager.add_location('root', test_folder_path)
    # Location for npz file where data from test run will be stored
    npz_file_path = test_folder_path + "/result.npz"
    gif_path = test_folder_path + "/result.gif"
    # Add a tmp location within root
    tmp_loc_path = test_folder_path + '/tmp'
    dir_manager.add_location('tmp', tmp_loc_path)

    # Reset environment
    obs = env.reset()

    # Camera matrix for active environment
    cam_mat = env.get_cam_mat()
    viz = SMVKDTest(args.smodel, cam_mat=cam_mat)
    viz.set_nframes(1)

    data_keys = ["frames", "true_sim_states", "pred_next_states"]
    data_dict = {}

    for key in data_keys:
        data_dict[key] = []

    while True:
        # Sample a random action, ignore action space all simple models are 1D
        action = np.random.uniform(-1.0, 1.0)
        action_pt = torch.tensor([action]).unsqueeze(0)
        # Simulate a step
        obs, _, done, info = env.step(action)
        data_dict['frames'].append(obs)
        # info is a dict from a mujoco model, NA for gym in built envs
        gt_state = torch.from_numpy(info['state'])
        dyn_in_state = sim_state_to_dyn_state(gt_state, indices_list)
        dyn_in_state = dyn_in_state.unsqueeze(0)
        data_dict['true_sim_states'].append(np.array(dyn_in_state))
        dyn_out_state = dyn(dyn_in_state, action_pt)
        data_dict['pred_next_states'].append(np.array(dyn_out_state))

        # If early termination criteria reached, reject traj
        if done:
            break

    env.close()

    # Identical data_dict with all the lists converted to np arrays for caching
    data_dict_np = {}
    # Fill up with the np version of what we need
    for key in data_keys:
        data_dict_np[key] = np.array(data_dict[key])

    viz.save_frames_with_current_and_predicted_frames(frames=data_dict_np['frames'],
                                                      cur_sim_states=data_dict_np['true_sim_states'],
                                                      next_pred_states=data_dict_np['pred_next_states'],
                                                      save_dir=tmp_loc_path)

    gif_maker = GIFMaker(delay=35)

    gif_maker.make_gif(gif_path=gif_path, frames_dir=tmp_loc_path)

    cleanup_frames(tmp_loc_path)

    # Expect tmp dir to be empty after run, try and remove
    try:
        os.rmdir(tmp_loc_path)
    except OSError:
        pass


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

    parser.add_argument("--seed",
                        action='store',
                        type=int,
                        default=1,
                        help="Integer random seed for gym/random/torch/numpy reproducability",
                        metavar="seed",
                        dest="seed",
                        required=False)

    args = parser.parse_args()

    main(args)
