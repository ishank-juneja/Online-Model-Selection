"""
Make simple model datasets for training encoders
"""
# Some of these import are needed even they haven't been used explicitly here
import argparse
from arm_pytorch_utilities.rand import seed
import glob
import gym
# pycharm may not highlight this one but it is needed
import gym_cenvs
import numpy as np
import os
from src.utils import EncDataset
from src.config import SegConfig, CommonEncConfig
from src.simp_mod_datasets import FramesHandler, nsd, SimpleModel
from src.plotting import GIFMaker, SMVOffline


def main(args):
    # Attempt to create EncDataset object out of provided folder name
    enc_dataset = EncDataset(data_dir_name=args.folder)
    # Infer nframes (single or 2stacked) and simple model from folder name
    simp_model = enc_dataset.get_simp_model()
    # Infer number of frames for which this dataset is being made from folder name
    nframes = enc_dataset.get_nframe()

    # Determine down-sample ratio for images
    seg_config = SegConfig()
    enc_config = CommonEncConfig()

    gifmaker = GIFMaker()

    # Do a consistency check and obtain the downsample ratio
    if seg_config.imsize % enc_config.imsize == 0:
        downsample_by = seg_config.imsize // enc_config.imsize
    else:
        raise ValueError("Segmentation and Encoder dimensions are incompatible at {0}, {1}".format(seg_config.imsize,
                                                                                                   enc_config.imsize))

    frames_handler = FramesHandler(nframes=nframes)
    cur_dataset_path = frames_handler.dir_manager.add_location('cur_dataset', 'data/{0}'.format(args.folder),
                                                               make_dir_if_none=True)

    for idx, dataset in enumerate(args.datasets):
        # Seed envs differently for every dataset
        env_seed = args.seed + idx

        # Set the type of dataset
        frames_handler.set_dataset_type(dataset)

        # Create and seed a gym object for the environment
        model = SimpleModel(simp_model=simp_model, seed=env_seed)

        seed(args.seed)

        model.make_env()
        print(model.long_name)
        model.reset()

        # Factor traj_len into square like shape for saving jpeg versions of images
        nhstack, nvstack = nsd(args.len[idx])
        frames_handler.set_PIL_stack_sizes(nhstack, nvstack)

        # Trajectory index
        traj_idx = 0

        # Get camera matrix for projecting points into pixel space
        # Plane in which the simple and complex objects live is the x-z plane for all except dubins car
        #  for dubins car the plane is the x-y plane
        cam_matrix = model.env.cam_matrix

        # Save camera matrix, dubins in x-y plane others in x-z plane
        if simp_model == 'dubins':
            np.save("data/cam_matrix_dubins.npy", cam_matrix)
        else:
            np.save("data/cam_matrix.npy", cam_matrix)

        trajplotter = SMVOffline(simp_model)
        trajplotter.set_nframes(nframes)

        # set delta_t for this model for visualizer object
        trajplotter.set_delta_t(model.get_dt())

        # Make a tmp dir to store the .png frames by viz object
        tmp_dir_path = frames_handler.dir_manager.add_location('tmp', cur_dataset_path +
                                                               '/tmp_{0}'.format(args.datasets[idx]))

        # Collect observations from ntraj number of trajectories for a sequence of trajlen number of random actions
        while traj_idx < args.ntraj[idx]:
            _ = model.reset()
            # List to hold traj state labels
            traj_states = []
            # List to hold traj actions
            traj_actions = []
            # List to hold traj observations
            traj_observations = []
            # Reset the prev observed frame
            prev_obs = None
            # Counter for within 1 trajectory
            jdx = 0
            # Step through a model instantiation upto args/len() number of times
            while jdx < args.len[idx]:
                # Sample a random action, ignore action space all simple models are 1D
                action = np.random.uniform(-1.0, 1.0)
                # action = -1.0
                # Simulate a step
                cur_obs, _, done, info = model.step(action)
                cur_obs = cur_obs[::downsample_by, ::downsample_by, :]
                # info is a dict from a mujoco model, NA for gym in built envs
                state = info["state"]

                if nframes == 2:
                    # Can only mean that this is the first frame of the trajectory
                    if prev_obs is None:
                        # In case of most of the models, we are interested in making the model
                        #  learn what zero velocity corresponds to, however, in the case of dubins car
                        #  giving it a pair of identical frames with a heading angle label leads to ambiguity
                        #  due to its symmetrical rectangular shape
                        if simp_model == 'dubins':
                            # Update prev_obs for next iteration
                            prev_obs = cur_obs
                            continue
                        else:
                            pushed_obs = frames_handler.concat(cur_obs, cur_obs)
                            # Set state labels corresponding to velocities to 0
                            model.zero_velocity_states(state)
                    else:
                        pushed_obs = frames_handler.concat(prev_obs, cur_obs)
                elif nframes == 1:
                    pushed_obs = cur_obs
                else:
                    raise ValueError("nframes must be either 1 or 2, found {0} instead".format(nframes))

                # Update prev_obs for next iteration
                prev_obs = cur_obs

                # If early termination criteria reached, reject traj
                if done:
                    break
                # We save
                # cur_obs at t
                # state at t
                # action at t-1
                # print(state)
                traj_states.append(state)
                traj_actions.append(action)
                # Add cur_obs to tmp list, it is important that obs are added AFTER the early_term done condition
                traj_observations.append(pushed_obs)
                jdx += 1
            # If traj was len long, then save to disk
            #  We accept 1 less than passed traj length for dubins car 2 frame dataset
            if len(traj_observations) == args.len[idx] and args.save:
                traj_dir_path = frames_handler.save_trajectory(traj_observations=traj_observations,
                                                               traj_actions=traj_actions, traj_states=traj_states,
                                                               traj_idx=traj_idx)
                if args.save_traj_viz:
                    trajplotter.save_train_animation_frames(traj_observations,
                                                            frames_handler.dir_manager.get_abs_path('tmp'), traj_states,
                                                            overlay_states=args.overlay)
                    gifmaker.make_gif(traj_dir_path + '/traj_observations.gif', tmp_dir_path)
                    path_where_frames_dumped = frames_handler.dir_manager.get_abs_path('tmp')
                    # Get rid of temp png files after making GIF
                    for file_name in glob.glob(os.path.join(path_where_frames_dumped, '*.png')):
                        os.remove(file_name)
                traj_idx += 1
                if traj_idx % 10 == 0:
                    print('Done {} trajectories'.format(traj_idx))
            elif not args.save:
                traj_idx += 1
                continue
        model.close()
        if args.save_traj_viz:
            frames_handler.dir_manager.remove_location('tmp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder",
                        action='store',
                        type=str,
                        help="Name of the folder into which to put dataset, "
                             "should follow format of EncFolder even if not saving generated frames to disk",
                        metavar="folder")

    parser.add_argument("--save",
                        action='store_true',
                        help="Whether to write generated frames to disk",
                        dest="save")

    parser.add_argument("--datasets",
                        action='store',
                        nargs='+',
                        type=str,
                        choices=["train", "test", "val"],
                        help="Kinds of datasets to make: example: --datasets train test val",
                        metavar="datasets",
                        dest="datasets")

    parser.add_argument("--ntraj",
                        action='store',
                        nargs='+',
                        type=int,
                        help="Number of trajectories to save in respective datasets",
                        metavar="ntraj",
                        dest="ntraj")

    parser.add_argument("--len",
                        action='store',
                        nargs='+',
                        type=int,
                        help="Number of trajectories to save in respective datasets",
                        metavar="len",
                        dest="len")

    parser.add_argument("--seed",
                        action='store',
                        default=0,
                        type=int,
                        help="random seed for reproducability",
                        metavar=None)

    parser.add_argument("--save-traj-viz",
                        action='store_true',
                        help="Save the visualization of the trajectory as a GIF",
                        dest="save_traj_viz")

    parser.add_argument("--overlay",
                        action='store_true',
                        help="Overlay ground truth state information onto viz",
                        dest="overlay")

    args = parser.parse_args()

    main(args)
