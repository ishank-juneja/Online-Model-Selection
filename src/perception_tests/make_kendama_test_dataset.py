"""
Make a stationary dataset of Kendama environment frames
"""
# Some of these import are needed even they haven't been used explicitly here
import argparse
import gym
# pycharm may not highlight this one but it is needed
import gym_cenvs
from src.simp_mod_datasets.frames_handler import FramesHandler
from src.simp_mod_datasets.make_coco_json import make_segmentation_coco
from src.utils import ResultDirManager
from arm_pytorch_utilities.rand import seed


def main(args):
    # Check if folder follow convention
    folder = args.folder

    # Number of frames to stack together in this dataset
    #  different from args.nframes
    nframes = folder.split('_')[-1]

    for idx, dataset in enumerate(args.datasets):
        # Seed envs differently for every dataset
        env_seed = args.seed + idx

        env = gym.make('Kendama-v0')
        env.seed(env_seed)
        env.action_space.seed(env_seed)
        env.hard_reset()

        # Create dir manager object for saving results
        mydirmanager = ResultDirManager()

        masks_base_path = mydirmanager.add_location('masks', "data/{0}/{1}/annotations".format(folder, dataset))
        images_base_path = mydirmanager.add_location('images', "data/{0}/{1}/images".format(folder, dataset))

        frames_handler = FramesHandler(images_base_path, masks_base_path, 'kendama')

        seed(args.seed)

        # Wait till here before making environment since simultaneous open gym environments
        #  are not supported (MuJoCo camera buffer overlap for example)
        traj_len = 0
        traj_done = False

        # Reset the previous cur_obs fow which we will publish concatenated frame and state in this iteration
        prev_obs = None

        # Number of frames saved for current simple model
        nsaved = 0

        while nsaved < args.nframes[idx]:
            # Maximum 30 frames per trajectory and traj_done is off-screen
            if traj_done or (traj_len > 29):
                # Number of frames from current traj
                traj_len = 0

                # Reset prev obs
                prev_obs = None

                state = env.hard_reset()

            # Sample a random action
            action = env.action_space.sample()
            # Simulate a step
            cur_obs, _, traj_done, _ = env.step(action)

            if nframes == '1frame':
                saved = frames_handler.save_1frame(cur_obs=cur_obs, unique_id=nsaved)
            elif nframes == '2frame1mask':
                saved = frames_handler.save_2frame1mask(prev_obs=prev_obs, cur_obs=cur_obs, unique_id=nsaved)
            elif nframes == '2frame2mask':
                saved = frames_handler.save_2frame2mask(prev_obs=prev_obs, cur_obs=cur_obs, unique_id=nsaved)
            else:
                raise ValueError("Unknown dataset type")

            if saved:
                # Done saving an image and a mask
                nsaved += 1
                if nsaved % 50 == 0:
                    print('Done {} images'.format(nsaved))
                # Another frame from the current trajectory
                traj_len += 1
            # Update prev_obs for next iter
            prev_obs = cur_obs

        # Done with this Model
        env.close()

    # pass dir name following format at: https://patrickwasp.com/create-your-own-coco-style-dataset/
    make_segmentation_coco(dir_name=folder, simple_model="kendama")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder",
                        action='store',
                        type=str,
                        help="Name of the folder into which to put dataset",
                        metavar="folder")

    parser.add_argument("--datasets",
                        action='store',
                        nargs='+',
                        default=None,
                        type=str,
                        help="Kinds of datasets to make: example: train test val",
                        metavar=None)

    parser.add_argument("--nframes",
                        action='store',
                        nargs='+',
                        default=None,
                        type=int,
                        help="Number of frames to save in respective datasets",
                        metavar=None)

    parser.add_argument("--seed",
                        action='store',
                        default=0,
                        type=int,
                        help="random seed for reproducability",
                        metavar=None)

    args = parser.parse_args()

    main(args)
