"""
Similar to script for traning segmentation but data created here intended as distractor objects
while traning the seg/enc of a simple model
"""

# Some of these import are needed even they haven't been used explicitly here
from arm_pytorch_utilities.rand import seed
import argparse
import gym
# pycharm may not highlight this one but it is needed
import gym_cenvs
from src.simp_mod_datasets import SimpleModel, FramesHandler
from src.utils import ResultDirManager, SegDataset


def main(args):
    # Check if folder follow convention
    folder = args.folder
    # Retrieve name of simp_model from folder name
    simp_model = folder.split('_')[0]

    model = SimpleModel(simp_model=simp_model, seed=args.seed)

    # Create dir manager object for saving results
    mydirmanager = ResultDirManager()

    images_base_path = mydirmanager.add_location('images', "data/{0}".format(folder))
    frames_handler = FramesHandler(path_to_images=images_base_path, path_to_masks=None, simp_model=simp_model)

    seed(args.seed)

    # Wait till here before making environment since simultaneous open gym environments
    #  are not supported (MuJoCo camera buffer overlap for example)
    model.make_env()
    print(model.long_name)
    traj_len = 0
    traj_done = False

    # Number of frames saved for current simple model
    nsaved = 0

    while nsaved < args.nframes:
        # Maximum 30 frames per trajectory and traj_done is off-screen
        if traj_done or (traj_len > 29):
            # Number of frames from current traj
            traj_len = 0

            state = model.reset()
        # Sample a random action
        action = model.env.action_space.sample()
        # Simulate a step
        cur_obs, _, traj_done, _ = model.step(action)

        saved = frames_handler.save_frame(cur_obs, unique_id=nsaved)

        if saved:
            nsaved += 1
            if nsaved % 50 == 0:
                print('Done {} images'.format(nsaved))

    # Done with this Model
    model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder",
                        action='store',
                        type=str,
                        help="Name of the folder into which to put dataset",
                        metavar="folder")

    parser.add_argument("--nframes",
                        action='store',
                        default=1000,
                        type=int,
                        help="Number of frames to save in folder",
                        metavar="nframes")

    parser.add_argument("--seed",
                        action='store',
                        default=0,
                        type=int,
                        help="random seed for reproducability",
                        metavar=None)

    args = parser.parse_args()

    main(args)
