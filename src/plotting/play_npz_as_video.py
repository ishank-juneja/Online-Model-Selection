import argparse
import numpy as np
from src.plotting import ClickableMujocoEnv, GIFMaker, SimpleModViz


def main(args):
    # player = ClickableMujocoEnv()
    # player.playback(args.file)

    dir_save = "data/hand_made_tests/tmp"

    arbitrary_simp_mod = 'cartpole'
    viz = SimpleModViz(arbitrary_simp_mod)

    npz_cache = np.load(args.file)

    frames = npz_cache['obs']

    viz.save_train_animation_frames(frames, dir_save)

    gif_maker = GIFMaker()

    gif_name = args.file.split('/')[-1].split('.')[0]

    gif_maker.make_gif("data/hand_made_tests/{0}.gif".format(gif_name), frames_dir=dir_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--file",
                        action='store',
                        type=str,
                        help="Full path to npz file",
                        metavar="file",
                        dest="file")

    args = parser.parse_args()

    main(args)
