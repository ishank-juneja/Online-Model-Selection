"""
Load a .xml file using pymjcf
From: https://github.com/deepmind/dm_control/tree/master/dm_control/mjcf
"""
import argparse
import matplotlib.pyplot as plt
from dm_control import mjcf


def main(args):
    # Parse from path
    mjcf_model = mjcf.from_path(args.filename)

    # Parse from file
    with open(args.filename) as f:
        mjcf_model = mjcf.from_file(f)

    print(type(mjcf_model))  # <type 'mjcf.RootElement'>

    physics = mjcf.Physics.from_mjcf_model(mjcf_model)

    frame = physics.render()

    plt.imshow(frame)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--xml-path",
                        action='store',
                        default="gym_cenvs/assets/Rover4We/rover4We-only.xml",
                        type=str,
                        help="Path to xml file to be loaded for inspection",
                        metavar="xml-filename",
                        dest="filename")

    args = parser.parse_args()

    main(args)

