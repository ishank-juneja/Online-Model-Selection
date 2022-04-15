import argparse
from arm_pytorch_utilities.rand import seed
from dm_control import mjcf
import numpy as np
import os
from PIL import Image
from src.config import AugmentationDataDirs, SegConfig
from src.pymjcf.regular_polygons import FallingPolygon
from src.utils import ResultDirManager


def main(args, seg_config):
    seed(0)
    dirs = AugmentationDataDirs()

    dir_manager = ResultDirManager()

    size = seg_config.imsize

    # Create location in case dne
    dir_manager.add_location('bla', dirs.outline_shapes.folder)

    for idx in range(args.nshapes):
        if (idx + 1) % 10 == 0:
            print("Done {0} shapes".format(idx + 1))
        shape = FallingPolygon()
        physics = mjcf.Physics.from_mjcf_model(shape.mjcf_model)
        pixels = physics.render(height=size, width=size, camera_id=0, depth=False)
        pixels_pth = os.path.join(dirs.outline_shapes.folder, "{0}.npy".format(idx + 1))
        np.save(pixels_pth, pixels)
        # Also save jpeg for easy viz
        pixels_img = Image.fromarray(pixels)
        pixels_img.save(os.path.join(dirs.outline_shapes.folder, "{0}.jpeg".format(idx + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--nshapes",
                        action='store',
                        type=int,
                        default=5000,
                        help="Number of distinct shape images",
                        metavar="nshapes",
                        dest="nshapes")

    args = parser.parse_args()

    seg_config = SegConfig()

    main(args, seg_config)
