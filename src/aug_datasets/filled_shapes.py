import argparse
from arm_pytorch_utilities.rand import seed
import numpy as np
import os
from PIL import Image
import random
from src.config.aug_data_dirs_config import AugmentationDataDirs
from src.utils.results_dir_manager import ResultDirManager
from src.config.seg_config import SegConfig


def main(args, seg_config):
    # See random, numpy
    seed(0)

    dir_manager = ResultDirManager()
    dirs = AugmentationDataDirs()
    output_size = seg_config.imsize

    dir_manager.add_location('dataset_save_loc', dirs.filled_shapes.folder)
    print(dirs.filled_shapes_og.folder)
    shapes_paths = dir_manager.list_dir_objects(dirs.filled_shapes_og.folder, dirs.filled_shapes_og.extension)
    shapes_dict = {}
    for fpath in shapes_paths:
        # Load in .bmp files as PIL
        shape = Image.open(fpath)
        shapes_dict[dir_manager.get_name_sans_extension_from_path(fpath)] = shape

    # Create a blank(black) canvas PIL Image on which to put a scaled/augmented version of the image
    canvas = Image.fromarray(np.zeros((output_size, output_size, 3)).astype('uint8'), 'RGB')

    for idx in range(args.nshapes):
        # Random shape, with random color and random orientation, random scale, and random position
        shape_name = random.choice(list(shapes_dict.keys()))
        shape = shapes_dict[shape_name]
        color_tx = np.random.uniform(0.5, 2.0, 3)
        color_tx_matrix = (color_tx[0], 0, 0, 0,
                           0, color_tx[1], 0, 0,
                           0, 0, color_tx[2], 0)
        # Pick random orientation change based on number of sides for better diversity
        if 'tri' in shape_name:
            upper = 180 / 3
        elif 'sq' in shape_name:
            upper = 180 / 4
        elif 'pent' in shape_name:
            upper = 180 / 5
        elif 'hex' in shape_name:
            upper = 180 / 6
        elif 'hept' in shape_name:
            upper = 180 / 7
        elif 'oct' in shape_name:
            upper = 180 / 8
        else:
            raise ValueError("Invalid Shape name in dict key")
        orientation = np.random.uniform(0, upper)
        pasticle = shape.rotate(orientation, expand=True)
        pasticle = pasticle.convert("RGB", color_tx_matrix)
        paste_size = pasticle.size
        scale = random.choice([1, 2, 4])
        pasticle = pasticle.resize((paste_size[0] // scale, paste_size[0] // scale))
        # Position of upper left corner avoids bottom 1/4th of final image
        position = tuple(np.random.randint(0, output_size * 3 // 4, 2))
        # Paste onto copy of blank canvas, copy needed since pasted in place
        canv_copy = canvas.copy()
        canv_copy.paste(pasticle, position)
        canv_copy.save(os.path.join(dirs.filled_shapes.folder, "{0}.jpeg".format(idx + 1)))
        np.save(os.path.join(dirs.filled_shapes.folder, "{0}.npy".format(idx + 1)), np.array(canv_copy))
        if idx % 10 == 0:
            print("Done {0}/{1} shapes".format(idx + 1, args.nshapes))


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
