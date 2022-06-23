import glob
import subprocess
import os


class GIFMaker:
    def __init__(self, delay: int = 20):
        """
        :param delay: Delay between consecutive frames in some units (100ms??)
        """
        # Delay between consecutive frames of the GIF in some units
        self.delay = str(delay)

    # Takes a dir where frames are stores and makes a gif out of them stored at gif_path
    def make_gif(self, gif_path: str, frames_dir: str):
        print("Baking GIF at {0}... ".format(gif_path))
        frames_path_pattern = os.path.join(frames_dir, '*.png')
        subprocess.call([
            'convert', '-delay', self.delay, '-loop', '0', frames_path_pattern, gif_path
        ])
