import glob
import subprocess
import os


class GIFmaker:
    def __init__(self, delay: int = 20):
        # Delay between consecutive frames of the GIF in some units
        self.delay = str(delay)

    # Takes a dir where frames are stores and makes a gif out of them stored at gif_path
    def make_gif(self, gif_path: str, frames_dir: str):
        print("Baking GIF at {0}... ".format(gif_path))
        frames_path_pattern = os.path.join(frames_dir, '*.png')
        subprocess.call([
            'convert', '-delay', self.delay, '-loop', '0', frames_path_pattern, gif_path
        ])
