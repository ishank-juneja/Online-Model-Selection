import glob
import subprocess
import os


class VideoMaker:
    def __init__(self, frame_rate: int = 2):
        """
        :param frame_rate: Delay between consecutive frames in some units (100ms??)
        """
        # frame rate in fps
        self.frame_rate = str(frame_rate)

    # Takes a dir where frames are stores and makes a gif out of them stored at gif_path
    def make_video(self, video_path: str, frames_dir: str):
        print("Baking Video at {0}... ".format(video_path))
        # Append png file name pattern to frames dir in a manner recognizable by ffmpeg
        frames_path_pattern = os.path.join(frames_dir, 'file%05d.png')
        subprocess.call([
            'ffmpeg', '-framerate', self.frame_rate, '-i', frames_path_pattern, '-c:v', 'libx264', '-r', '30',
            '-pix_fmt', 'yuv420p', video_path
        ])
