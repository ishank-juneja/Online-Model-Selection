import numpy as np
import random
from src.config import SegConfig, CommonEncConfig


class MujocoBase:
    """
    For MujocoEnv related things that need to be kept common and consistent across all models
    """
    def __init__(self):
        # image size related config
        self.seg_config = SegConfig()

        # Image sizes used by encoders and by camera matrix for eventual overlay
        self.enc_config = CommonEncConfig()

        # Place holders over-riden during multple inheritance
        self.data = None
        self.model = None
        self.viewer = None

        # Random seed for stuff that happens in mujoco-gym environments (like color randomization)
        random.seed(0)

        # Color choices to be randomized over
        self.color_options = np.array([(0, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1), (1, 1, 1, 1)])
        self.ncolors = len(self.color_options)

    def reset_model(self):
        raise NotImplementedError

    def render(self, **kwargs):
        raise NotImplementedError

    def _get_obs(self):
        size_ = self.seg_config.imsize
        return self.render(mode='rgb_array', width=size_, height=size_, camera_id=0)

    # https://github.com/deepmind/dm_control/blob/87e046bfeab1d6c1ffb40f9ee2a7459a38778c74/dm_control/mujoco/engine.py#L686
    def get_cam_mat(self):
        # Width and height in pixel units
        width = height = self.enc_config.imsize
        camera_id = 0
        pos = self.data.cam_xpos[camera_id]
        rot = self.data.cam_xmat[camera_id].reshape(3, 3).T
        fov = self.model.cam_fovy[camera_id]
        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = -pos
        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = rot
        # Focal transformation matrix (3x4).
        focal_scaling = (1. / np.tan(np.deg2rad(fov) / 2)) * height / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
        # Image matrix (3x3).
        # This is correct since matplotlib/np array viz is -0.5 to 63.5 for a 64x64 image
        image = np.eye(3)
        image[0, 2] = (width - 1) / 2.0
        image[1, 2] = (height - 1) / 2.0
        return image @ focal @ rotation @ translation

    # To be kept the same across simulated environments
    # Called before __init__
    # Purpose unclear ....
    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
