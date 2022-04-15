import gym
import gym_cenvs
import numpy as np
from src.config import BallConfig, CartpoleConfig, DcartpoleConfig, DubinsConfig, SegConfig
import subprocess


class SimpleModel:
    """
    Wrapper around a mujoco-gym environment for handling observations from simple models,
    1. Makes masks from images.
    2. Containers helper methods for getting multiple masks (annotation files) out of images that are made from a combination of simp model
        generated images
    Annotation/mask naming convention format from: https://patrickwasp.com/create-your-own-coco-style-dataset/
    """
    def __init__(self, simp_model: str, seed: int = 0, twin: bool = False):
        """
        Initialize this wrapper but do not make the environment yet since multiple open environments not supported
        :param long_name: Registered gym name of the environment: ex: MujocoCartpole-v0
        :param seed: Random seed for making the model evolve in a deterministic manner
        :param twin: depracated paramter, indicates whether the model is making frames for itself or just ball masks
        for say cartpole/doublecartpole framesd
        :return: An env object
        """
        if simp_model == 'ball':
            self.long_name = 'MujocoBall-v0'
            self.enc_config = BallConfig()
        elif simp_model == 'cartpole':
            self.long_name = 'MujocoCartpole-v0'
            self.enc_config = CartpoleConfig()
        elif simp_model == 'dcartpole':
            self.long_name = 'MujocoDoublecartpole-v0'
            self.enc_config = DcartpoleConfig()
        elif simp_model == 'dubins':
            self.long_name = 'MujocoDubins-v0'
            self.enc_config = DubinsConfig()
        else:
            raise NotImplementedError("Simple model {0}".format(simp_model))

        self.simp_model = simp_model

        # -v2 are depracated since we don't use the cartpole-ball relationship anymore
        # This dictionary is depracated
        self.model_viz_content = {
            'MujocoBall-v0': 'ball',
            'MujocoCartpole-v0': 'cartpole',
            'MujocoCartpole-v2': 'ball',
            'MujocoDoublecartpole-v0': 'dcartpole',
            'MujocoDoublecartpole-v2': 'ball',
            'MujocoDubins-v0': 'dubins',
        }
        # List of environments for which we perform mjcf augmentations at the xml file level
        self.mjcf_augmentations = [
            'MujocoDoublecartpole-v0',
            'MujocoDoublecartpole-v2'
        ]

        # Create uninitialized environment
        self.env = None
        self.seed = seed

        self.twin = twin

        # Masking related
        self.seg_config = SegConfig()
        self.imsize = self.seg_config.imsize

    def which_model(self):
        """
        :return: model shortname
        """
        return self.simp_model

    def zero_velocity_states(self, state: np.ndarray):
        """
        Takes in a state and zeros out the velocity (qvel related) components of that state
        :param state:
        :return:
        """
        if self.simp_model == 'dubins':
            raise NotImplementedError("This function should never be entered for the dubins car model .. ")
        nqpos = self.enc_config.nqpos
        # Assume states after qpos states till the end are all velocity states
        state[nqpos:] = 0.0
        return

    def which_model_in_mask(self):
        """
        :return: model shortname
        """
        return self.simp_model

    def make_env(self):
        self.env = gym.make(self.long_name)
        # https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        self.reset()

    def reset(self):
        # On every reset, recompile a doublecartpole_dynamic file, close current environment
        #  and remake a new environment with the new dynamic file
        if self.long_name in self.mjcf_augmentations:
            # Close the currently open gym environment since we are about to open a new one
            #  after the recompile and we can't have multiple open togther
            self.close()
            self.refresh_xml()
            self.env = gym.make(self.long_name)
            self.seed += 1
            # https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
            self.env.seed(self.seed)
            self.env.action_space.seed(self.seed)
        # Reset either an already open environment or a freshly created one after recompilation
        self.env.reset()

    def close(self):
        self.env.close()

    def step(self, action: float):
        """
        Wrapper around the step of self.env to also return the mask(s) over the observation
        Example returns both a Ball mask and a cartpole mask for a cartpole image
        param: Action to step with
        return: a dictionary of masks contained in obs in addition to =returns by env.step()
        """
        # Action space of gym environment is 3D for Dubins Car env
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info

    def refresh_xml(self):
        """
        If the mjcf of a model is being modified at the xml level, as opposed to using a mujoco_py or dm_control
        Entity wrapper use the mjcf augmenter object
        :return:
        """
        if 'Doublecartpole' in self.long_name:
            # Initiate a subprocess to access dm_control to prevent graphics issues
            #  on using both dm_control and mujoco_py
            subprocess.call(['python3', 'src/pymjcf/dcartpole_aug.py'])

    def get_dt(self) -> float:
        return self.env.dt
