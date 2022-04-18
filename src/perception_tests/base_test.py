from abc import ABCMeta
import glob
import numpy as np
import os
from src.networks import SimpModPerception
from src.plotting import SimpleModViz, GIFmaker
from src.simp_mod_datasets import FramesHandler
from src.utils import ResultDirManager
import subprocess
from typing import Callable, List, Union


class BaseVizTest(metaclass=ABCMeta):
    """
    Base class for visualizing perception tests on saved frames or with mujoco environments
    Loads in a segmenter and an encoder and infers config parameters/the simple_model from filenames
    """
    def __init__(self, enc_model_name: str, seg_model_name: str = None):
        """
        :param enc_model_name: Required, name of the encoder model being used in models/encoder being used
        :param seg_model_name: Optional, Name of the segmentation model being used in models/segmentation
        """
        # Name str associated with the test
        self.test_name = None

        # Init a Visual Model object
        self.model = SimpModPerception(encoder_model_name=enc_model_name, seg_model_name=seg_model_name)

        # Create a file handler object
        self.dir_manager = ResultDirManager()
        # Hand made tests created on kendama environment
        self.dir_manager.add_location('hand_made', 'data/hand_made_tests/')
        # Get a list of test cases from the location
        self.cached_tests = self.dir_manager.scrape_loc_for_glob('hand_made', 'kendama-*.npz')

        # Frames handler utils object
        self.frames_handler = FramesHandler()

        # Dummy env var over-riden by child classes
        self.env = None
        self.env_name = None
        self.use_env = False

        # Visualizer object for running a test
        self.viz = SimpleModViz(self.model.simp_model)

        # nframes per observation to visual_model
        self.nframes = self.model.get_nframes()
        self.viz.set_nframes(self.nframes)

        # Fetch observation dimension (dim of output of encoder)
        self.obs_dim = self.model.encoder.get_obs_dim()

        # Fetch number of members that from the ensemble
        self.nmembers = self.model.encoder.get_n_ensembles()

        # Maximum length of an environment stepped trajectory
        self.max_env_traj = 50

    @classmethod
    def __new__(cls, *args, **kwargs):
        """
        Make abstract base class non-instaiable
        :param args:
        :param kwargs:
        """
        if cls is BaseVizTest:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return object.__new__(cls)

    def simulate_actions_get_frames(self, actions: np.ndarray):

        if self.env_name == 'MujocoDoublecartpole-v0':
            # Randomize geometry
            subprocess.call(['python3', 'src/pymjcf/dcartpole_aug.py'])
        else:
            pass

        # Reset environment before starting simulation
        self.env.reset()

        nactions = min(self.max_env_traj, actions.shape[0])

        frames = []

        # Take a step for every action
        for idx in range(nactions):
            cur_obs, _, done, _ = self.env.step(actions[idx])
            frames.append(cur_obs)
            if done:
                break

        return frames

    def run_perception_on_tests(self, viz_ver: str = "ver3", save_raw: bool = False):
        """
        Function that runs loaded SimpModelPerception on the cached test cases
        :param viz_ver: Version of visualization to use. Versions vary in the number of axes/quantities displayed in GIF
        :param save_raw: Whether to save raw_frames as a GIF. This is only useful when the raw frames are different
          from the masked frames, i.e. when a segmenter is present
        :return:
        """
        # Retrieve the viz function based on passed version number
        viz_function = self.viz.get_viz_function(viz_ver)

        for pdx, file_path in enumerate(self.cached_tests):
            # Load the data from this episode of files
            episode_data = np.load(file_path)

            if self.use_env:
                # Get frames by running cached actions on environment
                actions = episode_data['actions']
                frames = self.simulate_actions_get_frames(actions)
            else:
                # Extract frames from loaded data
                frames = episode_data['obs']

            # Retrieve information to plot by running simple model perception on list of frames
            rets_dict = self.run_perception_on_frames(frames)

            # Save visualization
            self.save_viz(rets_dict=rets_dict, viz_suffix="test", viz_function=viz_function)

            # Once GIF is made cleanup folder of png frames
            self.cleanup_frames()

            # If test asks for plotting raw frames
            if save_raw:
                self.save_viz(rets_dict={'frames': frames}, viz_suffix="raw", viz_function=self.viz.save_frames)
                self.cleanup_frames()

        # Done making GIFs, rm tmp dir
        self.cleanup_tmp_dir()

    def cleanup_frames(self):
        """
        Cleanup the frames from the tmp location once done making GIF
        :return:
        """

        path_where_frames_dumped = self.dir_manager.get_abs_path('tmp')

        # Get rid of temp png files
        for file_name in glob.glob(os.path.join(path_where_frames_dumped, '*.png')):
            os.remove(file_name)

        return

    def cleanup_tmp_dir(self):
        """
        Remove the tmp directory used to store frames
        :return:
        """
        path_where_frames_dumped = self.dir_manager.get_abs_path('tmp')

        # If this dir is empty after frames cleaned up attempt (try:) to delete it, else do nothing
        try:
            os.rmdir(path_where_frames_dumped)
        except OSError:
            pass

    def save_viz(self, rets_dict: dict, viz_suffix: str, viz_function: Callable):
        # Temporary location at which intermediate png frames are saved
        dir_save_frames = self.dir_manager.get_abs_path('tmp')

        # Add the location to save frames
        rets_dict['save_dir'] = dir_save_frames

        # Invoke the chosen viz function of self.viz
        viz_function(**rets_dict)

        # Location where GIFmaker can find the frames to make GIF with
        dir_frames = self.dir_manager.get_abs_path('tmp')

        # Find the next available GIF name in the folder where GIFs are being saved
        gif_path = self.dir_manager.next_path('vid_results', '{0}_{1}'.format(self.model.model_name, viz_suffix),
                                              postfix='%s.gif')

        gif_maker = GIFmaker(delay=35)
        gif_maker.make_gif(gif_path, dir_frames)

    def invoke_perception(self, frame: np.ndarray):
        # Invoke perception model
        masked, conf, mu, stddev = self.model(frame)

        mu = mu.view(self.model.encoder.config.num_ensembles,
                     self.model.encoder.config.observation_dimension)
        mu_np = mu.cpu().detach().numpy()

        stddev = stddev.view(self.model.encoder.config.num_ensembles,
                             self.model.encoder.config.observation_dimension)
        stddev_np = stddev.cpu().detach().numpy()

        return masked, mu_np, stddev_np

    def run_perception_on_frames(self, frames: Union[List, np.ndarray]):
        # raw = raw incoming frames from simulator, masked = masked frames from segmenter
        raw_frames = []
        masked_frames = []

        # Number of frames
        # We query the model this many times regardless of whether we are
        # performing inference on single images or pairs of images since for pairs of images
        # we repeat the first image to form a pair as the first query
        # TODO: Make sure that while using models online, we either do this correctly or not at all
        nframes = len(frames)

        # Allocate arrays to hold mean and stddev predictions over the observations
        traj_mu = np.zeros((nframes, self.obs_dim), dtype=np.float64)

        # 3 kinds of uncertainties: total, alea, epi
        traj_stddev = np.zeros((nframes, 3, self.obs_dim), dtype=np.float64)

        # To hold separate mean predictions for every ensemble member
        full_mu = np.zeros((nframes, self.nmembers, self.obs_dim), dtype=np.float64)

        # Reference to the cur_obs at time {t-1}
        prev_obs = None

        # See results from perception on this cached sequence of actions
        for idx in range(nframes):
            cur_obs = frames[idx]
            if self.nframes == 2:
                # For the very first frame, duplicate cur_obs (would want to do something like this online)
                if prev_obs is None:
                    # Raw frame for visualization without white line que
                    raw_frames.append(self.frames_handler.concat(cur_obs, cur_obs))
                    # Test frame to pass to model with white line que
                    test_frame = self.frames_handler.concat(cur_obs, cur_obs, white_line=False)
                else:
                    raw_frames.append(self.frames_handler.concat(prev_obs, cur_obs))
                    test_frame = self.frames_handler.concat(prev_obs, cur_obs, white_line=False)
                # Update prev_obs for next iteration
                prev_obs = cur_obs
            # Else use single frames
            elif self.nframes == 1:
                raw_frames.append(cur_obs)
                test_frame = cur_obs
            else:
                raise NotImplementedError("Tests for {0} frames not implemented".format(self.nframes))

            # Invoke perception model
            masked, mu_np, stddev_np = self.invoke_perception(test_frame)

            # Add masked frame to results
            masked_frames.append(masked)

            # Predicted observable state means
            full_mu[idx] = mu_np
            traj_mu[idx, :] = mu_np.mean(axis=0)

            # Compute uncertainty terms
            traj_stddev[idx, 1, :] = self.get_aleatoric_uncertainty(stddev_np)
            traj_stddev[idx, 2, :] = self.get_epistemic_uncertainty(mu_np)
            traj_stddev[idx, 0, :] = np.sqrt(
                np.square(traj_stddev[idx, 1, :]) + np.square(traj_stddev[idx, 2, :]))

        # Return a dictionary of results from running perception on the passed frames
        rets = {}

        """
        Save animation test frames as image files
        :param masked_frames: The masked version of the raw input frames, In case there is no segmenter in the
         simple model perception, the masked_frames are the same as the raw frames
        :param traj_mu: Combined mean of the encoder ensemble predictions
        :param traj_stddev: stddevs is a concatenated array of all 3 (total, alea, epi) kinds of uncertanties
        :param traj_mu_full: Individual means of the ensemble members
        :return:
        """

        rets['masked_frames'] = masked_frames
        rets['traj_mu'] = traj_mu
        rets['traj_stddev'] = traj_stddev
        # returning traj_mu is redundant if returning the full version
        #  they are kept separate so the plotting an be agnostic to the relationship between the 2 and doesn't
        #  have to average traj_mu_full to get traj_mu
        rets['traj_mu_full'] = full_mu

        return rets

    @staticmethod
    # Compute epistemic uncertainty from ensemble mean predictions as stddev of means output by base models
    def get_epistemic_uncertainty(z_mu):
        return np.std(z_mu, axis=0)

    @staticmethod
    # Compute aleatoric uncertainty from ensemble mean predictions as mean of sigmas output by base models
    def get_aleatoric_uncertainty(z_std):
        return np.mean(z_std, axis=0)
