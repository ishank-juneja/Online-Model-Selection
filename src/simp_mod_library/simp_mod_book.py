import logging
import numpy as np
from src.config import PerceptionConfig
from src.learned_models import SimpModPerception
from src.plotting import GIFmaker, SMVOnline
from src.transition_distributions import HeuristicUnscentedKalman
import torch
from typing import Dict


class SimpModBook:
    """
    Class to encapsulate the attributes associated with a single simple model, Notation same as paper
    Collection of SimpModBooks is a SimpModLib
    Attributes:
    1. Kinematic/dynamic/kino-dynamic transition function (trans_fn)
    2. Object to query for current cost (cost_fn)
    3. A combined filter-transition distribution class (i.e. model p_z)
        example: (GPUKF, or UKF with hard-coded transition uncertainty scheme)
    4. A perception that maps the current observation to state and uncertainty phi(o_t) -> mu_y, Sigma_y
    """
    def __init__(self, simp_mod: str, dir_manager, device: str):
        """
        :param simp_mod: string of simple model name
        :param device: string for device to put tensors on: 'cpu', 'cuda:0' etc.
        """
        # Name of the simple model kept by this book
        self.name = simp_mod

        # dir_manager object handed down from upper classes
        self.dir_manager = dir_manager

        # Device to put data structures associated with this book on
        self.device = device

        # Create perception file_name config object for initializing the perception of this simple model
        per_config = PerceptionConfig()
        # Load in perception object
        self.perception = SimpModPerception(**per_config[self.name])
        logging.info("Loaded perception for {0} model".format(self.name.capitalize()))

        # Config for simple model book-keeping is identical to config from perception
        self.cfg = self.perception.cfg()

        # State dimension used for keeping the book of model with name
        # self.nstates = self.cfg.state_dim + self.cfg.rob_dim
        self.nstates = self.cfg.state_dim

        self.trans_dist = HeuristicUnscentedKalman(self.cfg, smodel_name=self.name)
        logging.info("Created Transition Model for {0} model".format(self.name.capitalize()))

        # Keys for the online collected dataset. gt = Ground Truth, ep = episode
        self.data_keys = ["mu_y",
                          "sigma_y",
                          "mu_z",
                          "sigma_z",
                          "param_mu",
                          "param_sigma",
                          "seg_conf",
                          "masked_frame"
                          ]

        # Container for the episode data collected for a particular simple model
        # Core datastruct that forms the book
        self.episode_history = dict()
        # Init dict entries with empty lists
        self.clear_episode_history_lists()

        # Containers for current simple model related estimates on the books,
        # add a dim. at axis=0 for batched pro.
        # Initialized t the standard prior for batched processing
        self.z_mu = torch.zeros(1, self.nstates, device=self.cfg.device, dtype=torch.float64)
        self.z_sigma = self.cfg.prior_cov * torch.eye(self.nstates, device=self.cfg.device,
                                                      dtype=torch.float64).unsqueeze(0)

    def __str__(self):
        return "Book for simple model {0}".format(self.name)

    def __repr__(self):
        return self.__str__()

    def reset_model_state(self):
        """
        Reset simple model state
        :return:
        """
        self.z_mu = torch.zeros(1, self.nstates, device=self.cfg.device, dtype=torch.float64)
        self.z_sigma = self.cfg.prior_cov * torch.eye(self.nstates, device=self.cfg.device,
                                                      dtype=torch.float64).unsqueeze(0)
        return

    def clear_episode_history_lists(self):
        """
        Initialize all the episode-specific datasets with empty lists
        :return:
        """
        # Use lits to account for variable traj lengths
        for data_key in self.data_keys:
            self.episode_history[data_key] = []

    def predict(self, action, rob_state):
        """
        Invoke predict of trans_dist
        :param action:
        :param rob_state: Needed to make predictions about next state
        :return:
        """
        self.z_mu, self.z_sigma = self.trans_dist.predict(action, rob_state, self.z_mu, self.z_sigma)
        return

    def observation_update(self, obs: np.ndarray):
        """
        Performs a trans_model + filter update based on received observation
        :param obs:
        :return:
        """
        # Encode single image
        #  mu_y and sigma_y here are type 1 states
        masked, conf, mu_y, sigma_y = self.perception(obs)

        # Use a mask instead of re-training perception to ignore cart-state
        obs_mask = self.cfg.obs_mask

        if obs_mask is not None:
            # Get pruned type 1 states
            mu_y_pruned = mu_y[:, obs_mask]
            sigma_y_pruned = sigma_y[:, obs_mask]
        else:
            # Get pruned type 1 states
            mu_y_pruned = mu_y
            sigma_y_pruned = sigma_y

        # Update belief in world
        self.z_mu, self.z_sigma = self.trans_dist.update(mu_y_pruned, self.z_mu, self.z_sigma)

        # Cache results to dataset
        self.episode_history['mu_y'].append(mu_y.cpu().detach().numpy())
        self.episode_history['sigma_y'].append(sigma_y.cpu().detach().numpy())
        self.episode_history['mu_z'].append(self.z_mu.cpu().detach().numpy())
        self.episode_history['sigma_z'].append(self.z_sigma.cpu().detach().numpy())
        self.episode_history['masked_frame'].append(masked)
        self.episode_history['seg_conf'].append(conf)

        return

    def state_dim(self) -> int:
        return self.cfg.state_dim

    def save_episode_data(self):
        """
        Save all the data associated accumulated for the last completed episode
        :return:
        """
        # Convert every quantity of interest to a np array and then save as npz
        self.data_keys = ["mu_y",
                          "sigma_y",
                          "mu_z",
                          "sigma_z",
                          "param_mu",
                          "param_sigma",
                          "seg_conf",
                          "masked_frame"
                          ]

    def reset_episode(self, obs: np.ndarray):
        """
        Reset the episode specific state for this simple model
        :param obs: The observation for an episode at t=0 obtained after env reset by agent
         invoking Lib and Book as tools
        :return:
        """
        # Reset state
        self.reset_model_state()
        # Clear any state built up over an episode for the transition distributions
        self.clear_episode_history_lists()
        # Infer the initial state for starting to plan
        self.observation_update(obs)

    def reset_book(self):
        """
        Reset all the state built up online for this simple model
        :return:
        """
        # Reset the online learned UKF related parameters
        self.trans_dist.transition.reset_params()
        # TODO: Increase the scope of the resets once planning etc. are added

        logging.info("Hard reset book for {0} model".format(self.name.capitalize()))

    def save_online_episode_viz(self, traj_hist_dict: Dict):
        """
        Save the visualization for this smodel's episode
        :param traj_hist_dict: Data needed by trajectory plotting methods in standard format
        :return:
        """
        # Get path to root folder for this run
        root_path = self.dir_manager.get_abs_path('run_log_root')
        # Create a temporary location within root to save frames into
        dir_frames = self.dir_manager.add_location("tmp", root_path + "/tmp")

        # Add a location for the generated frames to
        rets_dict['save_dir'] = dir_save_frames

        # Invoke the chosen viz function from self.viz
        self.viz(**rets_dict)

        # Find the next available GIF name in the folder where GIFs are being saved
        gif_path = self.dir_manager.next_path('', '{0}_{1}'.format(self.model.model_name, viz_suffix),
                                              postfix='%s.gif')

        gif_maker = GIFmaker(delay=35)
        gif_maker.make_gif(gif_path, dir_frames)
