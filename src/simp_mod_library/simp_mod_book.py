import logging
import numpy as np
from src.config import PerceptionConfig
from src.learned_models import SimpModPerception
from src.transition_distributions import HeuristicUnscentedKalman
import torch


class SimpModBook:
    """
    Class to encapsulate the attributes associated with a single simple model, Notation same as paper
    Collection of SimpModBooks is a SimpModLib
    Attributes:
    1. Kinematic/dynamic/kinodynamic transition function (trans_fn)
    2. Object to query for current cost (cost_fn)
    3. A combined filter-transition distribution class (i.e. model p_z)
        example: (GPUKF, or UKF with hard-coded transition uncertainty scheme)
    4. A perception that maps the current observation to state and uncertainty phi(o_t) -> mu_y, Sigma_y
    """
    def __init__(self, simp_mod: str, device: str):
        """
        :param simp_mod: string of simple model name
        :param device: string for device to put tensors on: 'cpu', 'cuda:0' etc.
        """
        # Name of the simple model kept by this book
        self.name = simp_mod

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

        # Containers for current simple model related estimates on the books, add a dim. at axis=0 for batched pro.
        self.z_mu = torch.zeros(1, self.nstates, device=self.cfg.device)
        self.z_sigma = self.cfg.prior_cov * torch.eye(self.nstates, device=self.cfg.device).unsqueeze(0)
        # Set the uncertainty in rob states to 0
        # self.cfg.prior_cov[:, :self.cfg.rob_dim, :self.cfg.rob_dim] = 0.0

        self.trans_dist = HeuristicUnscentedKalman(self.cfg)
        logging.info("Created Transition Model for {0} model".format(self.name.capitalize()))

        # Keys for the online collected dataset. gt = Ground Truth, ep = episode
        self.data_keys = ["mu_y_history",
                          "sigma_y_history",
                          "mu_z_history",
                          "sigma_z_history",
                          "param_mu_history",
                          "param_sigma_history",
                          "seg_conf_history",
                          "masked_frames_history"
                          ]

        # Container for the episode data collected for a particular simple model
        # Core datastruct that forms the book
        self.episode_data = dict()
        # Init dict entries with empty lists
        self.clear_episode_lists()

    def __str__(self):
        return "Book for simple model {0}".format(self.name)

    def __repr__(self):
        return self.__str__()

    def reset_model_state(self):
        """
        Reset simple model state
        :return:
        """
        self.z_mu = torch.zeros(1, self.cfg.state_dim, device=self.cfg.device)
        self.z_sigma = self.cfg.prior_cov * torch.eye(self.cfg.state_dim, device=self.cfg.device).unsqueeze(0)
        return

    def clear_episode_lists(self):
        # Initialize all the episode-specific datasets with empty lists
        for data_key in self.data_keys:
            self.episode_data[data_key] = []

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
        return

    def state_dim(self) -> int:
        return self.cfg.state_dim

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
        self.clear_episode_lists()
        # Infer the initial state for starting to plan
        self.observation_update(obs)

    def hard_reset(self):
        """
        Reset all the state built up online for this simple model
        :return:
        """
        # Reset the online learned UKF related parameters
        self.trans_dist.transition.reset_params()
        # TODO: Increase the scope of the resets once planning etc. are added

        logging.info("Hard reset book for {0} model".format(self.name.capitalize()))
