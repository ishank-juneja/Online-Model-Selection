import logging
import numpy as np
from src.config import PerceptionConfig
from src.learned_models import SimpModPerception
from src.transition_distributions import HeuristicUnscentedKalman
import torch


class SimpModBook:
    """
    Class to encapsulate the attributes associated with a single simple model
    Collection of SimpModBooks is a SimpModLib
    Attributes:
    1. Kinematic/dynamic/kinodynamic transition function (trans_fn)
    2. Object to query for current cost (cost_fn)
    TODO: Emph. via code/comments that the filter-transition distribution
     is the most important piece of the Book and most of the book keeping is in fact just happening
     in self.trans with the Book only performing a few extra tasks like choosing the right cost function
     given the task and model_name and providing a class that can be duplicated easily for every model to form the library
    3. A combined filter-transition distribution class (i.e. model p_z)
        example: (GPUKF, or UKF with hard-coded transition uncertainty scheme)
    4. A perception that maps the current observation to state and uncertainty phi(o_t) -> mu_y, Sigma_y
    """
    # TODO: Add more init parameters when adding planning
    def __init__(self, simp_mod: str, device: str):
        """
        :param simp_mod: string of simple model name
        :param device: string for device to put tensors on: 'cpu', 'cuda:0' etc.
        """
        # Name of the simple model keep-ed by this book
        self.name = simp_mod

        # Device to put data structures associated with this book on
        self.device = device

        # Create perception file_name config object for initializing the perception of this simple model
        per_config = PerceptionConfig()
        # Load in perception object
        self.perception = SimpModPerception(**per_config[self.name])

        # Config for simple model book-keeping is identical to config from perception
        self.cfg = self.perception.cfg()

        # Infer state and parameter dimensions from config
        self.state_dim = self.cfg.state_dimension

        # Container for current simple model related estimates on the books
        self.mu_z = torch.zeros(1, self.state_dim, device=self.device)

        self.trans_dist = HeuristicUnscentedKalman(self.cfg)

        # States and actions
        logging.info("Initialized struct and perception for {0} model".format(simp_mod))

    def observation_update(self, obs: np.ndarray):
        """
        Performs a trans_model + filter update based on received observation
        :param obs:
        :return:
        """
        # Encode single image
        _, _, mu_y, sigma_y = self.perception(obs)

        z = z_mu

        # Use R from encoder
        R = None
        # Update belief in world
        self.x_mu, self.x_sigma = self.trans_dist.update(z, self.x_mu, self.x_sigma, R)
        if self.config.viewer:
            self.viewer_history.append(self.env.get_view())

        return z_mu, z_std

    def state_dim(self) -> int:
        return self.cfg.state_dimension

    def reset_episode(self, obs: np.ndarray):
        """
        Reset the episode specific state for this simple model
        :param obs: The observation for an episode at t=0 obtained after env reset by agent
         invoking Lib and Book as tools
        :return:
        """
        # Clear any state built up over an episode for the transition distirbutions
        # TODO: This may be redundant if there is no episode specific state built up in
        #  trans distribution
        self.trans_dist.reset_episode()

        self.observation_update(obs)

    def hard_reset(self):
        """
        Reset all the state built up online for this simple model
        :return:
        """
        # Reset the online learned UKF related parameters
        self.trans_dist.transition.reset_params()
        # TODO: Increase the scope of the resets once planning etc. are added
