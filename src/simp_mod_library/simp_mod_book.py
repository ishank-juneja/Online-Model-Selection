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

        # Containers for current simple model related estimates on the books
        self.x_mu = torch.zeros(1, self.cfg.state_dim, device=self.cfg.device)
        self.x_sigma = self.cfg.prior_cov * torch.eye(self.cfg.state_dim, device=self.cfg.device).unsqueeze(0)

        self.trans_dist = HeuristicUnscentedKalman(self.cfg)

        # States and actions
        logging.info("Initialized struct and perception for {0} model".format(simp_mod))

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

    def step(self):
        """
        Take a step forward with the simp model being kept on this book
        :return:
        """
        # Get action from controller
        # Sample states
        # state = self.x_mu.repeat(self.config.controller_N, 1)
        # state = torch.from_numpy(self.true_state_history[-1]).to(device='cuda:0').repeat(self.config.controller_N, 1)
        # state = state.reshape(self.config.controller_N, self.state_dim)
        state = self.x_mu.reshape(1, -1)

        action = np.random.uniform(-1.0, 1.0)

        # print(self.x_mu)

        # print('--')

        self.x_mu, self.x_sigma = self.model_lib['cartpole'].trans_dist.predict(actions[i].view(1, -1), self.x_mu,
                                                                                self.x_sigma, )
        # print(self.x_mu)
        # Act in world and get observation
        # TODO for cartpole need to minus action
        observation, reward, done, info = self.env.predict()
        true_state = info['state']
        # print(true_state[:5])
        total_reward += reward
        # Render and update
        z_mu, z_std = self.observation_update(observation)
        # print(z_mu)
        # Log all data from this step
        self.true_state_history.append(true_state)
        self.action_history.append(actions[i].cpu().numpy())
        self.z_mu_history.append(z_mu.cpu().numpy())
        self.z_std_history.append(z_std.cpu().numpy())
        self.img_history.append(observation)

        state_dimension = self.model_lib['cartpole'].cfg.state_dimension

        self.state_cov_history.append(self.x_sigma[0, :state_dimension, :state_dimension].detach().cpu().numpy())
        self.state_mu_history.append(self.x_mu[0, :state_dimension].detach().cpu().numpy())
        self.param_cov_history.append(torch.diag(self.x_sigma[0])[state_dimension:].detach().cpu().numpy())
        self.param_mu_history.append(self.x_mu[0, state_dimension:].detach().cpu().numpy())

        if done:
            return done, False, total_reward, info

        return done, False, total_reward, info

    def clear_episode_lists(self):
        # Initialize all the episode-specific datasets with empty lists
        for data_key in self.data_keys:
            self.episode_data[data_key] = []

    def observation_update(self, obs: np.ndarray):
        """
        Performs a trans_model + filter update based on received observation
        :param obs:
        :return:
        """
        # Encode single image
        masked, conf, mu_y, sigma_y = self.perception(obs)

        # Update belief in world
        self.x_mu, self.x_sigma = self.trans_dist.update(mu_y, self.x_mu, self.x_sigma)

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
        self.clear_episode_lists()

        self.observation_update(obs)

    def hard_reset(self):
        """
        Reset all the state built up online for this simple model
        :return:
        """
        # Reset the online learned UKF related parameters
        self.trans_dist.transition.reset_params()
        # TODO: Increase the scope of the resets once planning etc. are added
