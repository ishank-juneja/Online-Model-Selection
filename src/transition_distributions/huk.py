import logging
import numpy as np
import torch
from torch import nn
from src.filters import UnscentedKalmanFilter
from src.learned_models import TransitionDeterministicModel, LinearEmission
from torch.distributions import Normal, MultivariateNormal
from torch.distributions.kl import kl_divergence


class HeuristicUnscentedKalman(nn.Module):
    def __init__(self, config):
        super(HeuristicUnscentedKalman, self).__init__()

        self.config = config

        self.max_std = 0.1

        # Assemble emission model to implement y_t = Cz_t + \eps from LVSPC paper
        # - - - - - - - - - - -
        self.emission = LinearEmission(state_dim=self.config.state_dimension,
                                       observation_dim=self.config.observation_dimension,
                                       device=self.config.device)
        # - - - - - - - - - - -

        # Assemble Unscented Kalman filter
        # - - - - - - - - - - -
        # Q (process noise/transition noise) and R (observation noise) matrices
        self.Q = torch.diag(self.config.transition_noise).to(self.config.device)
        # R implements the eps_t term in y_t = Cz_t + eps_t
        self.R = self.config.emission_noise * torch.eye(self.config.observation_dimension, device=self.config.device)
        self.ukf = UnscentedKalmanFilter(state_dim=self.config.state_dimension,
                                         obs_dim=self.config.observation_dimension,
                                         control_dim=self.config.action_dimension,
                                         Q=self.Q,
                                         R=self.R,
                                         device=self.config.device)
        # - - - - - - - - - - -

        self.transition = self.config.dynamics_fn(device=self.config.device, log_normal_params=self.config.log_params)

        # Keys for the online collected dataset. gt = Ground Truth, ep = episode
        self.data_keys = ["mu_y_history",
                          "sigma_y_history",
                          "mu_z_history",
                          "sigma_z_history",
                          "param_mu_history",
                          "param_sigma_history",
                          "all_ep_frames",
                          "all_ep_rollouts"]

        # Container for the episode data collected for a particular simple model
        self.episode_data = dict()
        # Initialize all the episode-specific datasets with empty lists
        for data_key in self.data_keys:
            self.episode_data[data_key] = []

    def reset_episode(self):
        for data_key in self.data_keys:
            self.episode_data[data_key] = []
