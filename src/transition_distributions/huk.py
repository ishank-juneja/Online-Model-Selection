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
        self.filter = UnscentedKalmanFilter(state_dim=self.config.state_dimension,
                                            obs_dim=self.config.observation_dimension,
                                            control_dim=self.config.action_dimension,
                                            Q=self.Q,
                                            R=self.R,
                                            device=self.config.device)
        # - - - - - - - - - - -

        self.transition = self.config.dynamics_fn(device=self.config.device, log_normal_params=self.config.log_params)

    def update(self, mu_y_next, mu_z_cur, sigma_z_cur):
        """
        :param mu_y_next: Next received observation
        :param mu_z_cur: Current full state estimate
        :param sigma_z_cur: Current uncertainty on estimate (from filter)
        :return:
        """
        mu_z, sigma_z = self.ukf.update(mu_y_next, mu_z_cur, sigma_z_cur, self.emission)

        return mu_z, sigma_z
