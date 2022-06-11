import torch
from torch import nn
from src.filters import UnscentedKalmanFilter
from src.learned_models import LinearEmission


class HeuristicUnscentedKalman(nn.Module):
    """
    Class definition for a state transition distribution with a hard-coded uncertainty heuristic
    Notation used is the one in the paper and different from the one in the UKF class def
    """
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

        self.saved_data = None
        self.start = 0

    def predict(self, action, hat_mu_z_prev, sigma_z_prev, transition=None, Q=None):
        """
        :param action: Action taken to prop. state
        :param hat_mu_z_prev: Best estimate of state at the end of the last time step
        :param sigma_z_prev: Best estimate of uncertainty until the last time step
        :param transition: Dynamics relationship in case different from the one HUK was inited with
        :param Q: Process noise for transitions, pass if time dependent
        :return:
        """
        # Infer the transition_fn to be used
        transition_fn = self.transition if transition is None else transition
        # Perform the predict/propagate step of the filter
        hat_mu_z_min, sigma_z_min = self.filter.predict(hat_x_plus_prev=hat_mu_z_prev, P_plus_prev=sigma_z_prev,
                                                        control=action, dynamics_fn=transition_fn, Q=Q)
        # Return predicted state estimates
        return hat_mu_z_min, sigma_z_min

    def update(self, mu_y_t, hat_mu_z_min, sigma_z_min, R):
        """
        :param mu_y_t: State estimate from perception at time-step t
        :param hat_mu_z_min: Uncorrected/propagated state estimate for time t
        :param sigma_z_min: Uncorrected/propagated uncertainty estimate at time t
        :param R: In case observation noise is time dependent
        :return:
        """
        hat_mu_z, sigma_z = self.filter.update(z_k=mu_y_t, hat_x_min_k=hat_mu_z_min, P_min_k=sigma_z_min,
                                               obs_fn=self.emission, R=R)
        # Current best estimate of mean and variance
        return hat_mu_z, sigma_z

