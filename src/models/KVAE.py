import torch
from torch import nn
from torch.distributions import MultivariateNormal
from src.models.base_model import BaseModel
from src.filters.kalman import KalmanFilter
from src.networks.conv import KVAEEncoder, KVAEDecoder
from src.networks.transition import DynamicsParameterNetwork


class KalmanVariationalAutoencoder(BaseModel):

    def __init__(self, config):
        super(KalmanVariationalAutoencoder, self).__init__(config)

        # Q and R matrices
        self.Q = torch.diag(self.config.transition_noise).to(self.config.device)
        self.R = torch.eye(self.config.observation_dimension, device=self.config.device) * self.config.emission_noise

        self.kf = KalmanFilter(config.action_dimension, self.Q, self.R, self.config.device)
        self.decoder = KVAEDecoder(self.config.observation_dimension, upscale_factor=2)
        self.encoder = KVAEEncoder(self.config.observation_dimension * 2, mc_dropout=self.config.mc_dropout)
        self.dynamics_par_net = DynamicsParameterNetwork(self.config.observation_dim, self.config.K)

        # Learnable parameters
        self.A = nn.Parameter(torch.empty(self.config.K, self.config.state_dimension ** 2))
        self.B = nn.Parameter(torch.empty(self.config.K, self.config.state_dimension * self.config.action_dimension))
        self.C = nn.Parameter(torch.empty(self.config.K, self.observation_dim * self.config.state_dimension))
        self.z0 = nn.Parameter(torch.zeros(self.config.observation_dimension))

        # Initialise A, B, C
        # A initialised to identity, B and C initialised randomly
        self.A.data = torch.eye(self.config.state_dimension).view(1, -1).repeat(self.config.K, 1)
        self.B.data = 0.05 * torch.randn(self.config.K, self.config.state_dimension * self.config.action_dimension)
        self.C.data = 0.05 * torch.randn(self.config.K, self.config.state_dimension * self.config.observation_dimension)

    def predict(self, state_mu, state_sigma, action, A, B):
        return self.kf.predict(A, B, action, state_mu, state_sigma)

    def filter(self, A, B, C, actions, observations, prior_mu, prior_sigma):
        return self.kf.filter(A, B, C, actions, observations, prior_mu, prior_sigma)

    def smooth(self, A, forward_states):
        return self.kf.smooth(A, forward_states)

    def get_transition_matrices(self, z):
        '''

        :param z:  N x T x z_dim trajectory of latent observations, calculates corresponding A,B,C
        :return: A, B, C matrices for linear SSM
        '''
        N, T, _ = z.shape
        init_alpha_hidden = (
                        torch.zeros(N, self.dynamics_par_net.hidden_size, device=self.config.device),
                        torch.zeros(N, self.dynamics_par_net.hidden_size, device=self.config.device))

        alpha, alpha_hidden = self.dynamics_par_net(self.z0, init_alpha_hidden)
        A = []
        B = []
        C = []
        for t in range(T):
            At = (alpha @ self.A).view(N, self.config.state_dimension, self.config.state_dimension)
            Bt = (alpha @ self.B).view(N, self.config.state_dimension, self.config.action_dimension)
            Ct = (alpha @ self.C).view(N, self.config.observation_dimension, self.config.state_dimension)

            A.append(At)
            B.append(Bt)
            C.append(Ct)

            alpha, alpha_hidden = self.dynamics_par_net(z[:, t], alpha_hidden)

        return torch.stack(A, 1), torch.stack(B, 1), torch.stack(C, 1), alpha_hidden

    def get_dynamics_elbo(self, observations, actions):
        N, T, _ = observations.shape
        prior_mu = torch.zeros(N, self.config.state_dimension, device=self.config.device).unsqueeze(2)
        prior_sigma = self.config.prior_conv * torch.eye(self.config.state_dimension, device=self.config.device)
        prior_sigma = prior_sigma.unsqueeze(0).repeat(N, 1, 1)

        A, B, C, _ = self.get_transition_matrices(observations)

        forward_states = self.filter(A, B, C, actions, observations, prior_mu, prior_sigma)
        backward_states = self.smooth(A, forward_states)
        log_likelihoods = self.kf.get_log_likelihoods(backward_states, A, B, C, observations,
                                                      actions, prior_sigma, forward_states)

        return log_likelihoods.sum()

    def rollout_actions(self, z_to_date, actions):
        N, T, _ = actions.shape
        _, init_obs, _ = z_to_date.shape
        # initial state and pred state -- Gaussian zero mean unit variance
        prior_mu = torch.zeros(N, self.config.state_dimension,
                               device=self.config.device).view(-1, self.config.state_dimension)

        prior_sigma = self.config.prior_cov * torch.eye(self.config.state_dimension,
                                                        device=self.config.device).unsqueeze(0).repeat(N, 1, 1)

        At, Bt, Ct, alpha_hidden = self.get_transition_matrices(z_to_date)

        forward_states = self.filter(actions[:, :init_obs], z_to_date, prior_mu, prior_sigma)
        x_mu = forward_states[2][:, -1]
        x_sigma = forward_states[2][:, -1]
        x = MultivariateNormal(x_mu, x_sigma).sample().view(-1, self.config.state_dimension, 1)

        z = []
        for t in range(init_obs, T):
            z.append(C @ x)
            alpha, alpha_hidden = self.dynamics_par_net(z[-1], alpha_hidden)
            A = (alpha @ self.A).view(N, self.config.state_dimension, self.config.state_dimension)
            B = (alpha @ self.B).view(N, self.config.state_dimension, self.config.action_dimension)
            C = (alpha @ self.C).view(N, self.config.observation_dimension, self.config.state_dimension)
            x = A @ x + B @ actions[:, t].view(-1, self.config.action_dimension, 1)

        z = torch.cat((z_to_date, torch.stack(z, 1)), 1)

        z_mu = z
        z_std = 0.01 * torch.ones_like(z)

        return z, z_mu, z_std