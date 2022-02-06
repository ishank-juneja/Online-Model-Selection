from src.networks.transition import TransitionDeterministicModel
from src.models.UKVAE import UnscentedKalmanVariationalAutoencoder
from torch.distributions import MultivariateNormal, Normal
import torch
from torch import optim

import numpy as np
import matplotlib.pyplot as plt

class UnscentedKalmanVariationalAutoencoderNNUncertainty(UnscentedKalmanVariationalAutoencoder):

    def __init__(self, config, load_name=None):
        super(UnscentedKalmanVariationalAutoencoderNNUncertainty, self).__init__(config, load_name)

        # Clone transition
        self.uncertainty_model = TransitionDeterministicModel(self.config.state_dimension, self.config.action_dimension)
        self.uncertainty_model = self.uncertainty_model.to(device=self.config.device)

        # We make R a trainable parameter
        self.z_logvar = torch.nn.Parameter(
            torch.log(self.config.emission_noise * torch.ones(self.config.observation_dimension, device=self.config.device)),
            requires_grad=True)

        self.trained = False
        self.beta_init = self.config.beta_init
        self.beta = self.beta_init
        self.max_std = 0

    def reset_model(self):
        self.beta = self.beta_init
        # Clone transition
        self.uncertainty_model = TransitionDeterministicModel(self.config.state_dimension, self.config.action_dimension)
        self.uncertainty_model = self.uncertainty_model.to(device=self.config.device)

        # We make R a trainable parameter
        self.z_logvar = torch.nn.Parameter(
            torch.log(self.config.emission_noise * torch.ones(self.config.observation_dimension, device=self.config.device)),
            requires_grad=True)

        self.trained = False
        self.max_std = 0

    def sample_dynamics(self, state, action, **kwargs):
        x = state[:, :self.config.state_dimension]
        mu = self.transition(x, action)

        if self.trained:
            std = self.uncertainty_model(x, action).exp()
        else:
            std = torch.zeros_like(mu)

        return torch.cat((mu, std), dim=1)

    def predict(self, action, state_mu, state_sigma, **kwargs):
        if self.trained:
            std = self.uncertainty_model(state_mu, action).exp() ** 2
            Q = torch.diag_embed(std)
        else:
            Q = None

        return super().predict(action, state_mu, state_sigma, self.transition, Q)

    def update(self, z, x_mu, x_sigma, R=None):
        R = torch.diag_embed(self.z_logvar.exp() + 1e-5)
        return super().update(z, x_mu, x_sigma, R)

    def train_on_episode(self):
        '''
            Train GP model based on data from episodes z_mu, z_std, u
            For now assume z_mu, z_std, u are a single episodes, of length T x nz, T x nu
        '''
        # Change to use GPUKF instead of standard UKF
        if self.config.fit_params_episodic:
            super().train_on_episode()

        if self.saved_data is not None:
            z_mu = self.saved_data['z_mu'].clone().to(device=self.config.device)
            z_std = self.saved_data['z_std'].clone().to(device=self.config.device)
            u = self.saved_data['u'].clone().to(device=self.config.device)


        self.trained = True

        # Load saved data now

        #TODO need to make compatible with data of form N x T x dim (since have subtrajectories)
        # TODO Ignore unexplainable subtrajectories? Maybe
        from torch.distributions.kl import kl_divergence
        N, T, _ = z_mu.shape

        # TODO currently for no angles - could make general for no. of angles in state
        # Initial x0
        x0_mu = torch.nn.Parameter(torch.zeros(N, self.config.state_dimension,
                                               device=self.config.device))

        x0_logvar = torch.nn.Parameter(torch.log(1. * torch.ones(N, self.config.state_dimension, device=self.config.device)))

        # Distribution over observations
        z_normal = Normal(z_mu, z_std)
        self.transition.train()
        optimiser = optim.Adam([
            {'params': self.uncertainty_model.parameters()},
            #{'params': self.z_logvar},
            {'params': x0_mu},
            {'params': x0_logvar}]
            ,
            lr=self.config.online_lr * 0.5
        )

        for it in range(self.config.online_epochs):
            log_likelihood = 0.0
            dynamics_kl = 0.0

            optimiser.zero_grad()

            # Sample observations #TODO can reduce memory usage here
            z = z_normal.rsample(sample_shape=(self.config.online_samples,))

            # Initialise first state
            x_mu = x0_mu.repeat(self.config.online_samples, 1, 1).view(self.config.online_samples * N, -1)
            x_sigma = torch.diag_embed(x0_logvar.exp().repeat(self.config.online_samples,
                                                              1, 1).view(self.config.online_samples * N, -1))

            # Perform update on initial observation
            z0 = z[:, :, 0].view(-1, self.config.observation_dimension)
            x_mu, x_sigma = self.update(z0, x_mu, x_sigma)

            # Sample initial state
            qx = MultivariateNormal(x_mu, x_sigma)
            x = qx.rsample()

            # Compute ll for first obs_frame
            pz = Normal(self.emission(x), self.z_logvar.exp().sqrt().unsqueeze(0))
            log_likelihood += pz.log_prob(z0).sum()

            predictive_distributions = []
            for t in range(T-1):
                ut = u[:, t].repeat(self.config.online_samples, 1, 1).view(self.config.online_samples * N, -1)
                u_overshoot = u[:, t:t+self.config.overshoot_d].repeat(self.config.online_samples, 1, 1, 1)
                u_overshoot = u_overshoot.view(N * self.config.online_samples, -1, self.config.action_dimension)

                zt = z[:, :, t+1].view(-1, self.config.observation_dimension)

                # Predictive distributions -- with overshooting
                for to in range(u_overshoot.size(1)):
                    # Get estimation of dynamics wrt previous qx
                    mu_x, std_x = torch.chunk(self.sample_dynamics(x, u_overshoot[:, to]), dim=1, chunks=2)
                    var_x = torch.diag_embed(std_x ** 2)
                    px = MultivariateNormal(mu_x, var_x)
                    x = px.rsample()
                predictive_distributions.append(px)

                ### Compute next qx
                # Predict step
                x_mu_bar, x_sigma_bar = self.predict(ut, x_mu, x_sigma)

                # Update step
                x_mu, x_sigma = self.update(zt, x_mu_bar, x_sigma_bar)
                try:
                    qx = MultivariateNormal(x_mu, x_sigma)
                except Exception as e:
                    print(e)
                    print(x_sigma)
                    exit(0)
                # Compute next x
                x = qx.rsample()

                ## Loss terms
                # ll of observation under qx
                pz = Normal(self.emission(x),
                            self.z_logvar.exp().sqrt().unsqueeze(0))
                log_likelihood += pz.log_prob(zt).sum()
                # KL on predictive distributions
                if t >= self.config.overshoot_d - 1:
                    dynamics_kl += kl_divergence(qx, predictive_distributions[t - self.config.overshoot_d + 1]).sum()

            dynamics_kl /= self.config.online_samples * (T - self.config.overshoot_d) * N
            log_likelihood /= self.config.online_samples * T * N

            loss = -(self.beta * log_likelihood - dynamics_kl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 100.0)

            optimiser.step()

            if it % (self.config.online_epochs-1) == 0:
                print('Iter: {}  Total Loss: {:.3}  nll: {:.3} dynamics_kl: {:.3}'.format(it,
                                                                                          loss.item(),
                                                                                          -log_likelihood.item(),
                                                                                          dynamics_kl.item()))


        std = np.asarray([px.stddev.detach().sum(dim=1).cpu().numpy() for px in predictive_distributions]).reshape(-1)
        true_std = z_std.permute(1, 0, 2)[1:].sum(dim=2).view(-1).cpu().numpy()
        self.max_std = np.max(std)

        #plt.scatter(true_std, std)
        #plt.show()


        #self.beta = max(0.9, self.beta - 0.01)

