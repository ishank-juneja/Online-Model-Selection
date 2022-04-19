from src.learned_models.gp_transition import GPDynamics
from src.transition_distributions.huk import HeuristicUnscentedKalman
from src.filters.gpukf import GPUnscentedKalmanFilter
from src.filters.ukf import UnscentedKalmanFilter
from torch.distributions import MultivariateNormal, Normal
import torch
from torch import optim
import numpy as np
import copy


class GPUnscentedKalman(HeuristicUnscentedKalman):
    def __init__(self, config):
        super(GPUnscentedKalman, self).__init__(config)

        # Clone transition
        self.nominal_transition = copy.deepcopy(self.transition)

        self.transition = GPDynamics(self.config.state_dimension + self.config.action_dimension,
                                     self.config.state_dimension, device=self.config.device,
                                     nominal_dynamics=self.nominal_transition)

        self.transition = self.transition.to(device=self.config.device)

        # We make R a trainable parameter
        self.z_logvar = torch.nn.Parameter(
            torch.log(self.config.emission_noise * torch.ones(self.config.observation_dimension, device=self.config.device)),
            requires_grad=False)

        self.trained = False
        self.beta_init = self.config.beta_init
        self.beta = self.beta_init
        self.max_std = 0.0

    def reset_model(self):
        self.beta = self.beta_init
        self.ukf = UnscentedKalmanFilter(self.enhanced_state_dim, self.config.observation_dimension,
                                         self.config.action_dimension, self.Q, self.R, self.config.device)

        self.transition = GPDynamics(self.config.state_dimension + self.config.action_dimension,
                                     self.config.state_dimension, device=self.config.device,
                                     nominal_dynamics=self.nominal_transition)

        self.transition = self.transition.to(device=self.config.device)

        # We make R a trainable parameter
        self.z_logvar = torch.nn.Parameter(
            torch.log(self.config.emission_noise * torch.ones(self.config.observation_dimension, device=self.config.device)),
            requires_grad=True)

        self.trained = False
        self.max_std = 0.0

    def sample_dynamics(self, state, action, do_mean=True):
        x = state[:, :self.config.state_dimension]
        if self.trained:
            px = self.transition(x, action)
            if do_mean:
                mu = px.mean
            else:
                mu = px.sample()

            std = px.stddev
        else:
            return super().sample_dynamics(x, action, self.transition.nominal_dynamics)
        return torch.cat((mu, std), dim=1)

    def predict(self, action, state_mu, state_sigma):
        transition_fn = self.transition if self.trained else self.transition.nominal_dynamics
        return super().predict(action, state_mu, state_sigma, transition_fn)

    def update(self, z, x_mu, x_sigma, R=None):
        R = torch.diag_embed(self.z_logvar.exp() + 1e-5)
        return super().update(z, x_mu, x_sigma, R)

    def train_on_episode(self):
        '''
            Train GP model based on data from episodes z_mu, z_std, u
            For now assume z_mu, z_std, u are a single episodes, of length T x nz, T x nu
        '''
        #self.reset_model()
        # Change to use GPUKF instead of standard UKF
        if not self.trained:
            self.ukf = GPUnscentedKalmanFilter(self.config.state_dimension, self.config.observation_dimension,
                                                                              self.config.action_dimension,
                                                                              self.config.param_dimension,
                                                                              self.Q, self.R, self.config.device)
        if self.config.fit_params_episodic:
            super().train_on_episode()

        if self.saved_data is not None:
            z_mu = self.saved_data['z_mu'].clone().to(device=self.config.device)
            z_std = self.saved_data['z_std'].clone().to(device=self.config.device)
            u = self.saved_data['u'].clone().to(device=self.config.device)


        self.trained = True

        # Load saved data now
        #z_mu = self.saved_data['z_mu']
        #z_std = self.saved_data['z_std']
        #u = self.saved_data['u']

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
        self.transition.train()
        optimiser = optim.Adam([
            {'params': self.transition.parameters()},
            #{'params': self.z_logvar},
            {'params': x0_mu},
            {'params': x0_logvar}],
            lr=self.config.online_lr * 0.5
        )

        B = 64
        num_batches = N // B
        if num_batches == 0:
            B = N
            num_batches = 1

        for it in range(self.config.online_epochs):
            for batch in range(num_batches):
                optimiser.zero_grad()

                log_likelihood = 0.0
                dynamics_kl = 0.0

                batch_start = B * batch
                batch_end = B * (batch+1)
                batch_selector = torch.eye(N, device=self.config.device)[batch_start:batch_end]

                z_mu_batch = z_mu[batch_start:batch_end]
                z_std_batch = z_std[batch_start:batch_end]
                u_batch = u[batch_start:batch_end]

                x0_mu_batch = batch_selector @ x0_mu
                x0_logvar_batch = batch_selector @ x0_logvar

                #x0_mu.register_hook(lambda grad: print(grad))

                # Sample observations #TODO can reduce memory usage here
                z_normal = Normal(z_mu_batch, z_std_batch)
                z = z_normal.rsample(sample_shape=(self.config.online_samples,))

                # Initialise first state
                x_mu = x0_mu_batch.repeat(self.config.online_samples, 1, 1).view(self.config.online_samples * B, -1)
                x_sigma = torch.diag_embed(x0_logvar_batch.exp().repeat(self.config.online_samples,
                                                                  1, 1).view(self.config.online_samples * B, -1))
                #x_mu, x_sigma = angular_transform(x_mu, x_sigma, 1)

                # Perform update on initial observation
                z0 = z[:, :, 0].view(-1, self.config.observation_dimension)
                x_mu, x_sigma = self.update(z0, x_mu, x_sigma)

                # Sample initial state
                qx = MultivariateNormal(x_mu, scale_tril=x_sigma.cpu().cholesky().to(device='cuda:0'))
                x = qx.rsample()

                # Compute ll for first obs
                pz = Normal(self.emission(x), self.z_logvar.exp().sqrt().unsqueeze(0))
                log_likelihood += pz.log_prob(z0).sum()

                predictive_distributions = []
                for t in range(T-1):
                    ut = u_batch[:, t].repeat(self.config.online_samples, 1, 1).view(self.config.online_samples * B, -1)
                    u_overshoot = u_batch[:, t:t+self.config.overshoot_d].repeat(self.config.online_samples, 1, 1, 1)
                    u_overshoot = u_overshoot.view(B * self.config.online_samples, -1, self.config.action_dimension)

                    zt = z[:, :, t+1].view(-1, self.config.observation_dimension)

                    # Predictive distributions -- with overshooting
                    for to in range(u_overshoot.size(1)):
                        # Get estimation of dynamics wrt previous qx
                        px = self.transition(x, u_overshoot[:, to])
                        x = px.rsample()
                    predictive_distributions.append(px)

                    ### Compute next qx
                    # Predict step
                    x_mu_bar, x_sigma_bar = self.predict(ut, x_mu, x_sigma)

                    # Update step
                    x_mu, x_sigma = self.update(zt, x_mu_bar, x_sigma_bar)
                    try:
                        qx = MultivariateNormal(x_mu, scale_tril=x_sigma.cpu().cholesky().to(device='cuda:0'))
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

                dynamics_kl /= self.config.online_samples * (T - self.config.overshoot_d) * B
                log_likelihood /= self.config.online_samples * T * B

                # Regularisation term on inducing points
                regularisation_kl = 0.01 * self.transition.model.variational_strategy.kl_divergence().div(N * T)

                loss = -(self.beta * log_likelihood - dynamics_kl - regularisation_kl)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 100.0)
                #print('grad')
                #print(x0_mu.grad)
                optimiser.step()

                # Monstrosity which constrains GP to have zero mean by constraining inducing output mean
                if self.config.gp_zero_mean:
                    self.transition.model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.data = \
                        0 * self.transition.model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.data

                #elf.z_logvar.data = self.z_logvar.clamp(min=np.log(1e-2))
                #self.transition.x_logvar.data = self.transition.x_logvar.clamp(min=np.log(5e-2))

            if it % (99) == 0:
                print('Iter: {}  Total Loss: {:.3}  nll: {:.3} dynamics_kl: {:.3}  Reg: {:.3}'.format(it,
                                                                                          loss.item(),
                                                                                          -log_likelihood.item(),
                                                                                          dynamics_kl.item(),
                                                                                          regularisation_kl.item()))

            #if it % 10 == 0:
            #    self.beta = max(self.beta - 0.01, .9)
        ## TODO sort this testing code out somewhere for using
        std = [px.stddev.detach().sum(dim=1).cpu().numpy() for px in predictive_distributions]
        self.max_std = np.max(std)

        print('min std', np.min(std))
        print('max std', np.max(std))

        debug = False
        if debug:

            print(np.mean(std))
            print(np.mean(z_std.detach().cpu().numpy()))
            print(np.max(std))
            print(np.max(z_std.detach().cpu().numpy()))
            print(self.transition(torch.randn(1000, self.config.state_dimension, device=z_mu.device),
                                  0.1*torch.randn(1000, 6, device=z_mu.device)).stddev.sum(dim=1).max())
            print(self.transition(torch.randn(1000, self.config.state_dimension, device=z_mu.device),
                                  0.1*torch.randn(1000, 6, device=z_mu.device)).stddev.sum(dim=1).mean())
            print(self.transition(torch.randn(1000, self.config.state_dimension, device=z_mu.device),
                                  0.1*torch.randn(1000, 6, device=z_mu.device)).stddev.sum(dim=1).min())

            #for name, param in self.transition.model.named_parameters():
            #    print(name, param)

            raw_outputscale = self.transition.model.covar_module.raw_outputscale
            constraint = self.transition.model.covar_module.raw_outputscale_constraint
            outputscale = constraint.transform(raw_outputscale)
            #print(outputscale)

            L = self.transition.model.variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar
            #print( L @ L.transpose(1, 2))

            predictive_distributions = []

            filter_mu = []
            # Initialise first state
            x_mu = x0_mu[0].repeat(self.config.online_samples, 1, 1).view(self.config.online_samples, -1)
            x_sigma = torch.diag_embed(x0_logvar[0].exp().repeat(self.config.online_samples,
                                                              1, 1).view(self.config.online_samples, -1))

            # Sample
            #z = z_normal.rsample(sample_shape=(self.config.online_samples,)).view(1, -1, self.config.observation_dimension)
            z = z_mu.view(1, -1, self.config.observation_dimension)
            p = z[:, :, :3]
            q = z[:, :, 3:]
            z = torch.cat((p, q / torch.norm(q, dim=2, keepdim=True)), dim=2)

            #torch.norm(z[:, :, 3:], dim=2))
            # Perform update on initial observation
            z0 = z[:, 0].view(-1, self.config.observation_dimension)
            x_mu, x_sigma = self.update(z0, x_mu, x_sigma)
            # Sample initial state
            qx = MultivariateNormal(x_mu, x_sigma)
            x = qx.mean
            filter_mu.append(x)
            T = z.size(1)
            uu = u.view(1, -1, self.config.action_dimension)
            for t in range(T - 1):
                ut = uu[:, t].repeat(self.config.online_samples, 1, 1).view(self.config.online_samples, -1)
                u_overshoot = uu[:, t:t + self.config.overshoot_d].repeat(self.config.online_samples, 1, 1, 1)
                u_overshoot = u_overshoot.view(self.config.online_samples, -1, self.config.action_dimension)

                zt = z[:, t + 1].view(-1, self.config.observation_dimension)

                # Predictive distributions -- with overshooting
                for to in range(u_overshoot.size(1)):
                    # Get estimation of dynamics wrt previous qx
                    px = self.transition(x, u_overshoot[:, to])
                    x = px.mean
                    predictive_distributions.append(px)

                ### Compute next qx
                # Predict step
                x_mu_bar, x_sigma_bar = self.predict(ut, x_mu, x_sigma)
                # Update step
                x_mu, x_sigma = self.update(zt, x_mu_bar, x_sigma_bar)

                qx = MultivariateNormal(x_mu, x_sigma)
                x = qx.mean
                filter_mu.append(x)

            import matplotlib.pyplot as plt
            zz_mu = z.detach().cpu().view(-1, self.config.observation_dimension).numpy()
            zz = z_std.detach().cpu().view(-1, self.config.observation_dimension).numpy()
            filter_mu = torch.stack(filter_mu, dim=1).detach().cpu().numpy()[0]
            std = torch.stack([px.stddev.detach().cpu() for px in predictive_distributions], dim=1).numpy()[0]
            mu = torch.stack([px.mean.detach().cpu() for px in predictive_distributions], dim=1).numpy()[0]
            fig, axes = plt.subplots(3, 3)
            t = np.arange(0, std.shape[0])

            print('actual')
            print('total variance', (zz ** 2).sum())
            print('total std', zz.sum())
            print('predicted')
            print('total variance', (std ** 2).sum())
            print('total std', std.sum())

            for i in range(6):
                x = [0, 1, 2] * 3
                y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
                labels = ['px', 'py', 'pz', 'qx', 'qy', 'qz']
                axes[x[i], y[i]].set_title(labels[i])
                axes[x[i], y[i]].plot(zz_mu[1:, i], color='k', label='measured')
                axes[x[i], y[i]].plot(mu[:, i], color='m', label='predicted')
                axes[x[i], y[i]].plot(filter_mu[1:, i], color='c', label='filtered')

                axes[x[i], y[i]].fill_between(t, zz_mu[1:, i] - zz[1:, i], zz_mu[1:, i] + zz[1:, i], color='b', alpha=0.3)
                axes[x[i], y[i]].fill_between(t, mu[:, i] - std[:, i], zz_mu[1:, i] + std[:, i], color='r', alpha=0.3)
            plt.legend()
            plt.show()
            fig, axes = plt.subplots(3, 3)

            for i in range(6):
                x = [0, 1, 2] * 3
                y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
                labels = ['px', 'py', 'pz', 'qx', 'qy', 'qz']
                axes[x[i], y[i]].set_title(labels[i])
                axes[x[i], y[i]].plot(std[:, i], color='k', label='predicted')
                axes[x[i], y[i]].plot(zz[1:, i], color='m', label='measured')

            fig = plt.figure()
            plt.scatter(zz[1:].sum(axis=1), std.sum(axis=1))
            plt.show()

        print(self.z_logvar.exp())
        print(self.transition.x_logstd.exp()**2)
        self.transition.eval()

        #self.beta = max(0.9, self.beta - 0.01)

