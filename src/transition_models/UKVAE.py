import torch
from src.filters import UnscentedKalmanFilter
from src.transition_models import TransitionDeterministicModel, EmissionModel, LinearEmission
from torch.distributions import Normal, MultivariateNormal
from torch.distributions.kl import kl_divergence


class UnscentedKalmanTransitions:
    def __init__(self, config):
        self.config = config

        self.max_std = 0.1
        # Q and R matrices
        self.Q = torch.diag(self.config.transition_noise).to(self.config.device)
        self.R = torch.eye(self.config.observation_dimension, device=self.config.device) * self.config.emission_noise

        if self.config.do_sys_id:
            self.enhanced_state_dim = self.config.state_dimension + self.config.param_dimension
        else:
            self.enhanced_state_dim = self.config.state_dimension

        self.ukf = UnscentedKalmanFilter(self.enhanced_state_dim, self.config.observation_dimension,
                                             self.config.action_dimension, self.Q, self.R, self.config.device)

        if self.config.linear_emission:
            self.emission = LinearEmission(self.enhanced_state_dim,
                                           self.config.observation_dimension,
                                           learnable=self.config.learn_emission,
                                           device=self.config.device)
        else:
            self.emission = EmissionModel(self.enhanced_state_dim, self.config.observation_dimension)

        if self.config.dynamics_fn is None:
            self.transition = TransitionDeterministicModel(self.enhanced_state_dim, self.config.action_dimension)
        else:
            self.transition = self.config.dynamics_fn(self.config.do_sys_id, device=self.config.device,
                                                      log_normal_params=self.config.log_params)

        self.saved_data = None
        self.start = 0

    def predict(self, action, state_mu, state_sigma, transition=None, Q=None):
        if self.config.use_sqrt_ukf:
            state_S = state_sigma.cholesky()
        else:
            state_S = state_sigma

        transition_fn = self.transition if transition is None else transition
        next_mu, next_S, _ = self.ukf.predict(action, state_mu, state_S, transition_fn, Q)

        if self.config.use_sqrt_ukf:
            return next_mu, next_S @ next_S.transpose(1, 2)

        return next_mu, next_S

    def update(self, z, x_mu, x_sigma, R=None):
        if self.config.use_sqrt_ukf:
            x_S = x_sigma.cholesky()
        else:
            x_S = x_sigma

        update_fn = self.ukf.update_linear if self.config.linear_emission else self.ukf.update
        mu, S = update_fn(z, x_mu, x_S, self.emission, R=R)

        if self.config.use_sqrt_ukf:
            return mu, S @ S.transpose(1, 2)

        return mu, S

    def sample_dynamics(self, state, action, transition=None, do_mean=False):
        x = state[:, :self.config.state_dimension]
        if transition is None:
            mu = self.transition(x, action)
        else:
            mu = transition(x, action)

        std = torch.zeros_like(mu)

        return torch.cat((mu, std), dim=1)

    def filter(self, actions, observations, prior_mu, prior_sigma, R=None):
        return self.ukf.filter(self.transition, self.emission, actions, observations, prior_mu, prior_sigma, R)

    def smooth(self, forward_states):
        return self.ukf.smooth(forward_states)

    def train_on_episode(self):
        """
        Fits parameters (for cartpole) given observation history via sgd
        N should really be 1
        :return:
        """
        if self.saved_data is not None:
            z_mu = self.saved_data['z_mu'].clone().to(device=self.config.device)
            z_std = self.saved_data['z_std'].clone().to(device=self.config.device)
            u = self.saved_data['u'].clone().to(device=self.config.device)

        N, T, _ = z_mu.size()
        print(z_mu.shape)
        av_uncertainty = z_std.mean(dim=2).mean(dim=1)

        thresh = 0.06
        mask = (av_uncertainty < thresh).nonzero().squeeze(1)
        if not len(mask):
            return
        z_mu = z_mu[mask]
        z_std = z_std[mask]
        u = u[mask]

        print(z_mu.shape)

        if self.config.fit_params_episodic:
            print('fitting params')
            N, T, _ = z_mu.size()

            init_params = self.transition.get_params().clone()

            param_mu = torch.nn.Parameter(torch.zeros(self.config.param_dimension, device=self.config.device))
            param_logstd = torch.nn.Parameter(torch.log(0.1 * torch.ones(self.config.param_dimension,
                                                                         device=self.config.device)))

            dynamics_fn = self.config.dynamics_fn(True, self.config.device, self.config.log_params)

            # My prior is
            x0_mu = torch.nn.Parameter(torch.zeros(N, self.config.state_dimension,
                                                   device=self.config.device))
            x0_logvar = torch.nn.Parameter(torch.log(1. * torch.ones(N, self.config.state_dimension, device=self.config.device)))

            optimiser = torch.optim.Adam([
                {'params': param_mu},
                {'params': x0_mu},
                {'params': x0_logvar},
                {'params': param_logstd}],
                lr=self.config.online_lr * 0.01
            )

            n_samples = 100
            start = self.start
            for it in range(start, 5 * self.config.online_epochs):
                loss_bound = max(1 - it * 0.01, 0.01)

                optimiser.zero_grad()

                # Sample from first state
                x_mu = x0_mu.view(N, -1)
                x_sigma = torch.diag_embed(x0_logvar.exp() + 1e-3)
                #x_mu, x_sigma = angular_transform(x_mu, x_sigma, 1)
                px = torch.distributions.MultivariateNormal(x_mu, x_sigma)
                x = px.rsample(sample_shape=(n_samples,))
                pparam = Normal(param_mu, param_logstd.exp())
                params = pparam.rsample(sample_shape=(n_samples, N,))


                x_enhanced = torch.cat((x, params), dim=2)

                x = [x_enhanced[:, :, :5]]
                for t in range(T-1):
                    a = u[:, t].view(1, -1, self.config.action_dimension).repeat(n_samples, 1, 1)

                    x_enhanced = dynamics_fn(x_enhanced.view(N*n_samples, -1),
                                             a.view(N*n_samples, -1)).view(n_samples, N, -1)
                    x.append(x_enhanced[:, :, :5])

                x = torch.stack(x, dim=2)

                z_pred = x[:, :, :, :3]

                #loss = ((z_pred - z_mu.view(1, N, T, -1).repeat(n_samples, 1, 1, 1)) ** 2) / (n_samples * T)
                #loss = loss.sum(dim=3).sum(dim=2).sum(dim=0)
                pz = Normal(z_mu, z_std)
                loss = -pz.log_prob(z_pred).sum(dim=3).sum(dim=2).sum(dim=0) / (n_samples * T)

                prior_x = MultivariateNormal(torch.zeros_like(x_mu), torch.diag_embed(torch.ones_like(x_mu)))
                prior_param = Normal(torch.zeros_like(param_mu), torch.ones_like(param_mu))
                kl_regularisation = kl_divergence(pparam, prior_param).sum() + kl_divergence(px, prior_x).sum()
                loss += kl_regularisation
                #loss_masked = torch.where(loss < loss_bound, loss, torch.zeros_like(loss))
                loss = loss.sum() / N
                loss.backward()
                optimiser.step()

                #if it % 10 == 0:
            #    print('Iter {}  Loss {}'.format(it, loss.item()))

            # Set parameters to MAP
            params = (param_mu - param_logstd.exp() ** 2)
            self.transition.set_params(params.detach())
            print('params: ', params.detach().exp())
            import numpy as np
            #mask = loss_masked.detach().cpu().numpy()
            #mask = np.where(mask > 0)[0]
            mask = []
