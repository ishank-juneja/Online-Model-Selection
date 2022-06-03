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

        # Dictionary to hold the online collected dataset for a particular simple model
        self.episode_history = dict()

        # Keys for the online collected dataset. gt = Ground Truth, ep = episode
        self.data_keys = ["gt_state_history",
                          "mu_y_history",
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

        #  These empty lists are concatenated together into torch tensors to build the
        #  online collected datasets
        self.online_dataset()
        for data_key in self.data_keys:
            self.episode_history[data_key] = []

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
        # TODO: Call the set_dynamics_mode of the Hybrid Dynamics object
        #  If changing dynamics mode (i.e. simple model) perform a state handover

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
        Fits sys-id parameters (for cartpole) given observation history via sgd
        N should really be 1
        :return:
        """
        logging.debug("Entered train episode of HUK ... ")

        if self.saved_data is not None:
            z_mu = self.saved_data['z_mu'].clone().to(device=self.config.device)
            z_std = self.saved_data['z_std'].clone().to(device=self.config.device)
            u = self.saved_data['u'].clone().to(device=self.config.device)
        else:
            raise ValueError("No saved data to estimate model params from ...")
        N, T, _ = z_mu.size()
        print(z_mu.shape)
        av_uncertainty = z_std.mean(dim=2).mean(dim=1)

        # Uncertainty threshold above which an observed trajectory is trusted to be used for sys-id
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

    def reset_episode(self):
        for data_key in self.data_keys:
            self.episode_data[data_key] = []

        # TODO: Variables and referencing here needs to be corrected
        # Initialise prior for state
        self.x_mu = torch.zeros(1, self.state_dim, device=self.device)
        self.x_sigma = self.model_lib['cartpole'].cfg.prior_cov * torch.eye(self.state_dim,
                                                                            device=self.device).unsqueeze(0)

        self.x_sigma[self.state_dim:, self.state_dim:] *= 0.3

        # First step -- we get an initial observation
        observation = self.env.hard_reset()
        self.model_lib['cartpole'].cost_fn.set_goal(goal=self.env.get_goal())
        observation, _, _, info = self.env.step(np.zeros(self.model_lib['cartpole'].cfg.action_dimension))
        self.true_state_history.append(info['state'])
        self.observation_update(observation)
