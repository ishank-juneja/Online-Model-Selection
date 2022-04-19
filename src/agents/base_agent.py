import logging
from abc import ABCMeta
import numpy as np
from src.mp2i import MPPI
from src.simp_mod_library.simp_mod_lib import SimpModLib
import torch
from typing import List


class BaseAgent(metaclass=ABCMeta):
    """
    Base class for constructing agents that control the complex object using passed Simple Model Library
    """
    def __init__(self, smodel_list: List[str]):
        # # Dummy env var over-riden by child classes
        # self.env = None

        # Whether to online learn GP as transition model
        online_gp = True

        self.model_lib = SimpModLib(smodel_list, online_gp)

        # Task dependent parameters
        self.episode_T: int = None
        self.actions_per_loop: int = None
        self.env = None

        self.action_dimension: int = None
        self.controller = None

        self.state_dim = self.model_lib['cartpole'].cfg.state_dimension
        self.device = self.model_lib['cartpole'].cfg.device

    @classmethod
    def __new__(cls, *args, **kwargs):
        """
        Make abstract base class non-instaiable
        :param args:
        :param kwargs:
        """
        if cls is BaseAgent:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return object.__new__(cls)

    def make_planner(self):
        self.controller = MPPI(dynamics=self.model_lib['cartpole'].trans_dist.sample_dynamics,
                               running_cost=None,
                               nx=self.model_lib['cartpole'].state_dim() * 2,
                               noise_sigma=self.model_lib['cartpole'].mppi_noise_sigma(),
                               num_samples=1000,
                               horizon=20,
                               lambda_=self.model_lib['cartpole'].mppi_lambda(),
                               device=self.model_lib['cartpole'].cfg.device,
                               terminal_state_cost=self.model_lib['cartpole'].cost_fn.compute_cost,
                               u_scale=self.model_lib['cartpole'].cfg.u_scale,
                               u_max=1.0,
                               u_min=-1.0,
                               u_per_command=self.actions_per_loop)

    def reset_trial(self):
        #TODO make this a dictionary?
        self.true_state_history = []
        self.state_mu_history = []
        self.state_cov_history = []
        self.param_cov_history = []
        self.param_mu_history = []
        self.action_history = []
        self.z_mu_history = []
        self.z_std_history = []
        self.img_history = []
        self.rollout_history = []
        self.viewer_history = []

        # Initialise prior for state
        self.x_mu = torch.zeros(1, self.state_dim, device=self.device)
        self.x_sigma = self.model_lib['cartpole'].cfg.prior_cov * torch.eye(self.state_dim,
                                                                            device=self.device).unsqueeze(0)

        self.x_sigma[self.state_dim:, self.state_dim:] *= 0.3

        # First step -- we get an initial observation
        observation = self.env.reset()
        self.model_lib['cartpole'].cost_fn.set_goal(goal=self.env.get_goal())
        observation, _, _, info = self.env.step(np.zeros(self.model_lib['cartpole'].cfg.action_dimension))
        self.true_state_history.append(info['state'])
        self.observation_update(observation)

    def render(self):
        raise NotImplementedError

    def vizualise(self):
        raise NotImplementedError

    def observation_update(self, observation, true_state=None):
        # Encode single image
        _, _, z_mu, z_std = self.model_lib['cartpole'].perception(observation)
        z = z_mu
        # Use R from encoder
        R = None
        # Update belief in world
        self.x_mu, self.x_sigma = self.model_lib['cartpole'].trans_dist.update(z, self.x_mu, self.x_sigma, R)
        return z_mu, z_std

    def do_episode(self, action_noise=False):
        with torch.no_grad():
            while True:
                self.reset_trial()
                done = False
                fail = False
                cumulative_reward = 0.0
                try:
                    for t in range(0, self.episode_T, self.actions_per_loop):
                        done, fail, reward, info = self.step(action_noise)
                        cumulative_reward += reward

                        if done or fail:
                            break
                    if t < 5:
                        continue
                    break
                except Exception as e:
                   print(e)
                   logging.debug("Caught exception in step")
                   continue

            fail = not info['success']

        return fail, t

    def step(self, action_noise=0):
        # get action from controller
        # Sample states
        #state = self.x_mu.repeat(self.config.controller_N, 1)
        #state = torch.from_numpy(self.true_state_history[-1]).to(device='cuda:0').repeat(self.config.controller_N, 1)
        #state = state.reshape(self.config.controller_N, self.state_dim)
        state = self.x_mu.reshape(1, -1)

        state = torch.cat((state, torch.zeros_like(state)), dim=1)
        # Get action
        actions, rollout = self.controller.command(state)
        actions.reshape(-1, self.action_dimension)

        #self.CostFn.compute_cost(rollout)

        #actions = torch.rand(*actions.shape, device=actions.device) * 2.0 - 1.0

        # Predict effect of action in world
        total_reward = 0
        observed_mu = self.x_mu
        self.save_rollout(rollout)
        for i in range(actions.size()[0]):
            #print(self.x_mu)

            #print('--')

            self.x_mu, self.x_sigma = self.model_lib['cartpole'].trans_dist.predict(actions[i].view(1, -1), self.x_mu, self.x_sigma)
            #print(self.x_mu)
            # Act in world and get observation
            #TODO for cartpole need to minus action
            observation, reward, done, info = self.env.step(actions[i].detach().cpu().numpy())
            true_state = info['state']
            #print(true_state[:5])
            total_reward += reward
            # Render and update
            z_mu, z_std = self.observation_update(observation)
            #print(z_mu)
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

    @staticmethod
    def chunk_trajectory(z_mu, z_std, u, H=5):
        '''
        Takes z_mu, z_std, u all tensors
        gets T x nz, T x nz, T x nu
        Chunks into N x H x nz/nu
        Even split trajectories (discards extra data)
        '''

        T = min(z_mu.size(0), u.size(0))
        N = T // H
        z_mu = z_mu[:N * H].view(N, H, -1)
        z_std = z_std[:N * H].view(N, H, -1)
        u = u[:N * H].view(N, H, -1)

        return z_mu, z_std, u

    def store_episode_data(self):
        u = torch.from_numpy(np.asarray(self.action_history)).squeeze(1)
        z_mu = torch.from_numpy(np.asarray(self.z_mu_history)).squeeze(1)
        z_std = torch.from_numpy(np.asarray(self.z_std_history)).squeeze(1)

        z_mu, z_std, u = self.chunk_trajectory(z_mu, z_std, u)

        # First iteration while building online dataset
        if self.model_lib['cartpole'].trans_dist.saved_data is None:
            self.model_lib['cartpole'].trans_dist.saved_data = dict()
            self.model_lib['cartpole'].trans_dist.saved_data['z_mu'] = z_mu
            self.model_lib['cartpole'].trans_dist.saved_data['z_std'] = z_std
            self.model_lib['cartpole'].trans_dist.saved_data['u'] = u
        else:
            self.model_lib['cartpole'].trans_dist.saved_data['z_mu'] = torch.cat((z_mu, self.model_lib['cartpole'].trans_dist.saved_data['z_mu']), dim=0)
            self.model_lib['cartpole'].trans_dist.saved_data['z_std'] = torch.cat((z_std, self.model_lib['cartpole'].trans_dist.saved_data['z_std']), dim=0)
            self.model_lib['cartpole'].trans_dist.saved_data['u'] = torch.cat((u, self.model_lib['cartpole'].trans_dist.saved_data['u']), dim=0)

        print(self.model_lib['cartpole'].trans_dist.saved_data['z_mu'].shape)

    def train_on_episode(self):
        #u = torch.from_numpy(np.asarray(self.action_history)).to(device=self.config.device).squeeze(1)
        #z_mu = torch.from_numpy(np.asarray(self.z_mu_history)).to(device=self.config.device).squeeze(1)
        #z_std = torch.from_numpy(np.asarray(self.z_std_history)).to(device=self.config.device).squeeze(1)
        self.model_lib['cartpole'].trans_dist.train_on_episode()
        self.model_lib['cartpole'].cost_fn.set_max_std(self.model_lib['cartpole'].trans_dist.max_std)
        self.model_lib['cartpole'].cost_fn.iter = min(self.model_lib['cartpole'].cost_fn.iter + 1, 10)
        #z #zelf.CostFn.iter = 20
        #self.CostFn.iter = 10
        print(self.model_lib['cartpole'].cost_fn.iter)

    def save_episode_data(self, fname):
        # TODO sort this out with proper file directories and stuff
        actions = np.asarray(self.action_history)
        z_mu = np.asarray(self.z_mu_history)
        z_std = np.asarray(self.z_std_history)
        obs = np.asarray(self.img_history)
        goal = np.asarray(self.env.get_goal())
        rollout_hist = np.asarray(self.rollout_history)
        view_history = np.asarray(self.viewer_history)
        true_history = np.asarray(self.true_state_history)
        state_mu_history = np.asarray(self.state_mu_history)
        state_cov_history = np.asarray(self.state_cov_history)

        np.savez('{}.npz'.format(fname), z_mu, z_std, actions, obs, goal, rollout_hist,
                 view_history, true_history, state_mu_history, state_cov_history)

    def load_data(self):
        #TODO add name to argument
        for i in range(20):
            foldername = '../data/trajectories/victor_rope_flossing_no_dr/online/'
            filename = 'with_gp_thick_rope_trial_0_ep_{}.npz'.format(i+1)
            data = np.load(foldername + filename)
            self.z_mu_history = data['arr_0']
            self.z_std_history = data['arr_1']
            self.action_history = data['arr_2']
            self.store_episode_data()

    def save_rollout(self, rollout):
        self.rollout_history.append(rollout[0].cpu().numpy())
