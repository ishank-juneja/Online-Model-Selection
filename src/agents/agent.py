''' Agent for acting in environment'''

import gym
import gym_cenvs
import numpy as np
import torch
from src.models.UKVAE import UnscentedKalmanVariationalAutoencoder
from src.models.GPUKVAE import GPUnscentedKalmanVariationalAutoencoder
from src.models.UKVAE_NN import UnscentedKalmanVariationalAutoencoderNNUncertainty
from pytorch_mppi import mppi
from pytorch_cem import cem


class Agent:

    def __init__(self, config, load_name=None, fixed_environment=False):
        self.config = config
        self.env = gym.make(self.config.env, fixed_environment=fixed_environment)
        self.env.unwrapped.symbolic = False

        if self.config.do_sys_id:
            self.state_dim = self.config.state_dimension + self.config.param_dimension
        else:
            self.state_dim = self.config.state_dimension

        # Load model
        if self.config.model_type == 'ukvae':
            if self.config.use_online_GP:
                self.model = GPUnscentedKalmanVariationalAutoencoder(config, load_name).to(device=self.config.device)
            elif self.config.use_online_NN:
                self.model = UnscentedKalmanVariationalAutoencoderNNUncertainty(config, load_name).to(device=self.config.device)
            else:
                self.model = UnscentedKalmanVariationalAutoencoder(config, load_name).to(device=self.config.device)
        else:
            raise Exception('Invalid model type')

        self.model.eval_mode()
        # Configure controller
        # Define goal
        goal = self.env.get_goal()
        self.CostFn = self.config.cost_fn(goal)#CartpoleTipCost(goal)
        self.RunningCost = None

        if self.config.controller == 'mppi':
            self.controller = mppi.MPPI(self.model.sample_dynamics, self.RunningCost, self.state_dim*2,
                                        noise_sigma=self.config.mppi_noise_sigma, num_samples=self.config.controller_N,
                                        horizon=self.config.controller_H, lambda_=self.config.mppi_lambda,
                                        device=self.config.device, terminal_state_cost=self.CostFn.compute_cost,
                                        u_scale=self.config.u_scale, u_max=1.0, u_min=-1.0,
                                        u_per_command=self.config.actions_per_loop)
        elif self.config.controller == 'cem':
            self.controller = cem.CEM(self.model.sample_dynamics, self.CostFn, self.state_dim,
                                      self.config.action_dimension, num_samples=self.config.controller_N,
                                      num_elite=self.config.controller_cem_K, horizon=self.config.controller_H,
                                      device=self.config.device, num_iterations=self.config.controller_cem_iters,
                                      u_min=self.config.action_min, u_max=self.config.action_max)

        #with torch.no_grad():
        #    self.reset_trial()

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
        self.x_mu = torch.zeros(1, self.state_dim, device=self.config.device)
        self.x_sigma = self.config.prior_cov * torch.eye(self.state_dim,
                                                         device=self.config.device).unsqueeze(0)

        self.x_sigma[self.config.state_dimension:, self.config.state_dimension:] *= 0.3

        # First step -- we get an initial observation
        observation = self.env.reset()
        self.CostFn.set_goal(self.env.get_goal())
        observation, _, _, info = self.env.step(np.zeros(self.config.action_dimension))
        self.true_state_history.append(info['state'])
        self.observation_update(observation)

    def render(self):
        raise NotImplementedError

    def vizualise(self):
        raise NotImplementedError

    def observation_update(self, observation, true_state=None):
        # Encode single image
        z_mu, z_std = self.model.encode_single_observation(observation)
        z = z_mu
        # Use R from encoder
        R = None
        # Update belief in world
        self.x_mu, self.x_sigma = self.model.update(z, self.x_mu, self.x_sigma, R)
        if self.config.viewer:
            self.viewer_history.append(self.env.get_view())

        return z_mu, z_std

    def do_episode(self, action_noise=False):
        with torch.no_grad():
            while True:
                self.reset_trial()
                done = False
                fail = False
                cumulative_reward = 0.0
                try:
                    for t in range(0, self.config.episode_T, self.config.actions_per_loop):
                        done, fail, reward, info = self.step(action_noise)
                        cumulative_reward += reward

                        if done or fail:
                            break
                    if t < 5:
                        continue
                    break
                except Exception as e:
                   print(e)
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

        # Some stuff to do with sys id #TODO tidy up
        if self.config.do_sys_id and self.config.param_map_estimate:
            mode_params = self.x_mu[0, self.config.state_dimension:] - \
                          torch.diag(self.x_sigma[0])[self.config.state_dimension:]

            state[:, self.config.state_dimension:] = mode_params.view(1, -1).repeat(self.config.controller_N, 1)

        state = torch.cat((state, torch.zeros_like(state)), dim=1)
        # Get action
        actions, rollout = self.controller.command(state)
        actions.reshape(-1, self.config.action_dimension)

        #self.CostFn.compute_cost(rollout)

        #actions = torch.rand(*actions.shape, device=actions.device) * 2.0 - 1.0

        # Predict effect of action in world
        total_reward = 0
        observed_mu = self.x_mu
        self.save_rollout(rollout)
        for i in range(actions.size()[0]):
            #print(self.x_mu)

            #print('--')

            self.x_mu, self.x_sigma = self.model.predict(actions[i].view(1, -1), self.x_mu, self.x_sigma)
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
            self.state_cov_history.append(self.x_sigma[0, :self.config.state_dimension,
                                          :self.config.state_dimension].detach().cpu().numpy())
            self.state_mu_history.append(self.x_mu[0, :self.config.state_dimension].detach().cpu().numpy())
            self.param_cov_history.append(torch.diag(self.x_sigma[0])[self.config.state_dimension:].detach().cpu().numpy())
            self.param_mu_history.append(self.x_mu[0, self.config.state_dimension:].detach().cpu().numpy())

            if self.config.viz_control:
                self.vizualise()
            #import time
            #time.sleep(0.1)

            if done:
                return done, False, total_reward, info

        return done, False, total_reward, info

    def store_episode_data(self):
        u = torch.from_numpy(np.asarray(self.action_history)).squeeze(1)
        z_mu = torch.from_numpy(np.asarray(self.z_mu_history)).squeeze(1)
        z_std = torch.from_numpy(np.asarray(self.z_std_history)).squeeze(1)
        from src.utils import chunk_trajectory
        z_mu, z_std, u = chunk_trajectory(z_mu, z_std, u)

        if self.model.saved_data is None:
            self.model.saved_data = dict()
            self.model.saved_data['z_mu'] = z_mu
            self.model.saved_data['z_std'] = z_std
            self.model.saved_data['u'] = u
        else:
            self.model.saved_data['z_mu'] = torch.cat((z_mu, self.model.saved_data['z_mu']), dim=0)
            self.model.saved_data['z_std'] = torch.cat((z_std, self.model.saved_data['z_std']), dim=0)
            self.model.saved_data['u'] = torch.cat((u, self.model.saved_data['u']), dim=0)

        print(self.model.saved_data['z_mu'].shape)

    def train_on_episode(self):
        #u = torch.from_numpy(np.asarray(self.action_history)).to(device=self.config.device).squeeze(1)
        #z_mu = torch.from_numpy(np.asarray(self.z_mu_history)).to(device=self.config.device).squeeze(1)
        #z_std = torch.from_numpy(np.asarray(self.z_std_history)).to(device=self.config.device).squeeze(1)
        self.model.train_on_episode()
        self.CostFn.set_max_std(self.model.max_std)
        self.CostFn.iter = min(self.CostFn.iter + 1, 10)
        #z #zelf.CostFn.iter = 20
        #self.CostFn.iter = 10
        print(self.CostFn.iter)

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
        if self.config.save_view:
            self.create_animation()
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

    def load(self, load_name):
        self.model.load_model(load_name)

    def create_animation(self):
        raise NotImplementedError


