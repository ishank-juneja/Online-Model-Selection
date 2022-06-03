import gym
import gym_cenvs
from abc import ABCMeta
import logging
import numpy as np
from src.simp_mod_library.simp_mod_lib import SimpModLib
from src.transition_distributions import JTM
import torch
from typing import List


class BaseAgent(metaclass=ABCMeta):
    """
    Base class for constructing agents that control the complex object using passed Simple Model Library
    """
    def __init__(self, smodel_list: List[str], device: str = 'cuda:0'):
        # String that is set in derived class
        self.env_name: str = None
        # Ref to step-able mujoco-gym environment
        self.env = None

        # Devices
        self.device = device

        self.model_lib = SimpModLib(smodel_list)

        self.mmt = JTM(self.model_lib, self.device)

        self.state_dim = self.model_lib['cartpole'].cfg.state_dimension
        self.device = self.model_lib['cartpole'].cfg.device

    def make_agent_for_task(self):
        """
        Invoked after task specific params have been set in derived class
        :return:
        """
        self.env = gym.make(self.env_name)
        self.env.seed(0)
        self.env.action_space.seed(0)

        # Actions per loop iteration / nrepeats for action
        self.actions_per_loop = 1
        # TODO: Infer action dimension from the environment
        #  Check consistency of this dimension with the controls dimension of simple model lib
        self.action_dimension = 1

        return

    def get_cost_fn(self, simp_model: str):
        """
        Returns the cost function corresponding to the passed simple model for the task at hand
        :param simp_model:
        :return:
        """
        return

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

    def render(self):
        raise NotImplementedError

    def visualize(self):
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

        # TODO: Eventually this should not be a loop ...
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

    # TODO: Modify load and save to handle data from all models as opposed to a single model
    def save_episode_to_disk(self, fname):
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
        # TODO add name to argument
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

    def reset_episode(self):
        # Clear action history for next episode
        self.action_history = []
        # Reset the episode-specific state of the simple model library
        self.model_lib.reset_episode()
