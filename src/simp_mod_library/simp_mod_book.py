import logging
import numpy as np
# TODO: Have a better way of importing the configs
from src.config.ball_config import Config as BallConfig
from src.config.cartpole_config import Config as CartpoleConfig
from src.learned_models import SimpModPerception
from src.simp_mod_library.costs import CartpoleTipCost
from src.simp_mod_library.kinodynamic_funcs import CartpoleDynamics
from src.transition_distributions import HeuristicUnscentedKalman, GPUnscentedKalman
import torch
from typing import Type, Union


class SimpModBook:
    """
    Class to encapsulate the attributes associated with a single simple model
    Collection of SimpModBook s is a SimpModLib
    Attributes:
    1. Kinematic/dynamic/kinodynamic transition function (trans_fn)
    2. Object to query for current cost (cost_fn)
    3. A combined filter-transition distribution class (i.e. model p_z)
        example: (GPUKF, or UKF with hard-coded transition uncertainty scheme)
    4. A perception that maps the current observation to state and uncertainty phi(o_t) -> mu_y, Sigma_y
    """
    def __init__(self, simp_mod: str, transition_dist: Union[Type[HeuristicUnscentedKalman],
                                                             Type[GPUnscentedKalman]], goal):
        """
        :param simp_mod: string of simple model name
        :param transition_dist: Class in Union[GPUKF, UKF]
        """
        # Initially create temp malformed config and after loading in perception infer full config
        if simp_mod == 'ball':
            tmp_cfg = BallConfig()
        elif simp_mod == 'cartpole':
            tmp_cfg = CartpoleConfig()
        else:
            raise NotImplementedError

        # Load in perception object
        self.perception = SimpModPerception(**tmp_cfg.perception)

        self.cfg = self.perception.encoder.get_config()

        # Dynamics function is usually a simple dynamics relationship, can also be kinematic relationship
        self.dynamics_fn = self.cfg.dynamics_fn

        # Cost function is a task dependent attribute of a simple model that allows planner to use
        #  approximate simple models to plan action sequences. Requires task at invocation
        # Since it is task dependent it is set in task_agent
        self.cost_fn = self.cfg.cost_fn(goal)

        # Transition distribution p_z is an online learned gaussian distribution parameterized as mu, Sigma
        #  Effectively it is a function from state and action to next state and predicted uncertainty
        #  p_z ~ N( hat{f}_{rho}(z_t, u_t), Q(z_t, u_t) )
        self.trans_dist = transition_dist(self.cfg)

        logging.info("Initialized struct and perception for {0} model".format(simp_mod))

    def state_dim(self) -> int:
        return self.cfg.state_dimension

    def mppi_noise_sigma(self) -> float:
        return self.cfg.mppi_noise_sigma

    def mppi_lambda(self) -> float:
        return self.cfg.mppi_lambda

    @staticmethod
    def chunk_trajectory(z_mu, z_std, u, H=5):
        """
        Takes z_mu, z_std, u as tensors of shapes T x nz, T x nz, T x nu resp.
        Divides these trajectories into chunks of size N x H x nz/nu
        Even splits the trajectories (discards extra trailing data)
        :param z_mu: Mean over-states returned by perception
        :param z_std: Std-dev over state validity returned by perception
        :param u: actions
        :param H: chunk size
        :return:
        """
        # TODO: This may be redundant if both are guaranteed same T
        T = min(z_mu.size(0), u.size(0))
        N = T // H

        # Truncate and view as desired shape
        z_mu_chunks = z_mu[:N * H].view(N, H, -1)
        z_std_chunks = z_std[:N * H].view(N, H, -1)
        u_chunks = u[:N * H].view(N, H, -1)

        return z_mu_chunks, z_std_chunks, u_chunks

    def append_to_dataset(self):
        """
        Agent appends current dataset with observed data from the perception on the entire episode
        D <- D U (mu^y_t, Sigma^y_t, u_t)_{t=1}^{T} where T < self.episode_T = duration of trajectory
        :return:
        """
        u = torch.from_numpy(np.asarray(self.action_history))    # size: T
        z_mu = torch.from_numpy(np.asarray(self.z_mu_history)).squeeze(1)   # size: T x obs_dim
        z_std = torch.from_numpy(np.asarray(self.z_std_history)).squeeze(1) # size: T x obs_dim

        z_mu, z_std, u = self.chunk_trajectory(z_mu, z_std, u)

        # First iteration while building online dataset
        if self.model_lib['cartpole'].trans_dist.saved_data is None:
            # Initialize dataset as empty dict
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
