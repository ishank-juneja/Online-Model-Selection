import logging
import numpy as np
# TODO: Have a better way of importing the configs
from src.config import PerceptionConfig
from src.learned_models import SimpModPerception
from src.transition_distributions import HeuristicUnscentedKalman
import torch


class SimpModBook:
    """
    Class to encapsulate the attributes associated with a single simple model
    Collection of SimpModBooks is a SimpModLib
    Attributes:
    1. Kinematic/dynamic/kinodynamic transition function (trans_fn)
    2. Object to query for current cost (cost_fn)
    TODO: Emph. via code/comments that the filter-transition distribution
     is the most important piece of the Book and most of the book keeping is in fact just happening
     in self.trans with the Book only performing a few extra tasks like choosing the right cost function
     given the task and model_name and providing a class that can be duplicated easily for every model to form the library
    3. A combined filter-transition distribution class (i.e. model p_z)
        example: (GPUKF, or UKF with hard-coded transition uncertainty scheme)
    4. A perception that maps the current observation to state and uncertainty phi(o_t) -> mu_y, Sigma_y
    """
    # TODO: Add more init parameters when adding planning
    def __init__(self, simp_mod: str):
        """
        :param simp_mod: string of simple model name
        """
        # Name of the simple model keep-ed by this book
        self.name = simp_mod

        # Create perception file_name config object for initializing the perception of this simple model
        per_config = PerceptionConfig()
        # Load in perception object
        self.perception = SimpModPerception(**per_config[self.name])

        # Config for simple model book-keeping is identical to config from perception
        self.cfg = self.perception.get_config()

        # Dynamics function is usually a simple dynamics relationship, can also be kinematic relationship
        # TODO: Add back dynamics relationship when adding back planning
        # self.dynamics_fn = self.cfg.dynamics_fn

        # Cost function is a task dependent attribute of a simple model that allows planner to use
        #  approximate simple models to plan action sequences. Requires task at invocation
        # Since it is task dependent it is set in task_agent
        # TODO: Invoke the correct cost function based on the TaskAgent that creates a SimpModLib object
        #  No cost fn for now since no planning
        # self.cost_fn = self.cfg.cost_fn(goal)

        # Transition distribution p_z is an online learned gaussian distribution parameterized as mu, Sigma
        #  Effectively it is a function from state and action to next state and predicted uncertainty
        #  p_z ~ N( hat{f}_{rho}(z_t, u_t), Q(z_t, u_t) )
        # self.trans_dist = transition_dist(self.cfg)

        self.trans_dist = HeuristicUnscentedKalman(self.cfg)

        # States and actions

        logging.info("Initialized struct and perception for {0} model".format(simp_mod))

    def state_dim(self) -> int:
        return self.cfg.state_dimension

    def train_on_episode(self):
        #u = torch.from_numpy(np.asarray(self.action_history)).to(device=self.config.device).squeeze(1)
        #z_mu = torch.from_numpy(np.asarray(self.z_mu_history)).to(device=self.config.device).squeeze(1)
        #z_std = torch.from_numpy(np.asarray(self.z_std_history)).to(device=self.config.device).squeeze(1)
        self.trans_dist.train_on_episode()
        self.cost_fn.set_max_std(self.trans_dist.max_std)
        self.cost_fn.iter = min(self.cost_fn.iter + 1, 10)
        #z #zelf.CostFn.iter = 20
        #self.CostFn.iter = 10
        print(self.cost_fn.iter)

    @staticmethod
    def chunk_trajectory(z_mu, z_std, u, H=5):
        """
        Takes z_mu, z_std, u as tensors of shapes T x nz, T x nz, T x nu resp.
        Divides these trajectories into chunks of size N x H x nz (for state/uncertainty)
         and N x H x nu (for actions)
        Evenly splits the trajectories (discards extra trailing data)
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

    def reset_epsidoe(self):
        """
        Reset only the episode specific state for this simple model
        :return:
        """
        # Run the trial reset method of the underlying transition distribution
        self.trans_dist.

    def hard_reset(self):
        """
        Reset all the state built up online for this simple model
        :return:
        """
        # Reset the online learned UKF related parameters
        self.trans_dist.transition.reset_params()
        # TODO: Increase the scope of the resets once planning etc. are added
