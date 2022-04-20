import logging
from src.config import BallConfig, CartpoleConfig, DcartpoleConfig, DubinsConfig
from src.learned_models import SimpModPerception
from src.simp_mod_library.costs import CartpoleTipCost
from src.simp_mod_library.kinodynamic_funcs import CartPoleDynamics
from src.transition_distributions import HeuristicUnscentedKalman, GPUnscentedKalman
from typing import Type, Union


class SimpModStruct:
    def __init__(self, simp_mod: str, transition_dist: Union[Type[HeuristicUnscentedKalman],
                                                             Type[GPUnscentedKalman]], goal):
        """
        Class to encapsulate the attributes associated with a single simple model
        Attributes:
        1. Kinematic/dynamic/kinodynamic transition function (trans_fn)
        2. Object to query for current cost (cost_fn)
        3. A combined filter-transition distribution class (i.e. model p_z)
            example: (GPUKF, or UKF with hard-coded transition uncertainty scheme)
        4. A perception that maps the current observation to state and uncertainty phi(o_t) -> mu_y, Sigma_y
        :param simp_mod: string of simple model name
        :param transition_dist: Class in Union[GPUKF, UKF]
        """
        # Initially create temp malformed config and after loading in perception infer full config
        if simp_mod == 'cartpole':
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
