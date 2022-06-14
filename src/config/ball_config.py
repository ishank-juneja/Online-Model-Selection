from src.config import CommonEncConfig
from src.simp_mod_library.kinodynamic_funcs import BallDynamics
import torch


class Config(CommonEncConfig):
    def __init__(self, data_folder: str = None):
        super(Config, self).__init__(data_folder)

        # - - - - - - - - - - - - - - - - - - - -
        # Sys-id-ed params related atts
        # Whether to perform sys-id at all
        self.do_sys_id = False
        # Number of parameters in the dynamics that are unobservable from images (example mass)
        #  and hence need to be sys-id-ed
        self.param_dimension = 1
        # Whether to use params passed through log (better for learning)
        self.log_params = True
        # - - - - - - - - - - - - - - - - - - - -

        # - - - - - - - - - - - - - - - - - - - -
        # Actuator related parameters
        # Number of dimensions in gt actuator state appended to simple model state
        self.actuator_dimension = 2
        # - - - - - - - - - - - - - - - - - - - -

        # Dim of state returned by gym environment
        self.state_dimension = 4

        # Number of position only states
        self.nqpos = 2

        if data_folder is not None:
            # Number of observable states out of state_dims
            #  Refer to state returned by gym env
            #  The quantities that are observable depend on the number of
            # consecutive simple model frames state encoder is trained with
            if self.nframes == 1:
                self.observation_dimension = 2
            else:
                self.observation_dimension = 4

            self.action_dimension = 1

        # Training
        self.epochs = 80
        self.batch_size = 64
        self.lr_init = 3e-3
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 20
        self.optimiser = 'adam'

        self.dynamics_class = BallDynamics

        # eps parameter in emission model of observations
        self.emission_noise = 0.03
        self.transition_noise = .1 * torch.ones(self.state_dimension, device=self.device)
        self.params_noise = 1e-2 * torch.ones(self.param_dimension, device=self.device)

        # Priors over params
        self.prior_cov = 1.0
