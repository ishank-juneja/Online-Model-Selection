from src.config import CommonEncConfig
from src.simp_mod_library.kinodynamic_funcs import CartpoleDynamics
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
        self.param_dim = 3
        # Whether to use params passed through log (better for learning)
        self.log_params = True
        # - - - - - - - - - - - - - - - - - - - -

        # - - - - - - - - - - - - - - - - - - - -
        # Actuation Related
        # Number of dimensions in gt robot state appended to simple model state
        self.rob_dim = 2
        # Currently simple model lib has single action dimension for all models
        self.action_dim = 1
        # - - - - - - - - - - - - - - - - - - - -

        # - - - - - - - - - - - - - - - - - - - -
        # Visualizations Related
        # Number of position only states
        self.nqpos = 3
        # - - - - - - - - - - - - - - - - - - - -

        # - - - - - - - - - - - - - - - - - - - -
        # Perception related
        if data_folder is not None:
            # TODO: Once choice of 1 frame / 2 frame is finalized remove this conditional stuff and its consequences
            # Number of observable states out of state_dims
            #  Refer to state returned by gym environment
            #  The quantities that are observable depend on the number of
            # consecutive simple model frames state encoder is trained with
            if self.nframes == 1:
                self.obs_dim = 3
                # Mask to remove things coming from perception for which GT is available
                self.obs_mask = [1, 2]
            else:
                self.obs_dim = 6
                # Mask to remove things coming from perception for which GT is available
                self.obs_mask = [1, 2, 4, 5]
            # Infer the dimensions of the observations as seen by the filter
            filter_obs_dim = len(self.obs_mask)
        else:
            filter_obs_dim = None
        # Training hparams
        self.epochs = 80
        self.batch_size = 64
        # Reduce learning rate if seeing nans while training
        self.lr_init = 3e-3
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 20
        self.optimiser = 'adam'
        # - - - - - - - - - - - - - - - - - - - -

        # - - - - - - - - - - - - - - - - - - - -
        # Filtering related
        # Dimension of observations passed to filter
        self.filter_obs_dim = filter_obs_dim
        # Needed for both training and assembling encoder
        # Dimension of filter-compatible state
        self.state_dim = 4
        # Variance settings
        # eps parameter in emission model of observations
        self.emission_noise = 0.03
        self.transition_noise = .1 * torch.ones(self.state_dim, device=self.device)
        self.prior_cov = 1.0
        # - - - - - - - - - - - - - - - - - - - -

        # - - - - - - - - - - - - - - - - - - - -
        # Sys - ID settings
        self.params_noise = 1e-2 * torch.ones(self.param_dim, device=self.device)
        # - - - - - - - - - - - - - - - - - - - -

        # - - - - - - - - - - - - - - - - - - - -
        # Dynamics related parameters
        self.dynamics_class = CartpoleDynamics
        # - - - - - - - - - - - - - - - - - - - -

        # - - - - - - - - - - - - - - - - - - - -
        # Planning related params
        self.mppi_noise_sigma = .5 * torch.eye(self.action_dim, device=self.device)
        self.mppi_lambda = 1.
        self.u_scale = 1.0
        self.beta_init = .7
        # - - - - - - - - - - - - - - - - - - - -
