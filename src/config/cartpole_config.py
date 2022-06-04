from src.config import CommonEncConfig
from src.simp_mod_library.kinodynamic_funcs import CartpoleDynamics
import torch


class Config(CommonEncConfig):
    def __init__(self, data_folder: str = None):
        super(Config, self).__init__(data_folder)

        # Number of parameters in the dynamics that are unobservable from images (example mass)
        self.param_dimension = 3

        # Needed for both training and assembling encoder
        # Dimension of complete state needed to perform planning with
        self.state_dimension = 5

        # Number of position only states
        self.nqpos = 3

        if data_folder is not None:
            # Number of observable states out of state_dims
            #  Refer to state returned by gym environment
            #  The quantities that are observable depend on the number of
            # consecutive simple model frames state encoder is trained with
            if self.nframes == 1:
                self.observation_dimension = 3
            else:
                self.observation_dimension = 6

        # Currently simple model lib has single action dimension for all models
        self.action_dimension = 1

        # Simple Model Perception Training
        self.epochs = 80
        self.batch_size = 64
        # Reduce learning rate if seeing nans while training
        self.lr_init = 3e-3
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 20
        self.optimiser = 'adam'

        # Pointers to filenames of trained perceptions
        self.perception = {'encoder_model_name': "model_cartpole_enc_1frame_Apr16_18-53-27",
                           'seg_model_name': "model_cartpole_seg_1frame_MRCNN_Apr16_08-59-19"}

        # Variance settings

        # eps parameter in emission model of observations
        self.emission_noise = 0.03
        self.transition_noise = .1 * torch.ones(self.state_dimension, device=self.device)
        self.params_noise = 1e-2 * torch.ones(self.param_dimension, device=self.device)

        self.do_sys_id = False
        self.param_map_estimate = False

        self.dynamics_fn = CartpoleDynamics
        self.learn_dynamics = False
        self.learn_emission = False
        self.linear_emission = True

        self.log_params = True

        # MPPI params
        self.mppi_noise_sigma = .5 * torch.eye(self.action_dimension, device=self.device)
        self.mppi_lambda = 1.
        self.u_scale = 1.0

        self.beta_init = .7

        self.prior_cov = 1.0

        self.use_sqrt_ukf = False
