from src.dynamics_models import CartPoleDynamics
from src.agents.costs import CartpoleTipCost
from src.image_trajectory_dataset import preprocess_identity
import torch


class CartpoleDataConfig:
    """
    Config for simple model perception training data
    """
    def __init__(self, model_name: str):
        # Dir where training data frames are available
        self.data_dir = 'data/{0}'.format(model_name)
        # Preprocess states and observations before using for training
        self.preprocess_state_fn = preprocess_identity
        self.preprocess_obs_fn = preprocess_identity
        # Size of square frames used a straining data
        self.imsize = 64
        # Use grayscale images
        self.grey = True
        # Whether depth frames are part of training data
        self.depth = False


# Configuration parameters for perceiving a Cartpole as a simple model and using it for control
class Config:
    def __init__(self):
        # config for training data
        self.data_config = CartpoleDataConfig('MujocoCartpole-v0')
        # Full model choice for being able to do full state estimation from sequence of frames
        self.model_type = 'ukvae'
        # Agent that willpotentially use this perception for control
        # TODO: Remove once confiremed that this is not required
        # self.agent = ConkersAgent
        # GPU device name
        self.device = 'cuda:0'
        # Total number of states [x_cart, x_mass, y_mass, v_cart, theta_dot_mass]
        self.state_dimension = 5
        # Number of angles in state
        self.dim_angles = 1
        # Number of observable states out of state_dims
        self.observation_dimension = 3
        self.action_dimension = 1
        # Number of parameters for which the GP tracks the uncertainty in the state estimate
        self.param_dimension = 3
        # Use logarithmic value of params
        self.log_params = True
        # Only for control task -- deprecated (doesn't work v well)
        # TODO: Retained for it to be defined in code, remove at later stage
        self.do_sys_id = False
        self.seed = 1234
        self.use_sqrt_ukf = False
        self.param_map_estimate = False
        # Once the final or right perception has been trained , specify model name here
        self.load_name = 'model_Jan17_10-22-20'

        # Encoder related parameters
        self.mc_dropout = False
        self.use_ensembles = True
        # Number of individual state-encoders in ensemble
        self.num_ensembles = 10
        self.dropout_p = 0.0

        # Training Loop
        self.epochs = 80
        self.batch_size = 128
        # Learning rate related
        self.lr_init = 5e-3
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 20
        self.optimiser = 'adam'
        # TODO: remove ref to below once confirmed redundant
        self.vae_only_epochs = 0
        # Multiprocessing data loading with the specified number of processes
        self.num_workers = 4

        # Loss related config params
        # Recon loss
        self.recon_weight_init = 1.0
        self.recon_weight_min = 0.1
        self.recon_weight_decay_rate = 0.9
        self.recon_weight_decay_steps = 10000
        # TODO: Decoder related param depracated
        self.img_distribution = 'bernoulli'
        self.train_decoder = False
        self.train_encoder = True

        # TODO: Most things under here are deprecated
        # Dynamics loss
        self.img_cov = 0.1 # For case when img distribution chosen to be Gaussian
        self.use_dynamics_loss = False
        self.dynamics_loss_weight_init = 1.0
        self.dynamics_loss_anneal_rate = 10.0
        self.dynamics_loss_anneal_steps = 1000000
        self.use_true_pos = True
        self.true_pos_weight = 1.0
        self.entropy_weight = 0.0
        self.sample_z = False

        # Variance settings
        # Emission (Observation) noise param for kalman filter (R)
        self.emission_noise = 0.03
        # Transition noise for kalman filter (Q)
        self.transition_noise = .1 * torch.ones(self.state_dimension, device=self.device)
        self.params_noise = 1e-2 * torch.ones(self.param_dimension, device=self.device)
        if self.do_sys_id:
            self.transition_noise = torch.cat((self.transition_noise, self.params_noise), dim=0)
        # Artificially cap variance on full state z
        self.cap_z_var = False
        # Prior covariance over state estimates for KF
        self.prior_cov = 1.0

        # For Observation Images
        self.grey = True
        self.imsize = 64

        # Dynamics
        self.dynamics_fn = CartPoleDynamics
        self.learn_dynamics = False
        self.learn_emission = False
        # Use linear emission model in KF
        self.linear_emission = True

        # Task for which this configuration is written
        # self.env = 'MujocoCartpole-v0'
        self.env = 'Conkers-v0'
        self.env_is_mujoco = True
        self.controller = 'mppi'

        # Online Learning related parameters
        self.save_view = False
        # TODO: Revise when running experiments
        self.episode_fname = '../data/trajectories/conkers_diverse/online/test_with_gp_2'
        self.save_episode_data = True
        self.viz_control = False
        self.viewer = False
        self.num_episodes = 20
        self.num_trials = 3
        self.num_tests = 20
        self.episode_T = 100
        # TODO: Remove below once hard coded numbers known good
        # self.state_cost_mat = torch.eye(self.state_dimension, device=self.device)
        self.state_cost_mat = torch.diag(torch.tensor([0.5, 0.001, 1., 1., 0.001], device=self.device))
        self.terminal_cost_mat = torch.diag(torch.tensor([3., 0.01, 5., 5., 0.01], device=self.device))

        # Control parameters
        self.cost_fn = CartpoleTipCost
        self.control_cost_mat = 0.00001 * torch.eye(self.action_dimension, device=self.device)
        self.controller_N = 1000
        self.controller_H = 20
        self.actions_per_loop = 1
        # MPPI specfic params
        self.mppi_noise_sigma = .5 * torch.eye(self.action_dimension, device=self.device)
        self.mppi_lambda = 1.
        self.u_scale = 1.0
        # CEM specific params
        self.controller_cem_K = 100
        self.action_min = torch.tensor(-1.0, device=self.device)
        self.action_max = torch.tensor(1.0, device=self.device)
        self.controller_cem_iters = 10

        # Paramas for doing online GP learning with this simple model
        self.do_online_learning = True
        self.use_online_GP = True
        self.use_online_NN = False
        self.fit_params_episodic = False
        self.online_epochs = 100
        self.online_lr = 1e-1
        self.online_samples = 1
        self.overshoot_d = 1
        self.beta_init = .7
        self.train_interval = 1
        self.test_first = True
        self.gp_zero_mean = True
