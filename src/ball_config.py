from src.dynamics_models import CartPoleDynamics
from src.agents.costs import CartpoleTipCost
from src.image_trajectory_dataset import preprocess_identity
import torch


class DataConfig:
    """Config for all data related stuff"""
    def __init__(self, model_name: str):
        self.data_dir = 'data/{0}'.format(model_name)
        self.preprocess_state_fn = preprocess_identity
        self.preprocess_obs_fn = preprocess_identity
        self.imsize = 64
        # Use grayscale images
        self.grey = True
        # No depth frames
        self.depth = False


class Config:
    def __init__(self):
        self.data_config = DataConfig('MujocoBall-v0')
        self.model_type = 'ukvae'
        # self.agent = ConkersAgent
        self.device = 'cuda:0'
        self.state_dimension = 2
        self.dim_angles = 0 # Number of angles
        # Number of observable states out of state_dims
        self.observation_dimension = 2
        self.action_dimension = 1
        self.param_dimension = 2
        self.log_params = True
        self.do_sys_id = False# Only for control task -- deprecated (doesn't work v well)
        self.seed = 1234
        self.use_sqrt_ukf = False
        self.param_map_estimate = False
        self.load_name = 'model_Jan17_10-22-20'

        # Encoder uncertainty
        self.mc_dropout = False
        self.use_ensembles = True
        self.num_ensembles = 10
        self.dropout_p = 0.0

        # Training
        self.epochs = 80
        self.batch_size = 128
        self.lr_init = 2e-3
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 20
        self.optimiser = 'adam'
        self.vae_only_epochs = 0
        self.num_workers = 4

        # Loss terms
        # Recon loss
        self.recon_weight_init = 1.0
        self.recon_weight_min = 0.1
        self.recon_weight_decay_rate = 0.9
        self.recon_weight_decay_steps = 10000
        self.img_distribution = 'bernoulli'
        self.train_decoder = False
        self.train_encoder = True

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
        self.emission_noise = 0.03
        self.transition_noise = .1 * torch.ones(self.state_dimension, device=self.device)
        #self.transition_noise = torch.ones(self.state_dimension, device=self.device) * 0.1
        self.params_noise = 1e-2 * torch.ones(self.param_dimension, device=self.device)
        if self.do_sys_id:
            self.transition_noise = torch.cat((self.transition_noise, self.params_noise), dim=0)

        self.cap_z_var = False
        self.prior_cov = 1.0

        # Images
        self.grey = True
        self.imsize = 64

        # Dynamics
        self.dynamics_fn = CartPoleDynamics
        self.learn_dynamics = False #True
        self.learn_emission = False
        self.linear_emission = True

        # Task configuration
        #self.env = 'MujocoCartpole-v0'
        self.env = 'Conkers-v0'
        self.env_is_mujoco = True
        self.controller = 'mppi'

        self.save_view = False
        self.episode_fname = '../data/trajectories/conkers_diverse/online/test_with_gp_2'
        self.save_episode_data = True
        self.viz_control = False
        self.viewer = False
        self.num_episodes = 20
        self.num_trials = 3
        self.num_tests = 20
        self.episode_T = 100
        self.state_cost_mat = torch.eye(self.state_dimension, device=self.device)
        self.state_cost_mat = torch.diag(torch.tensor([0.5, 0.001, 1., 1., 0.001], device=self.device))
        self.terminal_cost_mat = torch.diag(torch.tensor([3., 0.01, 5., 5., 0.01], device=self.device))

        self.cost_fn = CartpoleTipCost
        self.control_cost_mat = 0.00001 * torch.eye(self.action_dimension, device=self.device)
        self.controller_N = 1000
        self.controller_H = 20
        self.actions_per_loop = 1

        # MPPI params
        self.mppi_noise_sigma = .5 * torch.eye(self.action_dimension, device=self.device)
        self.mppi_lambda = 1.
        self.u_scale = 1.0

        # CEM params
        self.controller_cem_K = 100
        self.action_min = torch.tensor(-1.0, device=self.device)
        self.action_max = torch.tensor(1.0, device=self.device)
        self.controller_cem_iters = 10

        # Online GP learning
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
