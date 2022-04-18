from src.config import CommonEncConfig


class Config(CommonEncConfig):
    def __init__(self, data_folder: str = None):
        super(Config, self).__init__(data_folder)

        # Needed for both training and assembling encoder
        # Dimension of complete state needed to perform planning with
        self.state_dimension = 5

        # Number of position only states (nstates that can be inferred from a single frame)
        #  given the geometry
        self.nqpos = 4

        if data_folder is not None:
            # Number of observable states out of state_dims
            #  Refer to state returned by gym environment
            #  The quantities that are observable depend on the number of
            # consecutive simple model frames state encoder is trained with
            if self.nframes == 1:
                self.observation_dimension = 2
            else:
                self.observation_dimension = 5

        # Currently simple model lib has single action dimension for all models
        self.action_dimension = 1

        # Simple Model Perception Training
        self.epochs = 80
        self.batch_size = 64
        self.lr_init = 3e-3
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 20
        self.optimiser = 'adam'

        # Requires a distinct camera matrix since motion is in x-y plane and so
        #  camera faces the x-y plane head on as opposed to the x-z plane
        # overrides base class cam_mat_path
        self.cam_mat_path = "data/cam_matrix_dubins.npy"
