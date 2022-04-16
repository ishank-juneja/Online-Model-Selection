from src.config import CommonEncConfig


class Config(CommonEncConfig):
    def __init__(self, data_folder: str = None):
        super(Config, self).__init__(data_folder)

        # Needed for both training and assembling encoder
        # Dimension of complete state needed to perform planning with
        self.state_dimension = 6

        # Number of position only states
        self.nqpos = 3

        # Number of observable states out of state_dims
        #  Refer to state returned by gym environment
        #  The quantities that are observable depend on the number of
        # consecutive simple model frames state encoder is trained with
        # If using 1 frame
        # self.observation_dimension = 3
        # If using 2frame
        self.observation_dimension = 6

        # Currently simple model library has single action dimension for all models
        self.action_dimension = 1

        # Simple Model Perception Training
        self.epochs = 80
        self.batch_size = 64
        # Reduce learning rate if seeing nans while training
        self.lr_init = 3e-3
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 20
        self.optimiser = 'adam'
