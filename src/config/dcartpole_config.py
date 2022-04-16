from src.config import CommonEncConfig


class Config(CommonEncConfig):
    def __init__(self, data_folder: str = None):
        super(Config, self).__init__(data_folder)

        # Needed for both training and assembling encoder
        # Dimension of complete state needed to perform planning with
        self.state_dimension = 10

        # Number of position only states
        self.nqpos = 5

        # Number of observable states out of state_dims
        #  This is the number of outputs we train ensemble encoder NN to predict
        #  The quantities that are observable depend on the number of
        # consecutive simple model frames state encoder is trained with
        # If using 1 frame
        # self.observation_dimension = 5
        # If using 2frame
        self.observation_dimension = 10

        # Currently simple model library has single action dimension for all models
        self.action_dimension = 1

        # Simple Model Perception Training
        self.epochs = 80
        self.batch_size = 128
        self.lr_init = 3e-3 # Works for 1frame
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 20
        self.optimiser = 'adam'
