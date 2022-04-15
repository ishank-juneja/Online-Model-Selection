

class DataConfig:
    """
    Config for building torch dataloader
    Different from CommonEncConfig or
    """
    def __init__(self, simple_model_name: str):
        self.data_dir = 'data/{0}'.format(simple_model_name)
        self.imsize = 64


class CommonEncConfig:
    """
    Config parameters common to the encoders for all simple models
    """
    def __init__(self, data_folder: str = None):
        # Set the data_config to be used for training data loader
        self.data_config = DataConfig(data_folder)

        # Encoder uncertainty
        self.num_ensembles = 10

        if data_folder is not None:
            # Whether dataset is based on 1 frame or stacked 2 frame images of the simple model
            if '2frame' in data_folder:
                self.nframes = 2
            elif '1frame' in data_folder:
                self.nframes = 1
            else:
                raise NotImplementedError("data folder {0} does not fit in either "
                                          "1frame or 2 frame categories".format(data_folder))

        # Size of single frames seen by encoder
        self.imsize = 64
        # Training on local with single GPU
        self.device = 'cuda:0'
        # Make training deterministic
        #  Different from seed used for dataset creation and gym environments
        self.seed = 0

        # Number of workers in data-loader
        self.num_workers = 2

        # Camera matrix path for visualization
        #  All models except for dubins car share their camera matrix
        self.cam_mat_path = "data/cam_matrix.npy"
