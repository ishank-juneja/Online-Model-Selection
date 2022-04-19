import numpy as np
from src.learned_models.encoder import Encoder
from src.utils import EncDataset
import torch
from torch import nn
from torchvision import transforms
import warnings


class EncoderEnsemble(nn.Module):
    def __init__(self, model_name: str, load_model: bool = False):
        super(EncoderEnsemble, self).__init__()

        # Infer the model parameters from the model being trained/loaded
        enc_dataset = self.disect_model(model_name)

        self.simp_model = enc_dataset.get_simp_model()
        # The kind of RGB frames architecture is trained to deal with
        self.nframes = enc_dataset.get_nframe()

        config = enc_dataset.get_enc_cfg()

        # Config parameters specific to the simple model we train on
        self.config = config

        # Whether model in testing mode
        self.test = False

        self.encoder = nn.ModuleList([])
        for i in range(self.config.num_ensembles):
            self.encoder.extend([Encoder(label_dim=self.config.observation_dimension, img_channels=3 * self.nframes)])

        # Load model
        # Have to load model before configuring GP stuff for state dimension stuff
        if load_model:
            self.load_model(model_name)

        self.saved_data = None
        self.start = 0

    def forward(self, x: np.ndarray):
        """
        :param x: np array of images with shape in [C x W x H, N x C x W x H, N x T x C x W x H], where
        C is number of channels in image, W, H are width and height of image, N is number of images/batch size,
        T is number of images in a trajectory as chosen while creating simple model datasets in make_enc_dataset.py
        Effective batch size (N) from neural net POV in N x T x C x W x H is (N x T) and 1 in C x W x H case
        :return: z_mu, z_std
        encode images to z_mu z_std using N_ensembles number of networks
        if test mode combines ensemble estimates into estimate size (N x T x nz)
        if train mode then returns (N_ensemble x N x T x nz) to train each ensemble independently
        """
        # TODO: Replace functionality of self.encode and self.encode_ensemble with this ...

    def get_obs_dim(self) -> int:
        """
        :return: Number of outputs neural network has been trained to output a distribution over
        """
        return self.config.observation_dimension

    def get_n_ensembles(self) -> int:
        """
        Number of members in the encoder ensemble
        :return:
        """
        return self.config.num_ensembles

    def get_config(self):
        return self.config

    def send_model_to_gpu(self):
        """
        Calling cuda on NN environment sends model to GPU
        :return:
        """
        self.encoder.cuda()

    def encode(self, observations):
        z_mu, z_std = self.encode_ensemble(observations)
        return z_mu, z_std

    def encode_ensemble(self, observations):
        """
        :param observations: N x T x C x W x H image tensors
        :return: encode images to z_mu z_std using N_ensembles number of networks
                z_mu, z_std
                if test mode combines ensemble estimates into estimate size (N x T x nz)
                if train mode then returns (N_ensemble x N x T x nz) to train each ensemble independently
        """
        z_mu = []
        z_logvar = []
        N, T, C, W, H = observations.shape
        o = observations.view(N*T, C, W, H)

        for i in range(self.config.num_ensembles):
            z_mu_tmp, z_logvar_tmp = torch.chunk(self.encoder[i](o), 2, dim=1)

            z_mu.append(z_mu_tmp)
            z_logvar.append(z_logvar_tmp)

        z_mu = torch.stack(z_mu, dim=0)
        z_logvar = torch.stack(z_logvar, dim=0)
        z_var = self.process_zlogvar(z_logvar)

        if self.test:
            z_mu, z_var = self.combine_estimates(z_mu, z_var)
            return z_mu.view(N, T, -1), z_var.sqrt().view(N, T, -1)

        N *= self.config.num_ensembles
        return z_mu.reshape(N, T, -1), z_var.sqrt().reshape(N, T, -1)

    def encode_single_obs(self, obs: np.ndarray):
        """
        Only used at test-time / online run-time not during traning
        :param obs: observation directly from environment -- is an unprocessed numpy array
        :return: z_mu, z_std for configuration estimate
        """
        o = self.preprocess_input(obs)
        o = o.view(1, 1, -1, self.config.imsize, self.config.imsize).to(device=self.config.device)
        z_mu, z_std = self.encode(o)
        return z_mu.reshape(1, -1), z_std.reshape(1, -1)

    def combine_estimates(self, z_mu, z_var):
        '''

        :param z_mu: set of N_ensembles x N x T x nz mean predictions
        :param z_var: set of N_ensembles x N x T x nz std predictiosn
        :return: combined prediction as gaussian approximation to mixture of N_ensemble gaussians
        '''
        # print("Using combined estimates")
        z_mu = z_mu.view(self.config.num_ensembles, -1, self.config.observation_dimension)
        z_var = z_var.view(self.config.num_ensembles, -1, self.config.observation_dimension)

        z_mu_total = z_mu.mean(dim=0)
        if not (z_mu_total == z_mu_total).all():
            print(z_mu_total)
            print(z_mu)
        z_var_total = z_var.mean(dim=0)
        z_var_total = z_var_total + (z_mu - z_mu_total).pow(2).mean(dim=0)

        return z_mu_total, z_var_total

    def process_zlogvar(self, z_logvar):
        return z_logvar.exp()

    def save_model(self, name):
        if name is None:
            warnings.warn('Saving but name is None')
        torch.save(self.state_dict(), 'models/encoder/{}.pt'.format(name))

    def load_model(self, model_name):
        self.load_state_dict(torch.load('models/encoder/{}.pt'.format(model_name), map_location=self.config.device))
        self.name = model_name

    def eval_mode(self):
        """
        Put model in eval mode
        """
        self.test = True
        self.eval()

    def train_mode(self):
        """
        Put model in train mode
        """
        self.test = False
        self.train()

    # Do the same pre-processing here that is done while training for the passed HxWxC np.ndarray (single image)
    #  Note must mirror actions by src.traning.torch_dataset_builder.ImageTrajectoryDataset.preprocess_imgs()
    def preprocess_input(self, obs: np.ndarray) -> torch.Tensor:
        # pt container to hold preprocessed image
        obs_pt = torch.from_numpy(obs.copy())
        # Permute dimensions to bring to torch CxHxW format
        obs_pt = obs_pt.permute((2, 0, 1))
        # Create float32 container for preprcocessed image to be returned
        obs_pt_float = torch.zeros_like(obs_pt, dtype=torch.float32)
        # create torchvision transform object to operate on 3 channels at a time
        preprocess_img = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((self.config.imsize, self.config.imsize)),
                                             transforms.ToTensor()
                                             ])

        # Process image 1 3 tuple of channels at a time
        for idx in range(self.nframes):
            obs_pt_float[3*idx:3*(idx+1)] = preprocess_img(obs_pt[3*idx:3*(idx+1)])
        return obs_pt_float

    @staticmethod
    def disect_model(model_name: str):
        """
        Infer the simple-model trained on, the arch used, and the number of frames this segmentation is
        trained to operate on (1 frame or 2 concatenated frames) from model_name
        :param model_name:
        :return:
        """
        # Split up the complete model name by _
        attrs = model_name.split('_')
        # Name of the dataset folder this model was created from
        folder_name = attrs[1] + '_' + attrs[2] + '_' + attrs[3]
        # Wrapper object around a segmentation dataset: Infers simple model and nframes trained on
        enc_dataset = EncDataset(folder_name)
        return enc_dataset


if __name__ == '__main__':
    encoder = EncoderEnsemble(model_name="model_cartpole_enc_2frame_Mar29_13-15-59", load_model=True)

    encoder.send_model_to_gpu()

    mu, std = encoder.encode_single_obs(np.zeros((64, 64, 6), dtype=np.float32))
