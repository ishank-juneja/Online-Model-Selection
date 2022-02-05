import torch
from torch import nn
import json
import warnings
from torchvision import transforms


class BaseModel(nn.Module):

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.decoder = None
        self.encoder = None
        self.name = None
        self.test = False

    def preprocess_observation(self, observation):
        img = torch.from_numpy(observation.copy()).permute(2, 1, 0)

        if self.config.data_config.grey:
            preprocess_img = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Grayscale(num_output_channels=1),
                                                 transforms.Resize((self.config.imsize, self.config.imsize)),
                                                 transforms.ToTensor()
                                                 ])
        else:
            preprocess_img = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Resize((self.config.imsize, self.config.imsize)),
                                                 transforms.ToTensor()
                                                 ])

        if self.config.data_config.depth:
            depth = img[3].float()
            img = img[:3].float()
            preprocessed = preprocess_img(img) + 0.01 * torch.randn_like(img)
            return torch.cat((preprocessed, depth.unsqueeze(0)), dim=0)

        return preprocess_img(img)

    def encode(self, observations):
        if self.config.use_ensembles:
            z_mu, z_std = self.encode_ensemble(observations)
        else:
            N, T, C, W, H = observations.shape
            o = observations.view(N * T, C, W, H)

            if self.config.mc_dropout and self.test:
                o = o.view(1, N * T, C, W, H).repeat(self.config.num_ensembles, 1, 1, 1, 1).view(-1, C, W, H)

            z_mu, z_logvar = torch.chunk(self.encoder(o), 2, dim=1)

            z_var = self.process_zlogvar(z_logvar)

            if self.config.mc_dropout and self.test:
                z_mu, z_var = self.combine_estimates(z_mu, z_var)

            z_mu = z_mu.reshape(N, T, -1)
            z_std = z_var.sqrt().reshape(N, T, -1)


        # need to swap order to be sine, cosine, x
        # TODO can just do this in data -- this is only for cartpole environment -- was trained without thsi
        #z_mu = torch.cat((z_mu[:, :, 1:], z_mu[:, :, 0].unsqueeze(2)), 2)
        #z_std = torch.cat((z_std[:, :, 1:], z_std[:, :, 0].unsqueeze(2)), 2)

        return z_mu, z_std

    def encode_ensemble(self, observations):
        '''

        :param observations: N x T x C x W x H image tensors
        :return: encode images to z_mu z_std using N_ensembles number of networks
                z_mu, z_std
                if test mode combines ensemble estimates into estimate size (N x T x nz)
                if train mode then returns (N_ensemble x N x T x nz) to train each ensemble independently
        '''
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

    def decode(self, z):
        N, T, z_dim = z.shape
        z_flat = z.view(N * T, z_dim)
        z_flat = torch.cat((z_flat[:, 2].unsqueeze(1), z_flat[:, :2]), dim=1)
        return self.decoder(z_flat).reshape(N, T, -1, self.config.imsize, self.config.imsize)

    def encode_single_observation(self, observation):
        '''

        :param observation: observation directly from environment -- is an unprocessed numpy array
        :return: z_mu, z_std for configuration estimate
        '''
        o = self.preprocess_observation(observation)
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
        if self.config.cap_z_var:
            return self.config.emission_noise * torch.sigmoid(z_logvar)
        else:
            return z_logvar.exp()

    def save_model(self, name):
        if name is None:
            warnings.warn('Saving but name is None')
        torch.save(self.state_dict(), 'models/CNN/{}'.format(name))

    def load_model(self, model_name):
        self.load_state_dict(torch.load('models/CNN/{}'.format(model_name), map_location=self.config.device))
        self.name = model_name

    def forward(self):
        raise NotImplementedError

    def eval_mode(self):
        self.test = True
        self.eval()

    def train_mode(self):
        self.test = False
        self.train()
