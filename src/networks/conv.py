import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


class KVAEEncoder(nn.Module):

    def __init__(self, observation_dim, imsize=32, mc_dropout=True, dropout_prob=0.1, depth=False, grey=True):
        super(KVAEEncoder, self).__init__()

        if imsize == 32:
            self.hidden_size = 512
            hidden_2_size = 512
        elif imsize == 64:
            self.hidden_size = 2048
            hidden_2_size = 2048
        elif imsize == 128:
            self.hidden_size = 4 * 2048
            hidden_2_size = 512

        self.n_input_channels = 3
        if grey:
            self.n_input_channels = 1

        if depth:
            self.depth_network = DepthNetwork(imsize, dropout_prob)
            self.fc_depth = nn.Linear(1024, 512)

        self.depth = depth
        self.conv1 = nn.Conv2d(self.n_input_channels, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.act_fn = F.relu
        self.fc1 = nn.Linear(self.hidden_size, hidden_2_size)
        self.fc_output = nn.Linear(hidden_2_size, observation_dim)

        p = dropout_prob
        self.drop = nn.Dropout(p=p)
        self.drop2 = nn.Dropout2d(p=p)

    def forward(self, x):
        image = x[:, :self.n_input_channels]

        hidden = self.act_fn(self.conv1(image))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = hidden.view(-1, self.hidden_size)
        hidden = self.drop(self.act_fn(self.fc1(hidden)))

        if self.depth:
            depth = x[:, -1].unsqueeze(1)
            depth_hidden = self.depth_network(depth)
            hidden = torch.cat((hidden, depth_hidden), dim=1)
            hidden = self.drop(self.act_fn(self.fc_depth(hidden)))

        hidden = self.fc_output(hidden)

        return hidden


class DepthNetwork(nn.Module):
    def __init__(self, imsize=32, dropout_prob=0.0):
        super(DepthNetwork, self).__init__()
        if imsize == 32:
            self.hidden_size = 512
        elif imsize == 64:
            self.hidden_size = 2048
        elif imsize == 128:
            self.hidden_size = 4 * 2048

        self.conv1 = nn.Conv2d(1, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(self.hidden_size, 512)
        self.act_fn = F.relu
        self.drop = nn.Dropout(p=dropout_prob)

    def forward(self, depth_img):
        hidden = self.act_fn(self.conv1(depth_img))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = hidden.view(-1, self.hidden_size)
        return self.drop(self.act_fn(self.fc1(hidden)))


class KVAEDecoder(nn.Module):

    def __init__(self, observation_dim, upscale_factor=2, imsize=32, grey=True, depth=False):
        super(KVAEDecoder, self).__init__()
        if imsize == 32:
            self.hidden_size = 512
        elif imsize == 64:
            self.hidden_size = 2048
        elif imsize == 128:
            self.hidden_size = 4 * 2048

        n_output_channels = 3
        if grey:
            n_output_channels = 1
        if depth:
            n_output_channels += 1

        self.filt = int(sqrt(self.hidden_size / 32))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.fc1 = nn.Linear(observation_dim, self.hidden_size)
        self.conv1 = nn.Conv2d(32, 32 * 4, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32 * 4, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32 * 4, 3, padding=1)
        self.conv4 = nn.Conv2d(32, n_output_channels, 1)
        self.act_fn = F.relu

    def forward(self, z):
        hidden = self.fc1(z).view(-1, 32, self.filt, self.filt)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.pixel_shuffle(hidden)
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.pixel_shuffle(hidden)
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.pixel_shuffle(hidden)
        hidden = torch.sigmoid(self.conv4(hidden))
        return hidden


