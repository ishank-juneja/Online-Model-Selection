
import torch
from torch import nn
import torch.nn.functional as funcs


class Encoder(nn.Module):
    """
    CNN model to extract state and aleatoric uncertainty over state from images
    """
    def __init__(self, label_dim: int, img_channels: int = 3):
        super(Encoder, self).__init__()

        # Side of square images expected
        self.imsize = 64

        # Number of channels in training/test images
        self.img_channels = img_channels

        # Build Arch
        #  Number of units in hidden layer = Cout x Hout x Wout of last conv layer
        #  In our case, this is 32 x 8 x 8 = 2048
        self.hidden_size = 2048

        # Convolutional Layers
        # nn.Conv2d shapes: input (N, Cin, Hin, Win) output (N, Cout, Hout, Wout)
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        #  Can make encoder fatter using arch:
        #  https://github.com/ishank-juneja/simple-model-perception/blob/b9e7c71bdb9e2f98f03bb13246356278860757fb/src/networks/encoder.py
        self.conv1 = nn.Conv2d(in_channels=self.img_channels, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        # Fully connected output later
        # Here observation dimension is number of outputs of NN, 2 x num of observables states
        # if outputting aleatoric uncertainty
        self.fc_out = nn.Linear(self.hidden_size, 2 * label_dim)

        # Non-linearity
        self.act_fn = funcs.relu

    # Encode state/uncertainty for a batch of images (first dimension)
    def forward(self, x):
        """
        :param x: A batch of N images, tensor of shape N x C x H x W
        :return: Encoded State and Aleatoric Uncertainty
        """
        hidden1 = self.act_fn(self.conv1(x))
        hidden2 = self.act_fn(self.conv2(hidden1))
        hidden3 = self.act_fn(self.conv3(hidden2))
        hidden3 = hidden3.view(-1, self.hidden_size)
        hidden4 = self.act_fn(self.fc1(hidden3))
        out = self.fc_out(hidden4)
        return out
