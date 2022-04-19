import torch
from torch import nn
from torch.nn import functional as F


class TransitionModel(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(TransitionModel, self).__init__()
        hidden = 64
        self.act_fn = F.relu
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, state_dim * 2)

    def forward(self, state, action):
        hidden = torch.cat((state, action), 1)
        hidden = self.act_fn(self.fc1(hidden))
        hidden = self.act_fn(self.fc2(hidden))
        z_mu, log_var = torch.chunk(self.fc3(hidden), 2, dim=1)
        z_std = log_var.exp() / 2.0

        return z_mu, z_std


class DynamicsParameterNetwork(nn.Module):

    def __init__(self, obs_dim, K, hidden_size=50):
        super(DynamicsParameterNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(obs_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, K)
        self.K = K

    def forward(self, z, hidden):
        hx, cx = self.rnn(z, hidden)
        alpha = self.fc(hx)
        alpha = F.softmax(alpha, dim=1)
        return alpha, (hx, cx)


class EmissionModel(nn.Module):

    def __init__(self, state_dim, observation_dim):
        super(EmissionModel, self).__init__()
        hidden = 32
        self.act_fn = torch.tanh
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, observation_dim)

    def forward(self, x):
        z = self.act_fn(self.fc1(x))
        z = self.act_fn(self.fc2(z))
        z = self.fc3(z)
        return z


class LinearEmission(nn.Module):

    def __init__(self, state_dim, observation_dim, learnable=False, device='cuda:0'):
        super(LinearEmission, self).__init__()

        #self.C = torch.tensor(([1.0, 0.0, 0.0, 0.0, 0.0],
        #                         [0.0, 1.0, 0.0, 0.0, 0.0],
        #                         [0.0, 0.0, 1.0, 0.0, 0.0]), device=device)


        self.C = torch.cat((torch.eye(observation_dim),
                              torch.zeros(observation_dim, state_dim - observation_dim)), dim=1).to(device=device)

        if learnable:
            self.C = torch.nn.Parameter(0.05 * torch.randn(3, 5, device=device))

    def get_C(self):
        return self.C

    def forward(self, x):
        return F.linear(x, self.C)


class TransitionDeterministicModel(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(TransitionDeterministicModel, self).__init__()
        self.recurrent = False

        hidden = 32
        self.act_fn = torch.nn.functional.relu

        if not self.recurrent:
            self.fc1 = nn.Linear(state_dim + action_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, state_dim)
        else:
            self.rnn = nn.GRUCell(action_dim, state_dim)
            self.fc3 = nn.Linear(state_dim, state_dim)
        self.update_delta = 1.0

    def forward(self, state, action):

        if self.recurrent:
            hidden = self.rnn(action, state)
        else:
            hidden = torch.cat((state, action), 1)
            hidden = self.act_fn(self.fc1(hidden))
            hidden = self.act_fn(self.fc2(hidden))

        d_state = self.fc3(hidden)

        return d_state

    def reset_params(self):
        pass


class TransitionModelWithPars(nn.Module):

    def __init__(self, state_dim, pars_dim, action_dim):
        super(TransitionModelWithPars, self).__init__()
        hidden = 32
        self.act_fn = torch.tanh
        self.fc1 = nn.Linear(state_dim + action_dim + pars_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, state_dim)

    def forward(self, state, action, pars):

        hidden = torch.cat((state, action, pars), 1)
        hidden = self.act_fn(self.fc1(hidden))
        hidden = self.act_fn(self.fc2(hidden))
        return self.fc3(hidden)


