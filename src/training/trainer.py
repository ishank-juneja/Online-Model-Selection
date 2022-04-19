import torch
from torch import optim
from torch.distributions import Normal
from src.learned_models.ensemble import EncoderEnsemble


class Trainer:
    def __init__(self, model_name: str, config, writer=None):
        self.config = config
        self.name = model_name
        self.writer = writer
        # Put model on GPU
        self.model = EncoderEnsemble(model_name).to(device=self.config.device)
        self.lr = self.config.lr_init

        if self.config.optimiser == 'adam':
            self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.config.optimiser == 'sgd':
            self.optimiser = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise Exception('Unsupported optimiser')

    def run(self, train_loader, test_loader):
        # Train model
        for e in range(self.config.epochs):
            # Record learning rate
            self.writer.add_scalar('learning rate for epoch', self.lr, e)

            loss, encoder_loss = self.train_epoch(train_loader)

            self.writer.add_scalar('training loss', loss, e)

            if (e + 1) % 1 == 0:
                print("Epoch: {0} Total loss: {1} Encoder loss: {2}".format(e + 1, loss, encoder_loss))
            # Reduce learning rate
            if (e + 1) % self.config.lr_decay_steps == 0:
                self.lr *= self.config.lr_decay_rate
                for param_group in self.optimiser.param_groups:
                    param_group['lr'] = self.lr

            if (e + 1) % 5 == 0:
                loss, encoder_loss = self.test(test_loader)
                print('------- TEST --------')
                print("Epoch: {0} Total loss: {1} Encoder loss: {2}".format(e + 1, loss, encoder_loss))
                self.writer.add_scalar('test loss', loss, e)

        self.save()

    def do_epoch(self, data_loader, grad_update=True):
        total_loss = 0.0
        total_encoder_loss = 0.0
        num_batches = len(data_loader.dataset) // data_loader.batch_size

        for obs, state, action in data_loader:
            obs = obs.to(device=self.config.device)
            state = state.to(device=self.config.device)
            # actions = action.to(device=self.config.device)

            N, T, _ = state.size()
            true_z = state[:, :, :self.config.observation_dimension].view(1, N, T, -1)
            elbo = 0

            if not self.model.test:
                true_z = true_z.repeat(self.config.num_ensembles, 1, 1, 1)

            # Encode observations
            z_mu, z_std = self.model.encode(obs)

            q_dist = Normal(z_mu, z_std)

            encoder_loss = - (q_dist.log_prob(true_z.view(-1, T, self.config.observation_dimension)))
            encoder_loss = (encoder_loss.mean(dim=0) * N).sum()
            total_encoder_loss += encoder_loss

            elbo -= encoder_loss
            loss = -elbo / N

            if grad_update:
                self.optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
                self.optimiser.step()
            total_loss += loss.item()

        total_loss /= num_batches
        total_encoder_loss /= (num_batches * N)

        return total_loss, total_encoder_loss

    def train_epoch(self, train_loader):
        self.model.train_mode()
        return self.do_epoch(train_loader)

    def test(self, test_loader):
        self.model.eval_mode()
        with torch.no_grad():
            return self.do_epoch(test_loader, grad_update=False)

    def save(self):
        self.model.save_model(self.name)
