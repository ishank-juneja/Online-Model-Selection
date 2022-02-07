import torch
import numpy as np
from torch import optim
from torch.distributions import Normal
from src.models.UKVAE import UnscentedKalmanVariationalAutoencoder
from src.models.KVAE import KalmanVariationalAutoencoder
from src.losses import log_p_img
from src.plotting.tom_plotter import frames_to_video, plot_latent_observations, plot_two_frames, plot_grid_images


class Trainer:
    def __init__(self, model_name, config, writer=None):
        self.config = config
        self.name = model_name
        self.writer = writer

        self.recon_weight = self.config.recon_weight_init
        self.dynamics_loss_weight = self.config.dynamics_loss_weight_init

        if self.config.model_type == 'ukvae':
            self.model = UnscentedKalmanVariationalAutoencoder(config).to(device=self.config.device)
        elif self.config.model_type == 'kvae':
            self.model = KalmanVariationalAutoencoder(config).to(device=self.config.device)
        else:
            raise Exception('Invalid model type')

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

            loss, encoder_loss, dynamics_loss = self.train_epoch(train_loader)

            self.writer.add_scalar('training loss', loss, e)

            if (e + 1) % 1 == 0:
                print("Epoch: {} Total loss: {} Encoder loss: {}  Dynamics loss: {}".format(e + 1, loss,
                                                                                              encoder_loss,
                                                                                              dynamics_loss))

            # Decay learning rate
            if (e + 1) % self.config.recon_weight_decay_steps == 0:
                self.recon_weight = max(self.recon_weight * self.config.recon_weight_decay_rate,
                                        self.config.recon_weight_min)

            # Reduce learning rate
            if (e + 1) % self.config.lr_decay_steps == 0:
                self.lr *= self.config.lr_decay_rate
                for param_group in self.optimiser.param_groups:
                    param_group['lr'] = self.lr

            # Annealing of dynamics term
            if (e + 1) % self.config.dynamics_loss_anneal_steps == 0:
                self.dynamics_loss_weight = min(1.0, self.dynamics_loss_weight * self.config.dynamics_loss_anneal_rate)
                print(self.dynamics_loss_weight)

            if (e + 1) % 5 == 0:
                loss, encoder_loss, dynamics_loss = self.test(test_loader)
                print('------- TEST --------')
                print("Episode: {} Total loss: {} Encoder loss: {}  Dynamics loss: {}".format(e + 1, loss,
                                                                                      encoder_loss,
                                                                                      dynamics_loss))
                self.writer.add_scalar('test loss', loss, e)

        self.save()

    def do_epoch(self, data_loader, grad_update=True):

        total_loss = 0.0
        total_dynamics_loss = 0.0
        total_encoder_loss = 0.0
        num_batches = len(data_loader.dataset) // data_loader.batch_size

        distances = []
        z_stds = []
        #TODO streamline and remove reference to ELBO
        # Should also code up KVAE in pytorch so I can put it on github
        for obs, state, action in data_loader:
            N, T, _ = state.shape
            obs = obs.to(device=self.config.device)
            state = state.to(device=self.config.device)
            actions = action.to(device=self.config.device)

            N, T, _ = action.size()
            true_z = state[:, :, :self.config.observation_dimension].view(1, N, T, -1)
            elbo = 0

            if self.config.use_ensembles and not self.model.test:
                true_z = true_z.repeat(self.config.num_ensembles, 1, 1, 1)

            if self.config.train_encoder:
                # Encode observations
                z_mu, z_std = self.model.encode(obs)

                # Sample and decode observations
                q_dist = Normal(z_mu, z_std)
                z = q_dist.rsample()

                entropy = q_dist.entropy().sum(dim=2).flatten()
                # No effect, entropy weight = 0.0
                elbo += self.config.entropy_weight * entropy.sum()
                # Only Used
                if self.config.use_true_pos:
                    encoder_loss = - self.config.true_pos_weight * \
                                   (q_dist.log_prob(
                                       true_z.view(-1, T, self.config.observation_dimension)))  # .mean(dim=0)*N).sum()
                    encoder_loss = (encoder_loss.mean(dim=0) * N).sum()
                    total_encoder_loss += encoder_loss

                    #encoder_loss = torch.nn.functional.mse_loss(z_mu, true_z.view(-1, T, 6), reduction='sum')

                    elbo -= encoder_loss

                    #print(z_mu - true_z.view(-1, T, 6))
            # No
            if self.config.train_decoder:
                if self.config.sample_z:
                    reconstructed_obs = self.model.decode(z_mu)
                else:
                    reconstructed_obs = self.model.decode(true_z[0])

                recon_loss = -log_p_img(obs,
                                        reconstructed_obs,
                                        img_dist=self.config.img_distribution,
                                        img_cov=self.config.img_cov,
                                        sum=False)

                recon_loss = recon_loss.sum(dim=4).sum(dim=3).flatten()
                elbo = -self.recon_weight * recon_loss.sum()
            # No
            if self.config.learn_dynamics:
                # Learn dynamics using true simulator data
                next_z = self.model.transition(true_z[0].view(-1, self.config.state_dimension),
                                               actions.view(-1, self.config.action_dimension))
                next_z = next_z.view(N, T, -1)[:, :-1, :]
                dynamics_loss = torch.nn.functional.mse_loss(next_z, true_z[0, :, 1:, :]) * N * T
                total_dynamics_loss += dynamics_loss
                elbo -= dynamics_loss

            #distances.append(torch.norm(z_mu.view(-1, 6)[:, :3] - z_mu.view(-1, 6)[:, 3:], dim=1))
            #z_stds.append(z_std.mean(dim=2))
            # If kniown dynamics but no states
            if self.config.use_dynamics_loss:
                elbo += self.dynamics_loss_weight * self.model.get_dynamics_elbo(z, actions)

            loss = -elbo / N


            if grad_update:
                self.optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)

                self.optimiser.step()
            else:
                if self.config.train_decoder:
                    self.plot_entropy_recon_loss(entropy, recon_loss, 'cartpole')
            total_loss += loss.item()

        total_loss /= num_batches
        total_encoder_loss /= (num_batches * N)
        total_dynamics_loss /= (num_batches * N)

        #z_stds = torch.cat(z_stds, dim=0).squeeze(-1)
        #        #istances = torch.cat(distances, dim=0)

        #import matplotlib.pyplot as plt
        #plt.title('Uncertainty vs gripper distance for 0.5m rope')
        #plt.scatter(distances.detach().cpu().numpy(), z_stds.detach().cpu().numpy(), alpha=0.2)
        #plt.xlabel('Distance between grippers (m)')
        #plt.ylabel('State uncertainty')

        #plt.show()
        return total_loss, total_encoder_loss, total_dynamics_loss

    def train_epoch(self, train_loader):
        self.model.train_mode()
        return self.do_epoch(train_loader)

    def test(self, test_loader):
        self.model.eval_mode()
        with torch.no_grad():
            return self.do_epoch(test_loader, grad_update=False)

    def test_no_state(self, test_loader):
        recon_loss = 0.0
        for obs, actions in test_loader:
            obs = obs.to(device=self.config.device)

            # Encode observations
            z_mu, z_std = self.model.encode(obs)
            q_dist = Normal(z_mu, z_std)
            z = q_dist.sample()
            entropy = q_dist.entropy().sum(dim=2).flatten()
            reconstructed_obs = self.model.decode(z)

            reconstruction_loss = -log_p_img(obs,
                                            reconstructed_obs,
                                            img_dist=self.config.img_distribution,
                                            img_cov=self.config.img_cov, sum=False)

            reconstruction_loss = reconstruction_loss.sum(dim=4).sum(dim=3).flatten()
            self.plot_entropy_recon_loss(entropy, reconstruction_loss, 'conkers')

            recon_loss += reconstruction_loss.sum()
        return recon_loss

    def plot_entropy_recon_loss(self, entropy, recon_loss, test_type):
        model_name = self.name.replace('-', '_')
        model_name = model_name.replace('.', '_')
        fig_dir = '../figures/experiments/{}'.format(model_name)

        import matplotlib.pyplot as plt
        plt.scatter(entropy.cpu().numpy(), recon_loss.cpu().numpy(), color='b')
        plt.xlabel('Entropy of state estimate')
        plt.ylabel('Reconstruction loss')
        plt.title('Cartpole state estimator on {} environment'.format(test_type))
        plt.savefig('{}/uncertainty_recon_loss_{}.png'.format(fig_dir, test_type))

    def viz_rollout(self, data_loader):

        for observations, states, actions in data_loader:
            # generate trajectory data
            z_encoded_mu, \
            z_encoded_sigma, \
            recon_imgs, \
            z_rollout_mu, \
            z_rollout_sigma, \
            gen_imgs, \
            true_imgs = self.generate_trajectory(observations, actions)

            N, T, C, W, H = gen_imgs.size()

            model_name = self.name.replace('-', '_')
            model_name = model_name.replace('.', '_')
            fig_dir = '../figures/experiments/{}'.format(model_name)

            plot_two_frames(recon_imgs.view(N, T, C, W, H).cpu().numpy(),
                            gen_imgs.view(N, T, C, W, H).cpu().numpy(),
                            '{}/recon_and_gen.mp4'.format(fig_dir), grey=self.config.grey)

            plot_two_frames(true_imgs.view(N, T, C, W, H).cpu().numpy(),
                            recon_imgs.view(N, T, C, W, H).cpu().numpy(),
                            '{}/true_and_recon.mp4'.format(fig_dir), grey=self.config.grey)

            plot_two_frames(true_imgs.view(N, T, C, W, H).cpu().numpy(),
                            gen_imgs.view(N, T, C, W, H).cpu().numpy(),
                            '{}/true_and_gen.mp4'.format(fig_dir), grey=self.config.grey)

            frames_to_video(true_imgs.view(N, T, C, W, H).cpu().numpy(),
                            '{}/true.mp4'.format(fig_dir), grey=self.config.grey)

            frames_to_video(recon_imgs.view(N, T, C, W, H).cpu().numpy(),
                            '{}/reconstructed.mp4'.format(fig_dir), grey=self.config.grey)

            frames_to_video(gen_imgs.view(N, T, C, W, H).cpu().numpy(),
                            '{}/generated.mp4'.format(fig_dir), grey=self.config.grey)

            plot_latent_observations(z_encoded_mu.cpu().numpy(),
                                     z_encoded_sigma.cpu().numpy(),
                                     z_rollout_mu.cpu().numpy(),
                                     z_rollout_sigma.cpu().numpy(),
                                     '{}/latent_observations.png'.format(fig_dir))

            return

    def generate_trajectory(self, observations, actions, init_obs=4, N=16):

        with torch.no_grad():

            # Get random set of trajectories
            n = np.random.permutation(observations.size()[0])[:N]
            o = observations[n].to(device=self.config.device)
            a = actions[n].to(device=self.config.device)

            # Encode
            z_mu, z_std = self.model.encode(o)
            z = Normal(z_mu, z_std).sample()

            z_encoded = z
            z_encoded_mu = z_mu
            z_encoded_std = z_std

            z_rollout, z_rollout_mu, z_rollout_std = self.model.rollout_actions(z[:, :init_obs], a)

            reconstructed_imgs = self.model.decode(z_encoded)
            generated_imgs = self.model.decode(z_rollout)

            return z_encoded_mu, z_encoded_std, reconstructed_imgs, \
                   z_rollout_mu, z_rollout_std, generated_imgs, observations[n]

    def save(self):
        self.model.save_model(self.name)
