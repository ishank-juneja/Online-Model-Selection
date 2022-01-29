import torch
from torch import nn
from torch.distributions import Normal, Bernoulli


TINY = 1e-10


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


def reconstruction_loss(true_img, pred_img, recurrent=False):
    ''' must have list of immages'''
    multistep = pred_img.size()[1]
    N = true_img.size()[0]
    loss_fn = nn.MSELoss(reduction='mean')

    if recurrent:

        t_img = true_img[:-multistep+1]

        for i in range(1, multistep):
            if multistep == i+1:
                t_img = torch.cat((t_img, true_img[i:]), 1)
            else:
                t_img = torch.cat((t_img, true_img[i:-multistep+i+1]), 1)

        loss = loss_fn(t_img, pred_img)
    else:
        loss = loss_fn(pred_img, true_img)

    return loss


def mmd_loss(true_img, pred_img, z_fake, z_true, recurrent=False):
    mmd_loss = compute_mmd(z_fake, z_true)
    recon_loss = reconstruction_loss(true_img, pred_img, recurrent)
    return mmd_loss + recon_loss, mmd_loss, recon_loss


def KL_loss(true_img, pred_img, mu, log_var, beta=1.0, recurrent=False):
    recon_loss = reconstruction_loss(true_img, pred_img, recurrent)
    KLD = beta * -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return KLD + recon_loss, KLD, recon_loss


def KL_only(mu, log_var, beta=1.0):
    KL = beta * -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return KL


def discriminator_loss(fake_class, true_class):
    adv_loss = - torch.mean(torch.log(true_class + TINY) + torch.log(1.0 - fake_class + TINY))
    return adv_loss


def generator_loss(fake_class):
    gen_loss = -torch.mean(torch.log(fake_class + TINY))
    return gen_loss


def latent_state_deterministic_loss(states_overshoot):
    loss = 0.0
    loss_fn = nn.MSELoss(reduction='mean')

    for i in range(len(states_overshoot) - 1):
        prior_states = states_overshoot[i][1:]
        posterior_states = states_overshoot[i + 1][:-1]
        for j in range(len(prior_states)):
            loss += loss_fn(prior_states[j], posterior_states[j])

    N = (len(states_overshoot) - 1) * len(prior_states)
    return loss / N


def log_p_img(img, reconstructed_img, img_dist='bernoulli', img_cov=0.1, sum=True):
    if img_dist == 'bernoulli':
        dist = Bernoulli(reconstructed_img)
    elif img_dist == 'gaussian':
        dist = Normal(reconstructed_img, img_cov)
    else:
        raise ValueError("Invalid image distribution, must be Gaussian or Bernoulli")
    if sum:

        return dist.log_prob(img).sum()
    return dist.log_prob(img)
