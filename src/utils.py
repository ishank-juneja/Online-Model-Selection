import torch


def vec_to_matsum(v, op):
    d = v.size(-1)
    v_tile = v[:, :, None].repeat(1, 1, d)
    if op == "sum":
        return v_tile + v[:, None, :]
    else:
        return v_tile - v[:, None, :]


def angular_transform(state_mu, state_var, dim_theta):
    """
        Takes gaussian belief over theta and approximates
        Gasussian belief over cosine and sine theta
        :params state_mu: N x d mean vector. Assumes angles are first elements in state
        :params state_var: N x d x d, agan assumes First elements are angles
        :params dim_theta: Number of angles, thus theta_mu = state_mu[:, :dim_theta]

        :returns new_state_mu: mean vector N x (d + dim_theta),
                               Elements are now [sin1, sin2, ..., cos1, cos2, ..., state_non_angles]
        :returns new_state_var: covariance matrix N x (d+dim_theta) x (d+dim_theta)

        reference: https://github.com/steindoringi/MetaLearningGP/blob/master/func_utils.py
    """
    device = state_mu.device
    n, d = state_mu.size()
    d_diff = d - dim_theta
    new_d = d_diff + 2 * dim_theta

    theta_mu = state_mu[:, :dim_theta]
    theta_var = state_var[:, :dim_theta, :dim_theta].view(n, dim_theta, dim_theta)
    theta_var = torch.diagonal(theta_var, dim1=-2, dim2=-1)

    exp_theta_var = torch.exp(-theta_var / 2.0)
    cos_theta_mu = torch.cos(theta_mu)
    sin_theta_mu = torch.sin(theta_mu)

    cos_mu = exp_theta_var * cos_theta_mu
    sin_mu = exp_theta_var * sin_theta_mu

    theta_mu_sum = vec_to_matsum(theta_mu, 'sum')
    theta_mu_sub = vec_to_matsum(theta_mu, 'sub')
    theta_var_sum = vec_to_matsum(theta_var, 'sum')
    theta_var_sum = -theta_var_sum / 2.0

    exp_theta_var_sum = torch.exp(theta_var_sum)
    exp_term_sum = torch.exp(theta_var_sum + theta_var.unsqueeze(2)) - exp_theta_var_sum
    exp_term_sub = torch.exp(theta_var_sum - theta_var.unsqueeze(2)) - exp_theta_var_sum

    U1 = exp_term_sum * torch.sin(theta_mu_sub)
    U2 = exp_term_sub * torch.sin(theta_mu_sum)
    U3 = exp_term_sum * torch.cos(theta_mu_sub)
    U4 = exp_term_sub * torch.cos(theta_mu_sum)

    cos_var = U3 + U4
    sin_var = U3 - U4
    cos_sin_cov = U1 + U2
    sin_cos_cov = cos_sin_cov.permute(0, 2, 1)

    new_theta_mu = torch.cat((sin_mu, cos_mu), dim=1)
    new_sin_var = torch.cat((sin_var, sin_cos_cov), dim=2)
    new_cos_var = torch.cat((cos_sin_cov, cos_var), dim=2)
    new_theta_var = torch.cat((new_sin_var, new_cos_var), dim=1) / 2.0

    cos_mu_diag = torch.diag_embed(cos_mu)
    sin_mu_diag = -torch.diag_embed(sin_mu)

    C = torch.cat((cos_mu_diag, sin_mu_diag), 2)
    C = torch.cat((C, torch.zeros(n, d_diff, 2 * dim_theta, device=device)), 1)

    inp_out_cov = state_var @ C
    new_old_cov = inp_out_cov[:, dim_theta:]
    old_var = state_var[:, dim_theta:, dim_theta:] / 2.

    lower = torch.cat((new_old_cov, old_var), 2)
    right = torch.cat((new_old_cov.permute(0, 2, 1), old_var), 1)

    zeros = torch.zeros(n, new_d, 2 * dim_theta, device=device)
    lower = torch.cat((zeros.permute(0, 2, 1), lower), 1)
    right = torch.cat((zeros, right), 2)

    zeros = torch.zeros(n, new_d, new_d, device=device)
    new_theta_var = torch.cat((new_theta_var, zeros[:, :d_diff, :2 * dim_theta]), 1)
    new_theta_var = torch.cat((new_theta_var, zeros[:, :, :d_diff]), 2)

    new_state_mu = torch.cat((new_theta_mu, state_mu[:, dim_theta:]), 1)
    new_state_var = new_theta_var + lower + right

    return new_state_mu, new_state_var


def chunk_trajectory(z_mu, z_std, u, H=5):
    '''
    Takes z_mu, z_std, u all tensors
    gets T x nz, T x nz, T x nu
    Chunks into N x H x nz/nu
    Even split trajectories (discards extra data)
    '''

    T = min(z_mu.size(0), u.size(0))
    N = T // H
    z_mu = z_mu[:N*H].view(N, H, -1)
    z_std = z_std[:N*H].view(N, H, -1)
    u = u[:N*H].view(N, H, -1)

    return z_mu, z_std, u

def bdot(a, b):
    B = a.shape[0]
    S = a.shape[1]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)