import torch
from src.filters.ukf import UnscentedKalmanFilter


class GPUnscentedKalmanFilter(UnscentedKalmanFilter):

    def __init__(self, state_dim, obs_dim, control_dim, param_dim, Q, R, device):
        super(GPUnscentedKalmanFilter, self).__init__(state_dim, obs_dim, control_dim, Q, R, device)
        self.param_dim = param_dim

    def predict(self, control, mu, sigma, gp_dynamics_fn, params=None):

        # get sigma points
        sigma_points = self.sigma_point_selector.sigma_points(mu, sigma)

        # Need to duplicate controls for sigma points
        n_sigma = self.sigma_point_selector.get_n_sigma()
        sigma_controls = control.repeat(1, n_sigma).view(-1, self.control_dim)

        if params is not None:
            sigma_params = params.view(-1, 1, self.param_dim).repeat(1, n_sigma, 1).view(-1, self.param_dim)
        else:
            sigma_params = None

        # Apply GP dynamics -
        # # sigma points are means of sigma point predictions,
        # # process noise is the variance of the mean prediction
        sigma_model_output = gp_dynamics_fn(sigma_points, sigma_controls, sigma_params)
        new_sigma_points = sigma_model_output.mean.view(-1, n_sigma, self.state_dim)

        # Qt is var of mean -- gets state dependent uncertainty
        Qt = torch.diag_embed(sigma_model_output.variance.view(-1, n_sigma, self.state_dim)[:, 0])
        # Get predicted next state
        mu_bar, cov = self.unscented_transform(new_sigma_points)
        P_bar = cov + Qt

        # Get cross covariance -- this is needed for smoothing
        CC = self.cross_covariance(mu, mu_bar, sigma_points.view(-1, n_sigma, self.state_dim), new_sigma_points)
        return mu_bar, P_bar, CC
