import torch
from torch.distributions import MultivariateNormal


class UnscentedKalmanFilter:
    """
    Standard Sigma-Point Kalman Filter Implementation
    Notation based on: https://www.anuncommonlab.com/articles/how-kalman-filters-work/
    """
    def __init__(self, state_dim: int, obs_dim: int, control_dim: int, Q: torch.Tensor, R: torch.Tensor, device: str):
        """
        param state_dim: Dimensionality of the (output) state to be estimated
        param obs_dim: Dimensionality of the input observations. Here it is the dim of the outputs
        coming from the perception system
        param control_dim: Number of controls
        param Q: Process noise covariance
        param R: Observation noise covariance
        param device: cpu/gpu name str to put params on
        """
        # Prior mean should be of size (batch, state_size)
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.control_dim = control_dim

        # Utils to deterministically compute sigma points as explained in
        #  https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
        self.sigma_point_selector = MerweSigmaPoints(self.state_dim, device=device)

        # process noise
        self.Q = Q
        # observation noise
        self.R = R

        self.device = device

    def filter(self, prior_mu, prior_sigma, controls, observations, dynamics_fn, measurement_fn):
        """
        Inference step for the Kalman filter model, a forward pass
        All params/callables here intended for batch inference with shape (B, n)
        :param prior_mu: Batch of priors over state from last iter. hat{x}^{+}_{k-1} (B, state_dim)
        :param prior_sigma: Batch of priors over covariance P^{+}_{k-1} (B, state_dim, state_dim)
        :param controls: Batch of actions u_{k-1} in case of actuated system, shape (B, action_dim)
        :param observations: Batch of cur. obs./measurements z_{k} (B, obs_dim)
        :param dynamics_fn: Either nominal simple model dynamics or something else depending on
         what model is invoking the UKF. Expects shape of states and actions to be (B, state_dim) and (B, action_dim)
        :param measurement_fn: Function that takes propagated state to predicted observation
         (B, state_dim), (B, action_dim)
        :return:
        """
        predictive_mus = []
        predictive_sigmas = []
        filtered_mus = []
        filtered_sigmas = []
        cross_covariances = []

        mu_bar = prior_mu
        sigma_bar = prior_sigma

        for i in range(controls.size()[1]):
            mu, sigma = self.update(observations[:, i].view(-1, self.obs_dim),
                                    mu_bar, sigma_bar, measurement_fn, R)
            # Predictive step
            mu_bar, sigma_bar, CC = self.predict(controls[:, i].view(-1, self.control_dim), mu, sigma, dynamics_fn)

            # Store
            predictive_mus.append(mu_bar)
            predictive_sigmas.append(sigma_bar)
            filtered_mus.append(mu)
            filtered_sigmas.append(sigma)
            cross_covariances.append(CC)

        return torch.stack(filtered_mus, 1), torch.stack(filtered_sigmas, 1), \
               torch.stack(predictive_mus, 1), torch.stack(predictive_sigmas, 1), torch.stack(cross_covariances, 1)

    def update(self, mu_y_next, mu_z_cur, sigma_z_cur, measurement_fn, R=None):
        """
        :param mu_y_next: Next received observation from perception
        :param mu_z_cur: Current full state estimate
        :param sigma_z_cur: Current uncertainty on estimate (from filter)
        :param measurement_fn: Measurement/observation function in KF framework
        :param R: Observation noise at current time step R_k (use if varies with k)
        :return:
        """
        if R is None:
            R = self.R
        N, _ = mu_z_cur.size()
        C = measurement_fn.get_C().unsqueeze(0)

        # Compute Kalman Gain
        S = C.matmul(sigma_z_cur).matmul(C.transpose(1, 2)) + R
        S_inv = S.inverse()
        K = sigma_z_cur.matmul(C.transpose(1, 2)).matmul(S_inv)

        # Get innovation
        innov = mu_y_next.unsqueeze(2) - C.matmul(mu_z_cur.unsqueeze(2))

        # Get new mu
        mu = mu_z_cur.unsqueeze(2) + K.matmul(innov)

        # Compute sigma using Joseph's form -- should be better numerically
        #  http://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html
        IK_C = torch.eye(self.state_dim, device=self.device) - K.matmul(C)
        KRK = K.matmul(R.matmul(K.transpose(1, 2)))
        sigma = IK_C.matmul(sigma_z_cur.matmul(IK_C.transpose(1, 2))) + KRK

        return mu.squeeze(dim=2), sigma

    def predict(self, mu, sigma, control, dynamics_fn, Q=None):
        """
        :param mu: Batch of states at the end of prev. iteration hat{x}^{+}_{k-1} (B, state_dim)
        :param sigma: Batch of covs at the end of previous iteration P^{+}_{k-1} (B, state_dim, state_dim)
        :param control: Actions to take me to next iteration u_{k-1} of shape (B, action_dim)
        :param dynamics_fn: Operates on a batch of current states and actions (B, state_dim) and (B, action_dim)
        :param Q: Process noise at current time step Q_k (use if varies with k, else None)
        :return:
        """
        if Q is None:
            Q = self.Q

        # Get sigma points, shape is (B, 2*n_sigma + 1)
        sigma_points = self.sigma_point_selector.sigma_points(mu, sigma)

        # Need to duplicate controls for sigma points
        n_sigma = self.sigma_point_selector.get_n_sigma()
        # control: (B, action_dim), sigma_controls: (B, n_sigma * action_dim)
        sigma_controls = control.repeat(1, n_sigma).view(-1, self.control_dim)

        # Apply dynamics: Input is batch of sigma points: [hat{x}^{+}_{i, k−1}]_{i=1}^{2n+1}, u_{k-1}]
        #  Output is batch: hat{x}^{-}_{k}
        new_sigma_points = dynamics_fn(sigma_points, sigma_controls).view(-1, n_sigma, self.state_dim)

        # Get predicted next state
        mu_bar, cov = self.unscented_transform(new_sigma_points)
        P_bar = cov + Q

        # Get cross covariance -- this is needed for smoothing
        CC = self.cross_covariance(mu, mu_bar, sigma_points.view(-1, n_sigma, self.state_dim), new_sigma_points)
        return mu_bar, P_bar, CC

    def unscented_transform(self, sigma_points):
        """
        Get propagated mean and covariance using weighted mean
        :param sigma_points: The "particles" taking part in unscented transform
        :return:
        """
        # Calculate mu and cov based on sigma points and weights
        Wm, Wc = self.sigma_point_selector.get_weights()
        # First the mean
        mu = (Wm * sigma_points).sum(dim=1)
        cov = self.get_cov(mu, mu, sigma_points, sigma_points, Wc)
        return mu, cov

    def cross_covariance(self, mu_x, mu_z, sigma_x, sigma_z):

        # Calculate mu and cov based on sigma points and weights
        Wm, Wc = self.sigma_point_selector.get_weights()

        # Now the covariance
        cov = self.get_cov(mu_x, mu_z, sigma_x, sigma_z, Wc)

        return cov

    @staticmethod
    def get_cov(mu_x, mu_z, sigma_x, sigma_z, Wc):
        """
        Get sample covariance from passed points
        :param mu_x: Set of p
        :param mu_z:
        :param sigma_x:
        :param sigma_z:
        :param Wc:
        :return:
        """
        batch_size = mu_x.size()[0]
        n_sigma = sigma_x.size()[1]
        nz = mu_z.size()[1]
        nx = mu_x.size()[1]
        tmp_x = torch.transpose(torch.transpose(sigma_x, 0, 1) - mu_x, 0, 1).view(-1, n_sigma, nx, 1)
        tmp_z = torch.transpose(torch.transpose(sigma_z, 0, 1) - mu_z, 0, 1).view(-1, n_sigma, nz, 1)
        # Need to reshape and duplicate weights so I can do elementwise multiplication
        Wc_tmp = Wc.view(1, n_sigma, 1, 1).repeat(batch_size, 1, nx, nz)

        return (Wc_tmp * torch.matmul(tmp_x, torch.transpose(tmp_z, 2, 3))).sum(dim=1)


class MerweSigmaPoints:
    """
    Class to implement the standard approach for selecting sigma points
    Notation based on: https://www.anuncommonlab.com/articles/how-kalman-filters-work/
    """
    def __init__(self, n, alpha=0.1, beta=2.0, kappa=None, device=None):
        """
        Param details on page 6 of https://www.gatsby.ucl.ac.uk/~byron/nlds/merwe2003a.pdf
        :param n: Number of states in UKF, called L in linked pdf
        :param alpha: Small positive constant that determines the spread of the sigma points around the mean
        :param beta: Incorporate prior knowledge about the distributions. For Gaussian distributions,
            beta=2.0 is optimal
        :param kappa: Additional scaling parameter set to 0 or 3 - n
        :param device: CPU/GPU str
        """
        self.alpha = alpha
        self.beta = beta
        if kappa is None:
            kappa = 3 - n
        # avoid 0
        if n + kappa == 0:
            kappa = 0
        self.kappa = kappa
        # lambda from linked PDF
        self.lamb = alpha ** 2 * (n + kappa) - n
        self.n = n
        self.device = device
        # List/vector of weights for transforming mean-m and cov-c
        self.Wm, self.Wc = self.compute_weights()

    def get_n_sigma(self):
        """
        :return:  Return the number of sigma points
        2 points along the elliptical spread of every state and
        """
        return 2 * self.n + 1

    def get_weights(self):
        """
        :return: Return pre-computed sig-point weights for both mean and covariance
        """
        return self.Wm, self.Wc

    def sigma_points(self, mu, sigma):
        """
        :param mu: Corrected mean hat{x}^{+}_{k-1} from end of previous filter iteration
        :param sigma: Corrected cov hat{P}^{+}_{k-1} ...
        :return:
        """
        # Put sigma on cpu() due to the bug here
        #  https://discuss.pytorch.org/t/cuda-illegal-memory-access-when-using-batched-torch-cholesky/51624/13
        # Cholesky needed to find sqrt of cov matrices
        U = torch.cholesky((self.lamb + self.n) * sigma.cpu()).to(self.device)
        # U = torch.cholesky((self.lamb + self.n) * sigma)

        # Init list of sigma points with just the previous mean
        sigmas = [mu]
        # Locate other axial sigma points
        for i in range(self.n):
            x1 = mu - U[:, :, i]
            x2 = mu + U[:, :, i]
            sigmas.extend([x1, x2])
        return torch.stack(sigmas, 1).view(-1, self.n)

    def compute_weights(self):
        """
        :return: Computed weights for propagating mean and variance
        """
        # First weight in weight list is for central sig-point
        Wm = [self.lamb / (self.n + self.lamb)]
        Wc = [Wm[0] + 1 - self.alpha ** 2 + self.beta]
        # Other 2n weights are for axial sig-points
        Wm.extend([1.0 / (2 * (self.n + self.lamb))] * 2 * self.n)
        Wc.extend([1.0 / (2 * (self.n + self.lamb))] * 2 * self.n)
        return torch.tensor(Wm, device=self.device).view(-1, 1), torch.tensor(Wc, device=self.device).view(-1, 1)
