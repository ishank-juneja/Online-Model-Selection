import torch
from torch.distributions import MultivariateNormal


class UnscentedKalmanFilter:
    def __init__(self, state_dim: int, obs_dim: int, control_dim: int, Q: torch.Tensor, R: torch.Tensor, device: str):
        """
        :param state_dim: Dimensionality of the (output) state to be estimated
        :param obs_dim: Dimensionality of the input observations, which are in turn the outputs coming from the
        perception system
        :param control_dim: Number of controls
        :param Q: Process noise covariance
        :param R: Observation noise covariance
        :param device: cpu/gpu name str to put params on
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

    def update(self, measurement, mu_bar, sigma_bar, measurement_fn, R=None):
        if R is None:
            R = self.R

        # Get sigma points
        n_sigma = self.sigma_point_selector.get_n_sigma()
        sigma_points = self.sigma_point_selector.sigma_points(mu_bar, sigma_bar)

        # Pass sigma points through measurement fn
        sigma_measurements = measurement_fn(sigma_points.view(-1, self.state_dim)).view(-1, n_sigma, self.obs_dim)

        mu_z, cov_z = self.unscented_transform(sigma_measurements)
        Pz = cov_z + R

        # Compute innovation
        y = measurement - mu_z

        # Compute cross covariance
        Pxz = self.cross_covariance(mu_bar, mu_z, sigma_points.view(-1, n_sigma, self.state_dim),
                                    sigma_measurements)

        # Get Kalman Gain
        kalman_gain = Pxz.matmul(Pz.inverse())

        # Get new state
        mu = mu_bar + kalman_gain.matmul(y.view(-1, self.obs_dim, 1)).view(-1, self.state_dim)
        sigma = sigma_bar - kalman_gain.matmul(Pz).matmul(kalman_gain.transpose(1, 2))

        return mu, sigma

    def update_linear(self, mu_y_next, mu_z_cur, sigma_z_cur, measurement_fn):
        """
        :param mu_y_next: Next received observation from perception
        :param mu_z_cur: Current full state estimate
        :param sigma_z_cur: Current uncertainty on estimate (from filter)
        :param measurement_fn: Measurement/observation function in KF framework
        :return:
        """
        N, _ = mu_z_cur.size()
        C = measurement_fn.get_C().unsqueeze(0)

        # Compute Kalman Gain
        S = C.matmul(sigma_z_cur).matmul(C.transpose(1, 2)) + self.R
        S_inv = S.inverse()
        K = sigma_z_cur.matmul(C.transpose(1, 2)).matmul(S_inv)

        # Get innovation
        innov = mu_y_next.unsqueeze(2) - C.matmul(mu_z_cur.unsqueeze(2))

        # Get new mu
        mu = mu_z_cur.unsqueeze(2) + K.matmul(innov)

        # Compute sigma using Joseph's form -- should be better numerically
        #  http://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html
        IK_C = torch.eye(self.state_dim, device=self.device) - K.matmul(C)
        KRK = K.matmul(self.R.matmul(K.transpose(1, 2)))
        sigma = IK_C.matmul(sigma_z_cur.matmul(IK_C.transpose(1, 2))) + KRK

        return mu.squeeze(dim=2), sigma

    def predict(self, control, mu, sigma, dynamics_fn, Q=None):
        if Q is None:
            Q = self.Q

        # get sigma points
        sigma_points = self.sigma_point_selector.sigma_points(mu, sigma)

        # Need to duplicate controls for sigma points
        n_sigma = self.sigma_point_selector.get_n_sigma()
        sigma_controls = control.repeat(1, n_sigma).view(-1, self.control_dim)

        # Apply dynamics
        new_sigma_points = dynamics_fn(sigma_points, sigma_controls).view(-1, n_sigma, self.state_dim)

        # Get predicted next state
        mu_bar, cov = self.unscented_transform(new_sigma_points)
        P_bar = cov + Q

        # Get cross covariance -- this is needed for smoothing
        CC = self.cross_covariance(mu, mu_bar, sigma_points.view(-1, n_sigma, self.state_dim), new_sigma_points)
        return mu_bar, P_bar, CC

    def filter(self, dynamics_fn, measurement_fn, controls, observations, prior_mu, prior_sigma, R=None):

        predictive_mus = []
        predictive_sigmas = []
        filtered_mus = []
        filtered_sigmas = []
        cross_covariances = []

        mu_bar = prior_mu
        sigma_bar = prior_sigma
        # Check if linear or not
        update_fn = self.update_linear if callable(getattr(measurement_fn, "get_C", None)) else self.update

        for i in range(controls.size()[1]):
            mu, sigma = update_fn(observations[:, i].view(-1, self.obs_dim),
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

    def smooth(self, forward_states):
        """
        RTS Smoothing for UKF: https://users.aalto.fi/~ssarkka/pub/uks-preprint.pdf Simo Sarkka 2008
        :param forward_states: 
        :return: 
        """
        filtered_mu, filtered_sigma, pred_mu, pred_sigma, cross_covariances = forward_states
        T = filtered_mu.size()[1]

        mu = filtered_mu[:, -1]
        sigma = filtered_sigma[:, -1]

        smoothed_mu = [mu]
        smoothed_sigma = [sigma]
        for t in range(2, T + 1):
            fmu = filtered_mu[:, -t]
            fsig = filtered_sigma[:, -t]
            pmu = pred_mu[:, -t]
            psig = pred_sigma[:, -t]
            cross_covariance = cross_covariances[:, -t]

            # Get smoother gain
            J = cross_covariance.matmul(psig.inverse())

            # Get smoothed state
            mu = fmu + J.matmul((mu - pmu).view(-1, self.state_dim, 1)).view(-1, self.state_dim)
            sigma = fsig + J.matmul(sigma - psig).matmul(J.transpose(1, 2))

            smoothed_mu.append(mu)
            smoothed_sigma.append(sigma)

        return torch.stack(smoothed_mu[::-1], 1), torch.stack(smoothed_sigma[::-1], 1)

    def unscented_transform(self, sigma_points):

        # Calculate mu and cov based on sigma points and weghts
        Wm, Wc = self.sigma_point_selector.get_weights()
        # First the mean
        mu = (Wm * sigma_points).sum(dim=1)
        cov = self.get_cov(mu, mu, sigma_points, sigma_points, Wc)
        return mu, cov

    def cross_covariance(self, mu_x, mu_z, sigma_x, sigma_z):

        # Calculate mu and cov based on sigma points and weghts
        Wm, Wc = self.sigma_point_selector.get_weights()

        # Now the covariance
        cov = self.get_cov(mu_x, mu_z, sigma_x, sigma_z, Wc)

        return cov

    def get_cov(self, mu_x, mu_z, sigma_x, sigma_z, Wc):
        batch_size = mu_x.size()[0]
        n_sigma = sigma_x.size()[1]
        nz = mu_z.size()[1]
        nx = mu_x.size()[1]
        tmp_x = torch.transpose(torch.transpose(sigma_x, 0, 1) - mu_x, 0, 1).view(-1, n_sigma, nx, 1)
        tmp_z = torch.transpose(torch.transpose(sigma_z, 0, 1) - mu_z, 0, 1).view(-1, n_sigma, nz, 1)
        # Need to reshape and duplicate weights so I can do elementwise multiplication
        Wc_tmp = Wc.view(1, n_sigma, 1, 1).repeat(batch_size, 1, nx, nz)

        return (Wc_tmp * torch.matmul(tmp_x, torch.transpose(tmp_z, 2, 3))).sum(dim=1)

    def get_log_likelihoods(self, smoothed_states, observations, controls, dynamics_fn, measurement_fn, prior_cov):
        smoothed_mu = smoothed_states[0]
        smoothed_sigma = smoothed_states[1]

        batch_size = smoothed_mu.size()[0]
        traj_len = smoothed_mu.size()[1]

        zero_state = torch.zeros(self.state_dim, device=self.device)
        zero_obs = torch.zeros(self.obs_dim, device=self.device)
        state_identity = prior_cov[0]

        mvn_smooth = MultivariateNormal(smoothed_mu.view(batch_size, traj_len, self.state_dim), smoothed_sigma)

        # Sample trajectory
        x_smooth = mvn_smooth.rsample()

        # Get entropy -- I think this is p(x|z, u) ?
        entropy = -mvn_smooth.log_prob(x_smooth)

        # Transition log_probs
        mu_transition = dynamics_fn(x_smooth[:, :-1].contiguous().view(-1, self.state_dim),
                                    controls[:, :-1].contiguous().view(-1, self.control_dim)
                                    ).view(-1, traj_len - 1, self.state_dim)

        # Center around zero mean
        transition_centered = x_smooth[:, 1:] - mu_transition
        mvn_transition = MultivariateNormal(zero_state, self.Q)
        log_probs_transition = mvn_transition.log_prob(transition_centered.view(-1, self.state_dim))

        # Do emission log probs
        mu_emission = measurement_fn(x_smooth.contiguous().view(-1, self.state_dim)).view(-1,
                                                                                          traj_len,
                                                                                          self.obs_dim)

        emission_centered = observations.view(batch_size, -1, self.obs_dim) - mu_emission
        mvn_emission = MultivariateNormal(zero_obs, self.R)

        log_probs_emission = mvn_emission.log_prob(emission_centered.view(-1, self.obs_dim))

        # Get log probs on prior
        mvn_prior = MultivariateNormal(zero_state, state_identity)
        log_prob_prior = mvn_prior.log_prob(x_smooth[:, 0, :].view(-1, self.state_dim))

        log_probs = torch.stack([log_probs_transition.sum(),
                                 log_probs_emission.sum(),
                                 log_prob_prior.sum(),
                                 entropy.sum()], 0)

        # scale log_probs by total data
        return log_probs / (batch_size)


class MerweSigmaPoints:

    def __init__(self, n, alpha=0.1, beta=2.0, kappa=None, device='cuda:0'):
        self.alpha = alpha
        self.beta = beta
        if kappa is None:
            kappa = 3 - n
        # avoid 0
        if n + kappa == 0:
            kappa = 0
        self.kappa = kappa
        self.l = alpha ** 2 * (n + kappa) - n
        self.n = n
        self.device = device
        self.Wm, self.Wc = self.compute_weights()

    def get_n_sigma(self):
        return 2 * self.n + 1

    def get_weights(self):
        return self.Wm, self.Wc

    def sigma_points(self, mu, sigma):

        # This is a hack that probably slows stuff down but this is a stupid bug
        U = torch.cholesky((self.l + self.n) * sigma.cpu()).to(self.device)
        # U = torch.cholesky((self.l + self.n) * sigma)
        sigmas = [mu]

        for i in range(self.n):
            x1 = mu - U[:, :, i]
            x2 = mu + U[:, :, i]
            sigmas.extend([x1, x2])
        return torch.stack(sigmas, 1).view(-1, self.n)

    def compute_weights(self):
        Wm = [self.l / (self.n + self.l)]
        Wc = [Wm[0] + 1 - self.alpha ** 2 + self.beta]

        Wm.extend([1.0 / (2 * (self.n + self.l))] * 2 * self.n)
        Wc.extend([1.0 / (2 * (self.n + self.l))] * 2 * self.n)
        return torch.tensor(Wm, device=self.device).view(-1, 1), torch.tensor(Wc, device=self.device).view(-1, 1)
