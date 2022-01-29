''' Square root UKF written in pytorch'''
import torch
from torch.distributions import MultivariateNormal


class SquareRootUnscentedKalmanFilter:

    def __init__(self, state_dim, obs_dim, control_dim, Q, R, device):

        # Prior mean should be of size (batch, state_size)
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.control_dim = control_dim

        self.sigma_point_selector = MerweSigmaPointsSR(self.state_dim, device=device)
        self.Q = Q
        self.R = R
        self.root_Q = Q.cholesky()
        self.root_R = R.cholesky()
        self.device = device

    def update(self, measurement, mu_bar, S_bar, measurement_fn):
        # Calculate new mu and S_bar, where Sigma_bar = S_bar @ S_bar.T
        # Get sigma points
        n_sigma = self.sigma_point_selector.get_n_sigma()
        sigma_points = self.sigma_point_selector.sigma_points(mu_bar, S_bar)
        # Pass sigma points through measurement fn
        sigma_measurements = measurement_fn(sigma_points.view(-1, self.state_dim)).view(-1, n_sigma, self.obs_dim)

        mu_z, S_z = self.unscented_transform(sigma_measurements, self.root_R)

        # Compute innovation
        y = measurement - mu_z

        # Compute cross covariance
        Pxz = self.cross_covariance(mu_bar, mu_z, sigma_points.view(-1, n_sigma, self.state_dim),
                                    sigma_measurements)
        # Get Kalman Gain
        tmp, _ = torch.solve(Pxz.transpose(1, 2), S_z)
        K, _ = torch.solve(tmp, S_z.transpose(1, 2))
        kalman_gain = K.transpose(1, 2)

        # Get new state
        mu = mu_bar + kalman_gain.matmul(y.view(-1, self.obs_dim, 1)).view(-1, self.state_dim)
        S = self.cholupdate(S_bar, kalman_gain.matmul(S_z), torch.tensor([-1.0], device=self.device))

        return mu, S

    def update_linear(self, measurement, mu_bar, S_bar, measurement_fn):

        N, _ = mu_bar.size()
        C = measurement_fn.get_C().unsqueeze(0)
        sigma_bar = S_bar.matmul(S_bar.transpose(1, 2))

        # Compute Kalman Gain
        S = C.matmul(sigma_bar).matmul(C.transpose(1, 2)) + self.R
        S_inv = S.inverse()
        K = sigma_bar.matmul(C.transpose(1, 2)).matmul(S_inv)

        # Get innovation
        innov = measurement.unsqueeze(2) - C.matmul(mu_bar.unsqueeze(2))

        # Get new mu
        mu = mu_bar.unsqueeze(2) + K.matmul(innov)

        # Compute sigma using Joseph's form -- should be better numerically
        IK_C = torch.eye(self.state_dim, device=self.device) - K.matmul(C)
        KRK = K.matmul(self.R.matmul(K.transpose(1, 2)))
        sigma = IK_C.matmul(sigma_bar.matmul(IK_C.transpose(1, 2))) + KRK

        return mu.squeeze(dim=2), sigma.cpu().cholesky().to(device=self.device)

    def predict(self, control, mu, S, dynamics_fn, dynamic_params=None):
        # get sigma points
        sigma_points = self.sigma_point_selector.sigma_points(mu, S)

        # Need to duplicate controls for sigma points
        n_sigma = self.sigma_point_selector.get_n_sigma()
        sigma_controls = control.repeat(1, n_sigma).view(-1, self.control_dim)

        # Apply dynamics
        new_sigma_points = dynamics_fn(sigma_points, sigma_controls).view(-1, n_sigma, self.state_dim)

        # Get predicted next state
        mu_bar, S_bar = self.unscented_transform(new_sigma_points, self.root_Q)

        # Get cross covariance -- this is needed for smoothing
        CC = self.cross_covariance(mu, mu_bar, sigma_points.view(-1, n_sigma, self.state_dim), new_sigma_points)
        return mu_bar, S_bar, CC

    def filter(self, dynamics_fn, measurement_fn, controls, observations, prior_mu, prior_sigma, dynamic_params=None):

        predictive_mus = []
        predictive_sigmas = []
        filtered_mus = []
        filtered_sigmas = []
        cross_covariances = []

        mu_bar = prior_mu

        K = 1 # Steps to update filter
        # Do cholesky on CPU due to CUDA error
        S_bar = prior_sigma.cpu().cholesky().to(device=self.device)

        # Check if linear or not
        update_fn = self.update_linear if callable(getattr(measurement_fn, "get_C", None)) else self.update

        for i in range(controls.size()[1]):
            if i % K == 0:
                # Measurement step
                mu, S = update_fn(observations[:, i].view(-1, self.obs_dim),
                                    mu_bar, S_bar, measurement_fn)
            else:
                mu, S = mu_bar, S_bar

            # Predictive step
            mu_bar, S_bar, CC = self.predict(controls[:, i].view(-1, self.control_dim), mu,
                                             S, dynamics_fn, dynamic_params)
            sigma_bar = S_bar.matmul(S_bar.transpose(1, 2))
            sigma = S.matmul(S.transpose(1, 2))

            # Store
            predictive_mus.append(mu_bar)
            predictive_sigmas.append(sigma_bar)
            filtered_mus.append(mu)
            filtered_sigmas.append(sigma)
            cross_covariances.append(CC)

        return torch.stack(filtered_mus, 1), torch.stack(filtered_sigmas, 1), \
            torch.stack(predictive_mus, 1), torch.stack(predictive_sigmas, 1), torch.stack(cross_covariances, 1)

    def smooth(self, forward_states):
        '''
            RTS Smoothing for UKF: https://users.aalto.fi/~ssarkka/pub/uks-preprint.pdf Simo Sarkka 2008
        '''

        filtered_mu, filtered_sigma, pred_mu, pred_sigma, cross_covariances = forward_states
        T = filtered_mu.size()[1]

        mu = filtered_mu[:, -1]
        sigma = filtered_sigma[:, -1]

        smoothed_mu = [mu]
        smoothed_sigma = [sigma]
        G_cross_covariances = []
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
            sigma = fsig + J.matmul(sigma - psig).matmul(J.transpose(1, 2))# + 1e-5*torch.eye(self.state_dim, device=self.device)

            smoothed_mu.append(mu)
            smoothed_sigma.append(sigma)
            G_cross_covariances.append(J)

        return torch.stack(smoothed_mu[::-1], 1), \
               torch.stack(smoothed_sigma[::-1], 1)#, \
               #torch.stack(G_cross_covariances[::-1], 1)

    def unscented_transform(self, sigma_points, noise_mat):

        # Calculate mu and cov based on sigma points and weghts
        Wm, Wc = self.sigma_point_selector.get_weights()

        # First the mean
        mu = (Wm * sigma_points).sum(dim=1)
        S = self.get_S(mu, sigma_points, noise_mat)

        return mu, S

    def get_S(self, mu, sigma_points, noise_mat):
        # noise matrix either Q or R depending on whether predict or update

        # Get some sizes
        batch_size = mu.size()[0]

        # Find difference of sigma points from mean and multiply by weighting
        sigma_point_difference = torch.transpose(sigma_points.transpose(0, 1) - mu, 0, 1)

        _, Wc = self.sigma_point_selector.get_weights()
        tmp = (Wc[1].sqrt() * sigma_point_difference[:, 1:])

        # Reformat noise matrix into batches and do QR decomposition
        noise = noise_mat.view(-1,
                               noise_mat.size()[0],
                               noise_mat.size()[1]).repeat(batch_size, 1, 1)

        # Do QR decomposition -- this is roughly 1000x faster when done on the CPU
        QR = torch.cat((tmp, noise), 1)
        Q, R = QR.cpu().qr()
        R = R.to(device=self.device)

        # Cholupdate
        S = self.cholupdate(R.transpose(1, 2), sigma_point_difference[:, 0].view(batch_size, -1, 1), Wc[0])
        return S

    def ridge_regression_recondition_S(self, C, k_max=1000):
        '''
            Attempts to recondition matrix to have condition number no more than k_max
        '''
        n = C.size()[1]
        batch = C.size()[0]
        eigvals, _ = C.symeig(eigenvectors=True)
        eig_max = eigvals[:, -1]
        eig_min = eigvals[:, 0]

        delta = (eig_max - eig_min * k_max) / (k_max - 1)
        delta = torch.clamp(delta, 0.0, max(delta.max().item(), 0.0))
        I = torch.eye(n, device=self.device).view(1, n, n).repeat(batch, 1, 1)
        return C + delta.view(-1, 1, 1) * I

    def cholupdate(self, S, X, w0):
        # This isn't really a proper cholupdate as I couldn't figure out a differentable way of implementing
        # TODO: Differentiable rank 1 cholesky update
        sos = X.matmul(X.transpose(1, 2))
        sign = torch.sign(w0)
        w = w0.abs().sqrt()

        #C = S.matmul(S.transpose(1, 2)) + sign * w * sos
        C = S.matmul(S.transpose(1, 2)) + w0 * sos# + 1e-5 * torch.eye(S.size()[1], device=self.device)
        #C = S.matmul(S.transpose(1, 2)) + w * sign * sos
        # Need to do cholesky on the CPU as I get some weird CUDA error
        # This is a workaround

        return C.cpu().cholesky().to(device=self.device)

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

    def get_log_likelihoods(self, smoothed_states, observations, controls, dynamics_fn,
                            measurement_fn, prior_cov, dynamic_params=None, plot=False):

        N = 1
        smoothed_mu = smoothed_states[0]
        smoothed_sigma = smoothed_states[1]

        batch_size = smoothed_mu.size()[0]
        traj_len = smoothed_mu.size()[1]

        zero_state = torch.zeros(self.state_dim, device=self.device)
        zero_obs = torch.zeros(self.obs_dim, device=self.device)
        state_identity = prior_cov[0]

        # having some issue with MultivariateNormal which does cholesky internally on GPU
        # so do cholesky on CPU before passing to MultivariateNormal
        #try:
        smoothed_scale = smoothed_sigma.cpu().cholesky().to(device=self.device)
        #except:
        #    print(smoothed_sigma)

        mvn_smooth_0 = MultivariateNormal(smoothed_mu[:, 0].contiguous().view(-1, self.state_dim),
                                        scale_tril=smoothed_scale[:, 0].contiguous().view(-1, self.state_dim, self.state_dim))
        #print(smoothed_sigma[0, 0])
        mvn_smooth_plus_1 = MultivariateNormal(smoothed_mu[:, 1:].contiguous().view(-1, self.state_dim),
                                        scale_tril=smoothed_scale[:, 1:].contiguous().view(-1, self.state_dim, self.state_dim))

        mvn_smooth_total = MultivariateNormal(smoothed_mu.view(-1, self.state_dim),
                                              scale_tril=smoothed_scale.view(-1,self.state_dim, self.state_dim))

        # Sample trajectory
        #x_smooth = mvn_smooth.rsample(sample_shape=(N,))
        x0 = mvn_smooth_0.rsample(sample_shape=(N,)).view(-1, self.state_dim)
        xt_plus_1 = mvn_smooth_plus_1.rsample(sample_shape=(N,)).view(-1, traj_len-1, self.state_dim)


        # Get entropy -- of smoothing distribution
        #entropy = -mvn_smooth.log_prob(x_smooth)

        # Reshape controls into one long vector
        controls = controls.view(1, batch_size, traj_len, self.control_dim).repeat(N, 1, 1, 1)
        ct = controls[:, :, :-1].contiguous().view(-1, self.control_dim)

        controls_overshoot = controls.view(-1, traj_len, self.control_dim)

        # First time just use x1
        mu_transition = [dynamics_fn(x0, controls_overshoot[:, 0])]
        scale_transition = [self.root_Q.view(1, self.state_dim, self.state_dim).repeat(N*batch_size, 1, 1)]
        overshoot_d = 3
        for i in range(overshoot_d - 1):
            mu_i, scale_i, _ = self.predict(controls_overshoot[:, i + 1],
                                            mu_transition[i], scale_transition[i],
                                            dynamics_fn)
            mu_transition.append(mu_i)
            scale_transition.append(scale_i)

        for i in range(traj_len - 1 - overshoot_d):

            x = dynamics_fn(xt_plus_1[:, i], controls_overshoot[:, i+1])
            scale = scale_transition[0]
            for j in range(1, overshoot_d):

                x, scale, _ = self.predict(controls_overshoot[:, i+j+1],
                                                x,
                                                scale,
                                                dynamics_fn)

            mu_transition.append(x)
            scale_transition.append(scale)


        mu_transition = torch.stack(mu_transition, dim=1)

        scale_transition = torch.stack(scale_transition, dim=1).view(-1, self.state_dim, self.state_dim)
        sigma_transition = scale_transition.matmul(scale_transition.transpose(1, 2))
        sigma_transition =self.Q
        px_plus1 = MultivariateNormal(mu_transition.view(-1, self.state_dim), sigma_transition)
        px0 = MultivariateNormal(zero_state, prior_cov)

        kl_transition = torch.distributions.kl.kl_divergence(mvn_smooth_plus_1, px_plus1).sum()
        kl_x0 = torch.distributions.kl.kl_divergence(mvn_smooth_0, px0).sum()

        x_smooth = mvn_smooth_total.rsample(sample_shape=(N,))

        transition = px_plus1.sample().view(-1, traj_len-1, self.state_dim)
        #print(mu_transition[0, :])
        #print(xt_plus_1[0, :])

        #print(mu_transition[0, :] - xt_plus_1[0, :])
        #plot = False
        if plot:
            n = 0
            #print(sigma_transition[:29])
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(self.state_dim, figsize=(16, 16))
            for i in range(self.state_dim):
                axes[i].plot(mu_transition[n, :, i].cpu().numpy(), label='transition')
                axes[i].plot(smoothed_mu[n, 1:, i].cpu().numpy(), label='sample')
                if i < 3:
                    axes[i].plot(observations[n, 1:, i].cpu().numpy(), label='observations')
                #axes[i].plot(xt[n:n+19, i].cpu().numpy(), label='one behind')
                axes[i].legend()

            plt.legend()
            plt.show()
            #transition_centered = (xt_plus_1 - mu_transition).view(-1, self.state_dim)
            #mvn_transition = MultivariateNormal(torch.zeros_like(transition_centered),
            #                                    sigma_transition)

            #log_probs_transition = mvn_transition.log_prob(transition_centered)

            # Do emission log probs
        mu_emission = measurement_fn(x_smooth.view(-1, self.state_dim)).view(-1, self.obs_dim)
        observations = observations.view(1, batch_size * traj_len, self.obs_dim).repeat(N, 1, 1)

        # Reshape observations into one long vector
        zt = observations.view(-1, self.obs_dim)

        # Centre around zero mean
        emission_centered = zt - mu_emission
        mvn_emission = MultivariateNormal(zero_obs, self.R)

        log_probs_emission = mvn_emission.log_prob(emission_centered.view(-1, self.obs_dim))

        # Get log probs on prior
        #mvn_prior = MultivariateNormal(zero_state, state_identity)
        #log_prob_prior = mvn_prior.log_prob(x0)

        #log_probs = torch.stack([log_probs_transition.sum(),
        #                         log_probs_emission.sum(),
        #                         log_prob_prior.sum(),
        #                         entropy.sum()], 0)

        if plot:
            print(log_probs_emission.sum())
            print(kl_transition.sum())
            print(kl_x0)

        dynamics_elbo = log_probs_emission.sum() - kl_transition - kl_x0
        # scale log_probs by total data
        return dynamics_elbo / (N * batch_size)

    def set_qr(self, q, r):
        ''' takes diagonal vectors of root_q and root_R'''

        # need to add some small minimum amount for stability
        qn = q.abs() + 1e-1
        rn = r.abs() + 1e-1
        self.root_Q = torch.diag(torch.abs(qn))
        self.Q = torch.diag(qn.pow(2))
        self.root_R = torch.diag(torch.abs(rn))
        self.R = torch.diag(rn.pow(2))

    def get_Pz(self, mu_x, sigma_x, measurement_fn):
        '''
        :param mu_x: batched state mean vector (N x state_dim)
        :param sigma_x: batched state covariance (N x state_dim x state_dim)
        :param measurement_fn: Measurement fn z = h(x)
        :return: mu_z and sigma_z parameterising Pz ~ N(mu_z, sigma_z)
        '''

        # Get sigma points
        n_sigma = self.sigma_point_selector.get_n_sigma()
        sigma_points = self.sigma_point_selector.sigma_points(mu_x, sigma_x.cholesky())

        # Pass sigma points through measurement fn
        sigma_measurements = measurement_fn(sigma_points.view(-1, self.state_dim)).view(-1, n_sigma, self.obs_dim)

        mu_z, S_z = self.unscented_transform(sigma_measurements, self.root_R)
        return mu_z, S_z.matmul(S_z.transpose(1, 2))


class MerweSigmaPointsSR:

    def __init__(self, n, alpha=0.1, beta=2.0, kappa=-2.0, device='cuda:0'):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.kappa = 3.0 - n
        self.l = alpha ** 2 * (n + kappa) - n
        self.n = n
        self.device = device
        self.Wm, self.Wc = self.compute_weights()
        self.p = torch.sqrt(torch.tensor([self.l + self.n], device=self.device))

        #self.p = torch.sqrt(self.n / (1.0 - self.Wm[0, 0]))

    def get_n_sigma(self):
        return 2 * self.n + 1

    def get_weights(self):
        return self.Wm, self.Wc

    def sigma_points(self, mu, S):
        ''' takes mu, S as input, where covariance = S @ S.T'''
        sigmas = [mu]

        for i in range(self.n):
            x1 = mu - self.p * S[:, :, i]
            x2 = mu + self.p * S[:, :, i]
            sigmas.extend([x1, x2])

        return torch.stack(sigmas, 1).view(-1, self.n)

    '''
    def compute_weights(self):
        W = [-1.0]

        W.extend([(1 - W[0]) / self.n] * 2 * self.n)

        W = torch.tensor(W, device=self.device).view(-1, 1)
        return W, W
    '''
    def compute_weights(self):
        Wm = [self.l / (self.n + self.l)]
        Wc = [Wm[0] + 1 - self.alpha ** 2 + self.beta]

        Wm.extend([1.0 / (2 * (self.n + self.l))] * 2 * self.n)
        Wc.extend([1.0 / (2 * (self.n + self.l))] * 2 * self.n)
        Wm = torch.tensor(Wm, device=self.device).view(-1, 1)
        Wc = torch.tensor(Wc, device=self.device).view(-1, 1)

        return Wm, Wc
