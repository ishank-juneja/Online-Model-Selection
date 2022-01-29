import torch
from torch.distributions import MultivariateNormal


class KalmanFilter:
    '''
        Implementation of Kalman filter in torch
        -- differentiable and takes batches
    '''

    def __init__(self, control_dim, Q, R, device):
        self.state_dim = Q.size()[1]
        self.control_dim = control_dim
        self.observation_dim = R.size()[1]
        self.Q = Q
        self.R = R
        self.device = device

    def predict(self, A, B, u, mu, sigma):
        # Prediction step
        mu_bar = A.matmul(mu) + B * u
        sigma_bar = A.matmul(sigma).matmul(A.transpose(1, 2)) + self.Q
        return mu_bar, sigma_bar

    def update(self, C, z, mu_bar, sigma_bar):

        # Compute Kalman Gain
        S = C.matmul(sigma_bar).matmul(C.transpose(1, 2)) + self.R
        S_inv = S.inverse()
        K = sigma_bar.matmul(C.transpose(1, 2)).matmul(S_inv)

        # Get innovation
        innov = z - C.matmul(mu_bar)

        # Get new mu
        mu = mu_bar + K.matmul(innov)

        # Compute sigma using Joseph's form -- should be better numerically
        IK_C = torch.eye(self.state_dim, device=self.device) - K.matmul(C)
        KRK = K.matmul(self.R.matmul(K.transpose(1, 2)))
        sigma = IK_C.matmul(sigma_bar.matmul(IK_C.transpose(1, 2))) + KRK

        return mu, sigma

    def filter(self, A, B, C, controls, observations, prior_mu, prior_sigma):
        predictive_mus = []
        predictive_sigmas = []
        filtered_mus = []
        filtered_sigmas = []

        mu_bar = prior_mu
        sigma_bar = prior_sigma

        # Gets predictive and filtered distributions
        for i in range(controls.size()[1]):
            # Measurement step
            mu, sigma = self.update(C[:, i], observations[:, i].view(-1, self.observation_dim, 1), mu_bar, sigma_bar)

            # Predictive step
            mu_bar, sigma_bar = self.predict(A[:, i], B[:, i],
                                             controls[:, i].view(-1, self.control_dim, 1), mu, sigma)

            predictive_mus.append(mu_bar)
            predictive_sigmas.append(sigma_bar)
            filtered_mus.append(mu)
            filtered_sigmas.append(sigma)

        return torch.stack(filtered_mus, 1), torch.stack(filtered_sigmas, 1), \
            torch.stack(predictive_mus, 1), torch.stack(predictive_sigmas, 1)

    def smooth(self, A, forward_states):
        filtered_mu, filtered_sigma, pred_mu, pred_sigma = forward_states
        T = filtered_mu.size()[1]

        sigma = filtered_sigma[:, -1]
        mu = filtered_mu[:, -1]
        smoothed_mu = [mu]
        smoothed_sigma = [sigma]

        for t in range(2, T+1):
            fmu = filtered_mu[:, -t]
            fsig = filtered_sigma[:, -t]
            pmu = pred_mu[:, -t]
            psig = pred_sigma[:, -t]

            # Get smoother gain
            J = fsig.matmul(A[:, -t].transpose(1, 2)).matmul(psig.inverse())

            mu = fmu + J.matmul(mu - pmu)
            sigma = fsig + J.matmul(sigma - psig).matmul(J.transpose(1, 2))

            smoothed_mu.append(mu)
            smoothed_sigma.append(sigma)

        return torch.stack(smoothed_mu[::-1], 1), torch.stack(smoothed_sigma[::-1], 1)

    def set_qr(self, q, r):
        ''' takes diagonal vectors of root_q and root_R'''

        # print(q)
        # need to add some small minimum amount for stability
        qn = q + 1e-2
        rn = r + 1e-2
        self.Q = torch.diag(qn.pow(2))
        self.R = torch.diag(rn.pow(2))

    def get_log_likelihoods(self, smoothed_states, A, B, C, observations, controls, prior_sigma, forward_states):
        smoothed_mu = smoothed_states[0]
        smoothed_sigma = smoothed_states[1]

        batch_size = smoothed_mu.size()[0]
        traj_len = smoothed_mu.size()[1]

        zero_state = torch.zeros(self.state_dim, device=self.device)
        zero_observation = torch.zeros(self.observation_dim, device=self.device)
        state_identity = prior_sigma[0]

        mvn_smooth = MultivariateNormal(smoothed_mu.view(batch_size, traj_len, self.state_dim), smoothed_sigma)

        # Sample trajectory
        x_smooth = mvn_smooth.rsample()
        entropy = -mvn_smooth.log_prob(x_smooth)

        # Transition log_probs
        A_tmp = A[:, :-1].matmul(x_smooth[:, :-1].view(batch_size, -1, self.state_dim, 1))
        B_tmp = B[:, :-1].matmul(controls[:, :-1].view(batch_size, -1, self.control_dim, 1))
        mu_transition = (A_tmp + B_tmp).view(batch_size, -1, self.state_dim)

        # Center around zero mean
        transition_centered = x_smooth[:, 1:] - mu_transition
        mvn_transition = MultivariateNormal(zero_state, self.Q)
        log_probs_transition = mvn_transition.log_prob(transition_centered.view(-1, self.state_dim))

        # Do emission log probs
        mu_emission = C.matmul(x_smooth.view(batch_size, -1, self.state_dim, 1)).view(batch_size,
                                                                                      -1, self.observation_dim)
        emission_centered = observations - mu_emission
        mvn_emission = MultivariateNormal(zero_observation, self.R)
        log_probs_emission = mvn_emission.log_prob(emission_centered.view(-1, self.observation_dim))

        # Get log probs on prior
        mvn_prior = MultivariateNormal(zero_state, state_identity)
        log_probs_prior = mvn_prior.log_prob(x_smooth[:, 0, :])

        log_probs = torch.stack([log_probs_transition.sum(),
                                 log_probs_emission.sum(),
                                 log_probs_prior.sum(),
                                 entropy.sum()], 0)

        # scale log_probs by total data
        return log_probs / batch_size
