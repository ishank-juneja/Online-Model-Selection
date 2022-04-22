import torch
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy, MultitaskVariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel


class MultitaskSVGP(gpytorch.models.ApproximateGP):

    def __init__(self, inducing_points, input_dim, output_dim):
        batch_shape = torch.Size([output_dim])
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=batch_shape)

        variational_strategy = MultitaskVariationalStrategy(VariationalStrategy(self, inducing_points,
                                                                                variational_distribution,
                                                                                learn_inducing_locations=True),
                                                            num_tasks=output_dim)
        super().__init__(variational_strategy)

        self.covar_module = ScaleKernel(RBFKernel(batch_shape=batch_shape,
                                                  ard_num_dims=input_dim),
                                        batch_shape=batch_shape,
                                        ard_num_dims=input_dim)
        self.mean_module = gpytorch.means.ZeroMean()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPDynamics(gpytorch.Module):
    def __init__(self, input_dim, output_dim, num_inducing=100, device='cuda:0', nominal_dynamics=None):
        super(GPDynamics, self).__init__()
        inducing_points = torch.randn(output_dim, num_inducing, input_dim, device=device)
        self.model = MultitaskSVGP(inducing_points, input_dim, output_dim)
        self.x_logstd = torch.nn.Parameter(torch.log(0.17 * torch.ones(output_dim, device=device)), requires_grad=True)
        self.nominal_dynamics = nominal_dynamics

        for params in self.nominal_dynamics.parameters():
            params.requires_grad = False

    def reset_params(self):
        if self.nominal_dynamics is not None:
            self.nominal_dynamics.reset_params()

    def set_params(self, params):
        if self.nominal_dynamics is not None:
            self.nominal_dynamics.set_params(params)

    def forward(self, x, a, params=None):
        model_input = torch.cat((x, a), dim=1)
        px = self.model(model_input)
        mu = px.mean
        # do scale to avoid singular issues
        scale = torch.diag_embed(px.stddev + self.x_logstd.exp())
        #sigma = torch.diag_embed(px.variance + (2 * self.x_logstd).exp())
        if self.nominal_dynamics is not None:
            mu = mu + self.nominal_dynamics(x, a)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale)

