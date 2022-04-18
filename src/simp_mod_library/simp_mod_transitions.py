import numpy as np
import torch
import torch.nn as nn


class CartPoleDynamics(nn.Module):
    def __init__(self, params_in_state=True, device='cuda:0', log_normal_params=False):
        super(CartPoleDynamics, self).__init__()
        self.dt = torch.tensor(0.05, device=device)
        self.g = torch.tensor(9.81, device=device)
        self.pi = torch.tensor(np.pi, device=device)
        self.device = device
        self.trig = True
        self.linear_damping = torch.tensor(0.2, device=device)

        if not params_in_state:
            self.cart_mass = torch.tensor(1.0, device=device)
            self.pole_mass = torch.tensor(1.0, device=device)
            self.angular_damping = torch.tensor(0.1, device=device)

        self.log_normal_params = log_normal_params
        self.params = params_in_state

        self.update_delta = 1.0

    def reset_params(self):
        self.cart_mass = torch.tensor(1.0, device=self.device)
        self.pole_mass = torch.tensor(1.0, device=self.device)
        self.angular_damping = torch.tensor(0.1, device=self.device)

    def set_params(self, params):
        if self.log_normal_params:
            mp, mc, b1 = self.log_transform(params)
        else:
            mp, mc, b1 = params

        self.pole_mass = mp
        self.cart_mass = mc
        self.angular_damping = b1

    def get_params(self):
        if self.log_normal_params:
            mp = self.pole_mass.log()
            mc = self.cart_mass.log()
            b1 = (10 * self.angular_damping).log()
        else:
            mp = self.pole_mass
            mc = self.cart_mass
            b1 = self.angular_damping

        return torch.stack((mp, mc, b1), dim=0)

    def log_transform(self, params):
        mp = torch.exp(params[0])
        mc = torch.exp(params[1])
        b1 = 0.1 * torch.exp(params[2])
        return mp, mc, b1

    def forward(self, state, action):
        # STATE INPUT: x, x_end, y_end, x_dot, theta_dot

        # Preprocess variables
        x = state[:, 0].view(-1, 1)
        xmass = state[:, 1].view(-1, 1)
        ymass = state[:, 2].view(-1, 1)
        x_dot = state[:, 3].view(-1, 1)
        theta_dot = state[:, 4].view(-1, 1)

        # Get angle and length
        dx = xmass - x
        theta = torch.atan2(dx, -ymass)
        sintheta = torch.sin(theta)
        costheta = torch.cos(theta)
        l = (dx**2 + ymass**2).sqrt()

        # Dynamic parameters
        if self.params:
            mp = state[:, 5].view(-1, 1)
            mc = state[:, 6].view(-1, 1)
            b1 = state[:, 7].view(-1, 1)

            if self.log_normal_params:
                mp = torch.exp(mp)
                mc = torch.exp(mc)
                b1 = 0.1 * torch.exp(b1)
        else:
            mp = self.pole_mass
            mc = self.cart_mass
            b1 = self.angular_damping

        b2 = self.linear_damping

        g = self.g
        force = -40.0 * action.clamp(min=-1, max=1)

        # Do dynamics
        tmp = l * (mc + mp * sintheta * sintheta)
        xacc = (force * l + mp * l * sintheta * (l * theta_dot * theta_dot + g * costheta) +
                costheta * b1 * theta_dot - l * b2 * x_dot) / tmp

        thetaacc = (-force * costheta -
                    mp * l * theta_dot * theta_dot * sintheta * costheta -
                    (mc + mp) * g * sintheta + b2 * x_dot * costheta - (mc + mp) * b1 * theta_dot) / tmp

        # Fillout new values
        new_x_dot = x_dot + self.dt * xacc
        new_theta_dot = theta_dot + self.dt * thetaacc

        new_x = x + 0.5 * self.dt * (x_dot + new_x_dot)
        theta = theta + 0.5 * self.dt * (theta_dot + new_theta_dot)

        new_xmass = new_x + l * torch.sin(theta)
        new_ymass = -l * torch.cos(theta)

        vlim = 20
        new_theta_dot = new_theta_dot.clamp(min=-vlim, max=vlim)
        new_x_dot = new_x_dot.clamp(min=-vlim, max=vlim)

        if self.params:
            if self.log_normal_params:
                mp = torch.log(mp)
                mc = torch.log(mc)
                b1 = torch.log(b1 / 0.1)

            state = torch.cat((new_x, new_xmass, new_ymass, new_x_dot, new_theta_dot,
                               mp, mc, b1), 1)

        else:
            state = torch.cat((new_x, new_xmass, new_ymass, new_x_dot,  new_theta_dot), 1)

        return state
