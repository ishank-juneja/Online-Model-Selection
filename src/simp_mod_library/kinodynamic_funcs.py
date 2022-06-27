from abc import ABCMeta
import numpy as np
import torch
import torch.nn as nn


class BaseDynamics(nn.Module, metaclass=ABCMeta):
    """
    Base class for simple model dynamics/kimematics
    NB: There is no consideration for uncertainty in any transitions in the derived classes
    """
    def __init__(self, device: str, log_normal_params: bool):
        super(BaseDynamics, self).__init__()
        # Device on which data/compute sits
        self.device = device
        # Whether estimating sys-id parameters as log_normal values
        self.log_normal_params = log_normal_params

        # - - - - - - - - - - - - - - - - -
        # Some numerical parameters (numbers) common to simple model transition functions
        # self.dt should be the same as env.frame_skip * env.dt for any of the complex environments
        # If using a sequence of frames to train simple model dynamics then below and env.frame_skip * env.dt of
        #  both the simple and complex model envs should be the same
        self.dt = torch.tensor(0.05, device=self.device)
        self.g = torch.tensor(9.81, device=self.device)
        self.pi = torch.tensor(np.pi, device=self.device)
        # y-position of cart/robot is fixed in 1D actuated models
        self.y_cart = 0.0
        # The gear/force action amplification value for all environments
        #  Must be the same as gear attribute of actuator in complex environment xml
        self.gear = 40.0
        # All velocities for mjco simulated problems are capped at 20m/s
        self.vlim = 20
        # - - - - - - - - - - - - - - - - -

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Placeholder/template for propagating dynamics
        :param state:
        :param action:
        :return:
        """
        raise NotImplementedError

    def reset_params(self):
        """
        Placeholder/template for reset-ing learnable (sys-id-ed) dynamics parametrs
        :return:
        """
        raise NotImplementedError

    def set_params(self, params):
        """
        Placeholder/template for setting learnable (sys-id-ed) dynamics parametrs to new learned values
        Call at the end of a sys-id iteration
        :return:
        """
        raise NotImplementedError

    def get_params(self):
        """
        Placeholder/template for getting the current learnable (sys-id-ed) dynamics parameters
        :return:
        """
        raise NotImplementedError

    @classmethod
    def __new__(cls, *args, **kwargs):
        """
        Make abstract base class non-instaiable
        :param args:
        :param kwargs:
        """
        if cls is BaseDynamics:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return object.__new__(cls)

    @staticmethod
    def log_transform(params: torch.Tensor):
        """
        :param params: 1D torch tensor
        :return:
        """
        return torch.exp(params)

    def propagate(self, var, var_dot):
        """
        Propagate var to the next time step using var_dot
        :param var:
        :param var_dot:
        :return:
        """
        new_var = var + var_dot * self.dt
        return new_var


class CartpoleDynamics(nn.Module):

    def __init__(self, params_in_state=False, device='cuda:0', log_normal_params=False):
        super(CartpoleDynamics, self).__init__()
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


class BallDynamics(BaseDynamics):
    """
    Known closed form dynamics for a freely falling ball
    Freely falling ball includes a point-robot (attached to) an unknown mass
    """
    def __init__(self, device: str = 'cuda:0', log_normal_params: bool = False):
        super(BallDynamics, self).__init__(device=device, log_normal_params=log_normal_params)

        # Number of states and actions in dynamics
        self.nx = 6
        self.nu = 1

        # Unknown dynamics parameters defaults (True values from xml)
        self.rob_mass_def = 1.5
        # Default values that parameterize the dynamics of the system
        self.rob_mass = torch.tensor(self.rob_mass_def, device=self.device)

    def reset_params(self):
        self.rob_mass = torch.tensor(self.rob_mass_def, device=self.device)

    def set_params(self, params):
        if self.log_normal_params:
            transformed = self.log_transform(params)
            m_robot = transformed[0]
        else:
            raise NotImplementedError("Non log normal params are not implemented for Ball")
        self.rob_mass = m_robot

    def get_params(self):
        if self.log_normal_params:
            m_robot = self.robot_mass.log()
        else:
            raise NotImplementedError("Non log normal params are not implemented for Cartpole")
        return torch.stack((m_robot,), dim=0)

    def forward(self, state, action):
        """
        Propagate state and action via freely-falling ball dynamics
        :param state: [x_rob, dot{x}_rob, x_ball, y_ball, dot{x}_ball, dot{y}_ball]
        :param action: Force on point-robot in N
        :return:
        """
        # Preprocess variables and view them as 2D tensors (instead of just vectors)
        x_rob = state[:, 0].view(-1, 1)
        x_rob_dot = state[:, 1].view(-1, 1)

        xball = state[:, 2].view(-1, 1)
        yball = state[:, 3].view(-1, 1)

        vx_ball = state[:, 4].view(-1, 1)
        vy_ball = state[:, 5].view(-1, 1)

        g = self.g

        # Clamp action magnitude just like it is clamped on complex object environment
        force = self.gear * action.clamp(min=-1, max=1)

        # Do dynamics for freely falling ball under gravity
        # Acceleration of robot
        robot_acc = self.rob_mass / force

        ball_yacc = -g
        ball_xacc = 0.0

        # Fill-out new values
        # 1st order terms
        new_vxball = self.propagate(vx_ball, ball_xacc)
        new_vyball = self.propagate(vy_ball, ball_yacc)
        new_v_rob = self.propagate(x_rob_dot, robot_acc)
        # 0 order terms with derivative average of old and new values
        new_xball = self.propagate(xball, 0.5 * (vx_ball + new_vxball))
        new_yball = self.propagate(yball, 0.5 * (vy_ball + new_vyball))
        new_x_rob = self.propagate(x_rob, 0.5 * (x_rob_dot + new_v_rob))

        vlim = self.vlim
        new_vxball = new_vxball.clamp(min=-vlim, max=vlim)
        new_vyball = new_vyball.clamp(min=-vlim, max=vlim)
        new_v_rob = new_v_rob.clamp(min=-vlim, max=vlim)

        # Assemble new state by stacking individuals column-wise
        next_state = torch.cat((new_x_rob, new_v_rob, new_xball, new_yball, new_vxball, new_vyball), dim=1)

        return next_state


if __name__ == '__main__':
    dynamics = CartpoleDynamics()
