from abc import ABCMeta
import numpy as np
import torch
import torch.nn as nn


class BaseDynamics(nn.Module, metaclass=ABCMeta):
    """
    Base class for defining the dynamics/kinematics of a simple-model that in turns acts like:
    1. The exact transition function in the Heuristic Unscented Kalman (HUK) model
    2. The nominal transition function hat{f} for the GP learned model (GPUK) model
    """
    def __init__(self, device: str, log_normal_params: bool, mode: str):
        super(BaseDynamics, self).__init__()
        # Device on which data/compute sits
        self.device = device
        # Whether estimating sys-id parameters as log_normal values
        self.log_normal_params = log_normal_params
        # Purpose for which using dynamics
        #  Supported Options: filter, MPC
        self.mode: str = mode

        # - - - - - - - - - - - - - - - - -
        # Some parameters common to simple model transition functions
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
        # - - - - - - - - - - - - - - - - -

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


class CartpoleDynamics(BaseDynamics):
    def __init__(self, device: str = 'cuda:0', log_normal_params: bool = False, mode: str = "MPC"):
        super(CartpoleDynamics, self).__init__(device=device, log_normal_params=log_normal_params, mode=mode)

        # Doesn't make much difference so hard-coded to fixed value
        self.linear_damping = torch.tensor(0.2, device=self.device)

        # Unknown dynamics parameters defaults
        self.cart_mass_def = torch.tensor(1.0, device=self.device)
        self.pole_mass_def = torch.tensor(1.0, device=self.device)
        self.angular_damping_def = torch.tensor(0.1, device=self.device)

        # Default values that parameterize the dynamics of the system
        self.cart_mass = self.cart_mass_def
        self.pole_mass = self.pole_mass_def
        self.angular_damping = self.angular_damping_def

        # Number of states and actions in dynamics
        self.nx = 5
        self.nu = 1

    def reset_params(self):
        self.cart_mass = self.cart_mass_def
        self.pole_mass = self.pole_mass_def
        self.angular_damping = self.angular_damping_def

    def set_params(self, params):
        if self.log_normal_params:
            # If the params are the log of the actual values exponentiate ...
            transformed = self.log_transform(params)
            mp = transformed[0]
            mc = transformed[1]
            # The prior on b1 is 0.1 but for uniformity in the sys_id grad-descent optimizer
            #  we pretend to the outside world that the value is 10x so we can use the same optimization hparams
            b1 = 0.1 * transformed[2]
        else:
            raise NotImplementedError("Non log normal params are not implemented for Cartpole")

        self.pole_mass = mp
        self.cart_mass = mc
        self.angular_damping = b1

    def get_params(self):
        if self.log_normal_params:
            mp = self.pole_mass.log()
            mc = self.cart_mass.log()
            # The prior on b1 is 0.1 but for uniformity in the sys_id grad-descent optimizer
            #  we pretend to the outside world that the value is 10x so we can use the same optimization hparams
            b1 = (10 * self.angular_damping).log()
        else:
            raise NotImplementedError("Non log normal params are not implemented for Cartpole")

        return torch.stack((mp, mc, b1), dim=0)

    def forward(self, state, action):
        """
        Propagate state and action via cartpole dynamics
        :param state: Cartpole state definition: [x_cart (m), x_mass (m), y_mass (m), v_cart (m/s), theta_dot (rad/s)]
        :param action: Force on cart in N
        :return:
        """
        # Preprocess variables
        x = state[:, 0].view(-1, 1)
        xmass = state[:, 1].view(-1, 1)
        ymass = state[:, 2].view(-1, 1)
        # x_dot == v_cart
        x_dot = state[:, 3].view(-1, 1)
        # TODO: Replace by vxmass and vymass ?
        theta_dot = state[:, 4].view(-1, 1)

        dx = xmass - x  # mass-cart x-distance
        # Setup such that acute angles are measured from downward pointing vertical
        dy = self.y_cart - ymass    # mass-cart y-distance
        theta = torch.atan2(dx, dy) # Pole angle from downward pointing direction
        sintheta = torch.sin(theta)
        costheta = torch.cos(theta)
        l = (dx**2 + dy**2).sqrt()  # Pole-length

        mp = self.pole_mass
        mc = self.cart_mass
        b1 = self.angular_damping
        # Not sys-ided since less important
        b2 = self.linear_damping

        g = self.g

        # Clamp action magnitude just like it is clamped on complex object environment
        force = self.gear * action.clamp(min=-1, max=1)

        # Do dynamics
        # Sources
        # https://ocw.mit.edu/courses/6-832-underactuated-robotics-spring-2009/72bc06c4dc73315bf49c28a81dc2b996_MIT6_832s09_read_ch03.pdf
        # https://ocw.mit.edu/courses/6-832-underactuated-robotics-spring-2009/
        tmp = l * (mc + mp * sintheta * sintheta)
        # Acceleration of cart
        xacc = (force * l + mp * l * sintheta * (l * theta_dot * theta_dot + g * costheta) +
                costheta * b1 * theta_dot - l * b2 * x_dot) / tmp

        thetaacc = (-force * costheta -
                    mp * l * theta_dot * theta_dot * sintheta * costheta -
                    (mc + mp) * g * sintheta + b2 * x_dot * costheta - (mc + mp) * b1 * theta_dot) / tmp

        # Fill-out new values
        new_x_dot = x_dot + self.dt * xacc
        new_theta_dot = theta_dot + self.dt * thetaacc

        new_x = x + 0.5 * self.dt * (x_dot + new_x_dot)
        theta = theta + 0.5 * self.dt * (theta_dot + new_theta_dot)

        new_xmass = new_x + l * torch.sin(theta)
        new_ymass = -l * torch.cos(theta)

        vlim = 20
        new_theta_dot = new_theta_dot.clamp(min=-vlim, max=vlim)
        new_x_dot = new_x_dot.clamp(min=-vlim, max=vlim)

        next_state = torch.cat((new_x, new_xmass, new_ymass, new_x_dot,  new_theta_dot), 1)

        return next_state


class BallDynamics(BaseDynamics):
    def __init__(self, device: str = 'cuda:0', log_normal_params: bool = False, mode: str = "MPC"):
        super(BallDynamics, self).__init__(device=device, log_normal_params=log_normal_params, mode=mode)

        # Unknown dynamics parameters defaults
        self.robot_mass_def = torch.tensor(1.0, device=self.device)

        # Default values that parameterize the dynamics of the system
        self.robot_mass = self.robot_mass_def

    def reset_params(self):
        self.robot_mass = self.robot_mass_def

    def set_params(self, params):
        if self.log_normal_params:
            transformed = self.log_transform(params)
            m_robot = transformed[0]
        else:
            raise NotImplementedError("Non log normal params are not implemented for Ball")

        self.robot_mass = m_robot

    def get_params(self):
        if self.log_normal_params:
            m_robot = self.robot_mass.log()
        else:
            raise NotImplementedError("Non log normal params are not implemented for Cartpole")
        return torch.stack((m_robot,), dim=0)

    def propagate_MPC(self, state, action):
        """
        Version of forward when using full dynamics including actuator kinematics/dynamics for planning
        :param state:
        :param action:
        :return:
        """

    def forward(self, state, action):
        """
        Propagate state and action via freely falling ball dynamics
        :param state: Ball state definition: [x_robot(m), x_ball (m), x_ball (m),
                                                v_robot (m), vx_ball (m), vy_ball (m/s)]
        :param action: Simply moves the cart around as if there were no tension in the rope (i.e. no effect on cup)
        :return:
        """
        if self.mode == "MPC":
            # Preprocess variables
            x_robot = state[:, 0].view(-1, 1)
            xball = state[:, 1].view(-1, 1)
            yball = state[:, 2].view(-1, 1)
            # x_dot == v_cart
            v_robot = state[:, 3].view(-1, 1)
            vx_ball = state[:, 4].view(-1, 1)
            vy_ball = state[:, 5].view(-1, 1)

            g = self.g

            # Clamp action magnitude just like it is clamped on complex object environment
            force = self.gear * action.clamp(min=-1, max=1)

            # Do dynamics for freely falling ball under gravity
            # Acceleration of robot
            robot_acc = self.robot_mass / force

            ball_yacc = -g
            ball_xacc = 0.0

            # Fill-out new values
            new_vxball = vx_ball + self.dt * ball_xacc
            new_vyball = vy_ball + self.dt * ball_yacc
            new_v_robot = v_robot + self.dt * robot_acc

            new_xball = xball + 0.5 * self.dt * (vx_ball + new_vxball)
            new_yball = yball + 0.5 * self.dt * (vy_ball + new_vyball)
            new_x_robot = x_robot + 0.5 * self.dt * (v_robot + new_v_robot)

            vlim = 20
            new_vxball = new_vxball.clamp(min=-vlim, max=vlim)
            new_vyball = new_vyball.clamp(min=-vlim, max=vlim)
            new_v_robot = new_v_robot.clamp(min=-vlim, max=vlim)

            next_state = torch.cat((new_x_robot, new_xball, new_yball, new_v_robot, new_vxball, new_vyball), 1)

            return next_state



if __name__ == '__main__':
    dynamics = CartpoleDynamics()
