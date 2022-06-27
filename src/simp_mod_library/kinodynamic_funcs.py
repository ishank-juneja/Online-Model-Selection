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


class CartpoleDynamics(BaseDynamics):
    """
    The known closed form dynamics for Cartpole
    # Sources
    # https://ocw.mit.edu/courses/6-832-underactuated-robotics-spring-2009/72bc06c4dc73315bf49c28a81dc2b996_MIT6_832s09_read_ch03.pdf
    # https://ocw.mit.edu/courses/6-832-underactuated-robotics-spring-2009/
    """
    def __init__(self, device: str = 'cuda:0', log_normal_params: bool = False):
        super(CartpoleDynamics, self).__init__(device=device, log_normal_params=log_normal_params)

        # Number of states and actions in dynamics
        self.nx = 6
        self.nu = 1

        # Doesn't make much difference so hard-coded to fixed value, not learned
        self.linear_damping = torch.tensor(0.2, device=self.device)

        # Unknown cartpole dynamics parameters defaults as floats
        self.pole_mass_def = 1.0
        self.angular_damping_def = 0.1
        self.rob_mass_def: float = 1.0

        # Init all the learned dynamics params tensors to defaults
        self.rob_mass = torch.tensor(self.rob_mass_def, device=self.device)
        self.pole_mass = torch.tensor(self.pole_mass_def, device=self.device)
        self.angular_damping = torch.tensor(self.angular_damping_def, device=self.device)

    def reset_params(self):
        self.rob_mass = torch.tensor(self.rob_mass_def, device=self.device)
        self.pole_mass = torch.tensor(self.pole_mass_def, device=self.device)
        self.angular_damping = torch.tensor(self.angular_damping_def, device=self.device)

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
        self.rob_mass = mc
        self.angular_damping = b1

    def get_params(self):
        if self.log_normal_params:
            mp = self.pole_mass.log()
            mc = self.rob_mass.log()
            # The prior on b1 is 0.1 but for uniformity in the sys_id grad-descent optimizer
            #  we pretend to the outside world that the value is 10x so we can use the same optimization hparams
            b1 = (10 * self.angular_damping).log()
        else:
            # TODO: Remove the non-log case if not being used so this goes away...
            raise NotImplementedError("Non log normal params are not implemented for Cartpole")

        return torch.stack((mp, mc, b1), dim=0)

    def forward(self, state, action):
        """
        Propagate state and action via cartpole dynamics
        :param state: [x_rob, dot{x}_rob (Formerly x_cart and v_cart), x_mass, y_mass, dot{x}_mass, dot{y}_mass]
        :param action: Force on cart in N
        :return:
        """
        # Preprocess variables and view them as 2D tensors (instead of just vectors)
        x_rob = state[:, 0].view(-1, 1)
        # x and y positions of mass
        xmass = state[:, 2].view(-1, 1)
        ymass = state[:, 3].view(-1, 1)
        # x_rob_dot == v_cart == v_robot
        x_rob_dot = state[:, 1].view(-1, 1)
        # x and y velocities of mass
        xmass_dot = state[:, 4].view(-1, 1)
        ymass_dot = state[:, 5].view(-1, 1)

        dx = xmass - x_rob  # mass-cart x_rob-distance
        # Setup such that acute angles are measured from downward pointing vertical
        dy = self.y_cart - ymass    # mass-cart y-distance
        theta = torch.atan2(dx, dy) # Pole angle from downward pointing direction
        sintheta = torch.sin(theta)
        costheta = torch.cos(theta)
        # Geom. length of the pole of the cartpole
        l = (dx**2 + dy**2).sqrt()
        # Compute angular velocity theta_dot from linear vel., l, and sin/cos thetas
        #  Theta dot can be estimated separately from xmass_dot and ymass_dot
        est_x = xmass_dot / (l * costheta)
        est_y = ymass_dot / (l * sintheta)
        theta_dot = (est_x + est_y) / 2

        mp = self.pole_mass
        mc = self.rob_mass  # Cart and robot are one and the same for the cartpole model
        b1 = self.angular_damping
        # Not sys-ided since less important
        b2 = self.linear_damping

        g = self.g

        # Clamp action magnitude just like it is clamped on complex object environment
        force = self.gear * action.clamp(min=-1, max=1)

        # Do dynamics
        tmp = l * (mc + mp * sintheta * sintheta)
        # Acceleration of cart (point robot welded to cart)
        x_rob_acc = (force * l + mp * l * sintheta * (l * theta_dot * theta_dot + g * costheta) +
                     costheta * b1 * theta_dot - l * b2 * x_rob_dot) / tmp

        thetaacc = (-force * costheta -
                    mp * l * theta_dot * theta_dot * sintheta * costheta -
                    (mc + mp) * g * sintheta + b2 * x_rob_dot * costheta - (mc + mp) * b1 * theta_dot) / tmp

        # Propagate all the state variables using the qty and its derivative
        new_x_rob_dot = self.propagate(x_rob_dot, x_rob_acc)
        new_theta_dot = self.propagate(theta_dot, thetaacc)
        # The derivative for the static qts is averaged between cur and next
        new_x_rob = self.propagate(x_rob, 0.5 * (x_rob_dot + new_x_rob_dot))
        new_theta = self.propagate(x_rob, 0.5 * (theta_dot + new_theta_dot))

        # Cache trig functions of new theta
        new_sintheta = torch.sin(new_theta)
        new_costheta = torch.cos(new_theta)

        # New mass coordinates
        new_xmass = new_x_rob + l * new_sintheta
        new_ymass = -l * new_costheta

        vlim = self.vlim
        new_theta_dot = new_theta_dot.clamp(min=vlim, max=vlim)
        new_x_rob_dot = new_x_rob_dot.clamp(min=vlim, max=vlim)

        # Use computed theta_dot to get estimates for next dot_{x/y}_mass
        new_dot_xmass = l * new_costheta * new_theta_dot
        new_dot_ymass = l * new_sintheta * new_theta_dot

        next_state = torch.cat((new_x_rob, new_x_rob_dot, new_xmass, new_ymass, new_dot_xmass, new_dot_ymass), dim=1)
        return next_state


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
