import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.distributions import Normal
from src.quaternion import qrot, qinverse, expmap_to_quaternion, qmul, eulerq, qeuler

from src.utils import bdot


class PendulumDynamics(nn.Module):

    def __init__(self, mass=1.0, length=1.0):
        super(PendulumDynamics, self).__init__()
        self.dt = torch.tensor(0.05)
        self.g = torch.tensor(9.8)
        self.pi = torch.tensor(np.pi)

    def forward(self, state, action):
        ''' state - theta, theta_dot, m, l action = torque'''

        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)
        m = state[:, 2].view(-1, 1)
        l = state[:, 3].view(-1, 1)

        action = torch.clamp(action, -2.0, 2.0)
        thdot = torch.clamp(thdot, -8.0, 8.0)

        newthdot = thdot + (-3 * self.g / (2 * torch.abs(l)) * torch.sin(th + self.pi) +
                            3. / (torch.abs(m) * torch.pow(l, 2)) * action) * self.dt

        newth = th + newthdot * self.dt
        new_state = torch.cat((newth, newthdot, l, m), 1)
        return new_state


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


class OneLinkRopeDynamics(nn.Module):

    def __init__(self, *args, **kwargs):
        super(OneLinkRopeDynamics, self).__init__()

    def forward(self, state, action):
        N, _ = state.size()
        pos = state[:, :3]
        quat = state[:, 3:]
        quat_inverse = qinverse(quat)

        # gripper relative positions
        gripper_1 = torch.zeros_like(pos)
        gripper_2 = torch.zeros_like(pos)
        gripper_1[:, 0] -= 0.5
        gripper_2[:, 0] += 0.5

        # Get dgrip (action) in frame of link
        dgrip_1 = qrot(quat_inverse, action[:, :3])
        dgrip_2 = qrot(quat_inverse, action[:, 3:])

        # Get new position in relative frame
        new_grip1 = gripper_1 + dgrip_1
        new_grip2 = gripper_2 + dgrip_2
        new_pos_rel = (new_grip1 + new_grip2) * 0.5

        # Find relative rotation
        direction = new_grip2 - new_pos_rel
        canonical_direction = torch.zeros_like(direction)
        canonical_direction[:, 0] = 1.
        quat_rel = rotation_between_two_vectors(canonical_direction, direction)

        # Get new state into world frame
        new_pos = pos + qrot(quat, new_pos_rel)

        # Compose relative rotation with new rotation to get into world frame
        new_quat_pre = qmul(quat, quat_rel)

        # Need to zero the x euler angle (for rotational symmetry)
        euler = qeuler(new_quat_pre, 'zyx') * torch.tensor([0., 1., 1.], device=quat.device).unsqueeze(0)
        new_quat = eulerq(euler, 'zyx')

        if False:#state.requires_grad:
            new_quat.register_hook(lambda grad: print('new_quat_post', (grad != grad).any()))
            euler.register_hook(lambda grad: print('euler', (grad != grad).any()))
            new_quat_pre.register_hook(lambda grad: print('new_quat_pre', (grad != grad).any()))
            #new_quat_pre.register_hook(lambda grad: print('new_quat_pre', grad))

            new_pos.register_hook(lambda grad: print('new_pos', (grad != grad).any()))
            quat_rel.register_hook(lambda grad: print('quat_rel', (grad != grad).any()))
            direction.register_hook(lambda grad: print('direction', (grad != grad).any()))

            new_pos_rel.register_hook(lambda grad: print('new_pos_rel', (grad != grad).any()))

            new_grip2.register_hook(lambda grad: print('ngrip_2', (grad != grad).any()))
            new_grip1.register_hook(lambda grad: print('ngrip_1', (grad != grad).any()))
            dgrip_2.register_hook(lambda grad: print('dgrip_2', (grad != grad).any()))
            dgrip_1.register_hook(lambda grad: print('dgrip_1', (grad != grad).any()))
            quat_inverse.register_hook(lambda grad: print('quat_inverse', (grad != grad).any()))

            state.register_hook(lambda grad: print('state', (grad != grad).any()))

        # Return new state
        return torch.cat((new_pos, new_quat), dim=1)

    def reset_params(self):
        pass


class OneLinkRopeDynamicsGripperPos(nn.Module):

    def __init__(self, *args, **kwargs):
        super(OneLinkRopeDynamicsGripperPos, self).__init__()

    def forward(self, state, action):
        gripper_1, gripper_2 = torch.chunk(state[:, :6], chunks=2, dim=1)
        dgripper_1, dgripper_2 = torch.chunk(action[:, :6], chunks=2, dim=1)
        length = torch.norm(gripper_1 - gripper_2, dim=1, keepdim=True)

        new_gripper_1 = gripper_1 + dgripper_1
        new_gripper_2 = gripper_2 + dgripper_2

        # Midpoint
        new_midpoint = 0.5 * (new_gripper_2 + new_gripper_1)

        # Keep constraint that they are the same length apart
        diff_g1 = (new_gripper_1 - new_midpoint) / torch.norm(new_gripper_1 - new_midpoint, dim=1, keepdim=True)
        diff_g2 = (new_gripper_2 - new_midpoint) / torch.norm(new_gripper_2 - new_midpoint, dim=1, keepdim=True)
        new_gripper_1 = new_midpoint + 0.5 * length * diff_g1
        new_gripper_2 = new_midpoint + 0.5 * length * diff_g2

        return torch.cat((new_gripper_1, new_gripper_2), dim=1)

    def reset_params(self):
        pass


def rotation_between_two_vectors(v1, v2):
    ''' rotation between two vectors'''
    eps = 1e-3
    # normalize
    v1_norm = v1 / torch.norm(v1, dim=1, keepdim=True)
    norm_v2 = torch.norm(v2, dim=1, keepdim=True)
    v2_norm = v2 / norm_v2

    dot_prod = (bdot(v1_norm, v2_norm)).unsqueeze(1).clamp(min=-1 + eps, max=1. - eps)

    angle = torch.acos(dot_prod)
    axis_pre = torch.cross(v1_norm, v2_norm, dim=1)

    # If norm of axis is approx zero then parallel
    # #TODO not sure about direction for parallel (probably not neccesariy for small actions)
    # Since would need to do a 180 flip in a single step

    norm = torch.norm(axis_pre, dim=1, keepdim=True)
    axis = torch.where(norm < eps, v1, axis_pre / norm)
    angle = torch.where(norm < eps, torch.zeros_like(angle), angle)

    if False:#v2.requires_grad:
        print(norm)
        for i in range(v2_norm.size(0)):
            print(axis[i])
            if (v2_norm[i] != v2_norm[i]).any() or (v2_norm[i].abs() > 1e9).any():
                print('--')
                print(v2_norm[i])
                print(v2[i])
                exit(0)
        dot_prod.register_hook(lambda grad: print('dot_prod', (grad != grad).any()))
        axis_pre.register_hook(lambda grad: print('axis_pre', (grad != grad).any()))
        norm.register_hook(lambda grad: print('norm', (grad != grad).any()))
        v2_norm.register_hook(lambda grad: print('normalized v2', (grad != grad).any()))
        v2_norm.register_hook(lambda grad: print('normalized v2', grad))
    return axis_angle_to_quat(axis, angle)


def axis_angle_to_quat(axis, angle):
    assert axis.size(-1) == 3

    ax, ay, az = torch.chunk(axis, dim=1, chunks=3)

    qx = ax * torch.sin(angle / 2)
    qy = ay * torch.sin(angle / 2)
    qz = az * torch.sin(angle / 2)
    qw = torch.cos(angle / 2)

    qt = torch.cat((qw, qx, qy, qz), dim=1)
    q = qt / torch.norm(qt, dim=1, keepdim=True)

    if False:#angle.requires_grad:
        q.register_hook(lambda grad: print('q', (grad != grad).any()))
        qt.register_hook(lambda grad: print('qt', (grad != grad).any()))
        angle.register_hook(lambda grad: print('angle', (grad != grad).any()))
        axis.register_hook(lambda grad: print('axis', (grad != grad).any()))

    return torch.where(bdot(q, q).unsqueeze(1) > 0, q, -q)


def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    dvec = (M - m) * torch.ones_like(x)

    wrapped_high = wrap_high(x, M, dvec)
    return wrap_low(wrapped_high, m, dvec)


def wrap_high(x, m, dvec):
    return torch.where(x > m, x - dvec, x)


def wrap_low(x, m, dvec):
    return torch.where(x < m, x + dvec, x)
