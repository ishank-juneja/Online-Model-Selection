from gym import utils
from gym.envs.mujoco import mujoco_env
from gym_cenvs.envs.base import MujocoBase
import json
import numpy as np
import os


class Doublecartpole(mujoco_env.MujocoEnv, utils.EzPickle, MujocoBase):
    def __init__(self, for_ball: bool = False, invisible_rods: bool = False):
        xml_path = os.path.abspath('gym_cenvs/assets/doublecartpole_dynamic.xml')

        # - - - - - Init Attributes Prior since some needed for model creation - - - - - -
        self.for_ball = for_ball
        self.invisible_rods = invisible_rods

        # Episode/task of a contiguous trajectory complete?
        self.done = False

        # This randomization randomizes over color only, rand over geom is via pymjcf
        #  Randomization over geom was unsuccessful with mujoco_py interface
        self.randomize = True

        with open('gym_cenvs/assets/dcpole.json') as f:
            params = json.loads(f.read())

        # # - - - - -
        # Geom size attributes randomized over
        # Full length of link1 in dcartpole model
        self.pole1_len = params['pole1']
        # Full length of link2 in dcartpole model
        self.pole2_len = params['pole2']
        # Pole widths (same for both poles)
        self.pole_width = params['pole_width']
        # Radius of mass at the end of pole
        self.mass_radius = params['mass_radius']

        # - - - - - - - - - - - - - - - -
        MujocoBase.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip=1)
        utils.EzPickle.__init__(self)

        # Must come after model init
        self.reset_model()

        # Create camera matrix for projection
        self.cam_matrix = self.get_cam_mat()

        # 0 position is straight up in xml file
        self.init_qpos[1] = np.pi
        self.init_qpos[0] = 1.00

    def initialize_geom_positions(self):
        # Flip a coin to decide whether cartpole starts pointing up or down, can be replaced by uniform 360deg init
        coin = self.np_random.randint(0, 2)
        if coin:
            self.init_qpos[1] = np.pi
        else:
            self.init_qpos[1] = 0.0

    def randomize_color(self):
        # Generate random index
        randidx = self.np_random.choice(a=self.ncolors)
        randjdx = self.np_random.choice(a=self.ncolors)
        if self.invisible_rods:
            rod_color = (0, 0, 0, 0)
        else:
            rod_color = self.color_options[randidx]
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('gpole1')] = rod_color
        # self.sim.model.geom_rgba[self.sim.model.geom_name2id('gpole2')] = (0, 0, 0, 0)
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('gpole2')] = rod_color
        mass_color = self.color_options[randjdx]
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('gmass')] = mass_color

    def step(self, action: float):
        action = np.clip(action, -1.0, 1.0)
        # Getting state before simulation step and observed frame after is a work around to 1-step delayed obs frame
        state = self._get_state()
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        # Terminate if the cart/slider joint goes out of the sim camera FOV
        out_of_view = np.abs(self.sim.data.qpos[0]) > 1.7
        # self.done is never set to True since there is no task being done
        done = out_of_view or self.done
        # dummy cost
        cost = 0.0
        return ob, -cost, done, {'success': self.done, 'state': state}

    # State is [x_cart, x_hinge2, y_hinge2, theta_hinge2 (xmass, ymass like in cartpole can be inferred for verifying
    # state visualization and at run-time), theta_dot_hinge1, theta_dot_hinge2, v_cart]
    # x_mass and y_mass here are the (x, y (or rather z)) of the contact between the mass and the rope
    # Sign of x components is flipped so that the sign is consistent with a std. x-y coordinate plane,
    # though technically we are looking at an x-z plane
    def _get_state(self):

        # Get current mass location in world frame
        cart_x = self.sim.data.qpos[0]
        # hinge2 xpos cart_x - l*sin(theta)
        hinge2_x = cart_x + self.pole1_len * np.sin(self.sim.data.qpos[1])
        # hinge1 ypos l*cos(theta), uses fact that ycart = 0.0
        hinge2_y = 0.0 + self.pole1_len * np.cos(self.sim.data.qpos[1])
        # Repeat above but for end-point of second pole and end of first pole acts as reference
        mass_x = hinge2_x + self.pole2_len * np.sin(self.sim.data.qpos[2] + self.sim.data.qpos[1])
        # hinge1 ypos l*cos(theta), uses fact that ycart = 0.0
        mass_y = hinge2_y + self.pole2_len * np.cos(self.sim.data.qpos[2] + self.sim.data.qpos[1])

        # Compute the prev versions of all these quantities using theta_dot
        #  Find current angular location of hinge1 (hinge connected to robot/cart)
        theta1_cur = self.sim.data.qpos[1]

        # Theta previous (wrap around taken care of by trig functions subsequently)
        # theta based angular velocity replaced with linear velocities in state representation
        theta1_prev = theta1_cur - self.sim.data.qvel[1] * self.dt
        # Location of cart in previous state
        prev_cart_x = cart_x - self.sim.data.qvel[0] * self.dt

        prev_hinge2_x = prev_cart_x + self.pole1_len * np.sin(theta1_prev)
        cart_y = 0.0
        prev_hinge2_y = cart_y + self.pole1_len * np.cos(theta1_prev)

        # Current angle for second hinge joint
        # theta2_cur = np.arctan2(mass_x - hinge2_x, mass_y - hinge2_y)
        theta2_cur = self.sim.data.qpos[2]

        # Previous angle for second hinge joint
        theta2_prev = theta2_cur - self.sim.data.qvel[2] * self.dt

        prev_mass_x = prev_hinge2_x + self.pole2_len * np.sin(theta1_prev + theta2_prev)
        prev_mass_y = prev_hinge2_y + self.pole2_len * np.cos(theta1_prev + theta2_prev)

        _st = np.array([cart_x,
                        # second hinge x and y positions
                        hinge2_x,
                        hinge2_y,
                        # mass x and y position
                        mass_x,
                        mass_y,
                        self.sim.data.qvel[0],
                        (hinge2_x - prev_hinge2_x) / self.dt,
                        (hinge2_y - prev_hinge2_y) / self.dt,
                        (mass_x - prev_mass_x) / self.dt,
                        (mass_y - prev_mass_y) / self.dt
                        ])

        # _st = np.concatenate([
        #     # cart x pos
        #     self.sim.data.qpos[:1],
        #     # second hinge x and y positions
        #     hinge2_x,
        #     hinge2_y,
        #     # mass x and y position
        #     mass_x,
        #     mass_y,
        #     # # theta_dot_hinge1, theta_dot_hinge2
        #     # np.clip(self.sim.data.qvel[1:], -20, 20),
        #     # # v_cart
        #     # -1 * np.clip(self.sim.data.qvel[:1], -20, 20)
        # ]).ravel()
        # Update state to Ball related quantities only if for ball
        if self.for_ball:
            # Get x_mass, y_mass and derivatives
            _st = self.get_ball_state(_st)
        return _st

    def get_ball_state(self, double_cartpole_st):
        # Get coordinates of cart assuming ycart = 0.0
        # cart_xy = np.array([double_cartpole_st[0], 0.0])
        # Get coordinates of first hinge
        hinge_xy = double_cartpole_st[1:3]
        # Distance between cart_xy and hinge_xy would be length, but we have direct access to length as
        #  class attribute so just use that
        # Get angle this hinge makes with vertical (Due to relative coordinate frame being used in first link/pole in
        #   kinematic chain being made to point downward via_initialize positions during every reset)
        theta_h2 = double_cartpole_st[3]
        mass_xy = np.array([hinge_xy[0] + self.plength_cur * np.sin(theta_h2),
                            hinge_xy[1] - self.plength_cur * np.cos(theta_h2)])
        # First find mass_vel_xy in hinge2 frame: wR sin theta, wR cos theta
        mass_vel_xy_h2f = double_cartpole_st[5] * self.plength_cur * np.array([np.cos(theta_h2), np.sin(theta_h2)])
        # Next find hinge2_vel_xy in hinge1 frame
        # Get angle hinge1 makes with downward vertical
        theta_h1 = np.arctan2(hinge_xy[0], hinge_xy[1])
        hinge2_vel_xy_h1f = double_cartpole_st[4] * self.plength_cur * np.array([np.cos(theta_h1), np.sin(theta_h1)])
        # Finally, obtain mass_vel_xy in ground frame mass_g = mass_h2 + h2_h1 + h1_g(v_cart)
        mass_vel_xy = mass_vel_xy_h2f + hinge2_vel_xy_h1f + double_cartpole_st[6]
        return mass_xy, mass_vel_xy

    def reset(self):
        self.done = False
        self.initialize_geom_positions()
        if self.randomize:
            # Geometry is randomized via external dm control based script
            self.randomize_color()
        return self.reset_model()

    def reset_model(self):
        # Set joint angles and velocities [qpos, qvel] in a noisy manner
        self.set_state(
            np.concatenate((# Cart position
                            self.init_qpos[0] + self.np_random.uniform(low=-1.5, high=1.5, size=1),
                            # Hinge1 Angle
                            self.init_qpos[1] + self.np_random.uniform(low=0.25 * -np.pi, high=0.25 * np.pi, size=1),
                            # Hinge2 Angle
                            self.init_qpos[2] + self.np_random.uniform(low=0.25 * -np.pi, high=0.25 * np.pi, size=1))),
                            # All 3 velocities initialized randomly
                            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()
