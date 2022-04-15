import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym_cenvs.envs.base import MujocoBase


class Cartpole(mujoco_env.MujocoEnv, utils.EzPickle, MujocoBase):
    def __init__(self, for_ball: bool = False, invisible_rod: bool = False):
        xml_path = os.path.abspath('gym_cenvs/assets/cartpole.xml')

        # - - - - - Init Attributes Prior since some needed for model creation - - - - - -
        self.for_ball = for_ball
        self.invisible_rod = invisible_rod
        # Episode/task of a contiguous trajectory complete?
        self.done = False
        # Fixed initial plen/2
        self.psemilength = 0.5
        # Updated based on currently being used
        self.plength_cur = 1.0
        self.pwidth = 0.1
        self.mradius = 0.2
        self.randomize = True
        # Maximum perturbations to semi length
        self.plength_max_sub = 0.2
        self.plength_max_add = 0.25
        self.pwidth_maxd = 0.05
        # - - - - - - - - - - - - - - - -
        MujocoBase.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip=1)
        utils.EzPickle.__init__(self)
        # Must come after model init
        self.reset_model()
        # Create camera matrix for projection
        self.cam_matrix = self.get_cam_mat()
        # 0 position is straight up in xml file
        self.init_qpos[1] += np.pi
        self.init_qpos[0] += 1.00

    def randomize_geometry(self):
        # zero state
        self.set_state(self.init_qpos, self.init_qvel)
        # Randomize changes via uniform delta perturbations
        pole_dlength = self.np_random.uniform(low=-self.plength_max_sub, high=self.plength_max_add)
        pole_dwidth = self.np_random.uniform(low=-self.pwidth_maxd, high=self.pwidth_maxd)
        # Make sure mass radius never goes below pole width
        mass_pole_diff = (pole_dwidth + self.pwidth) - self.mradius
        # pick low < 0.0, and mass only gets smaller
        mass_dradius = self.np_random.uniform(low=mass_pole_diff + self.pwidth_maxd, high=0.0)
        # Overwrite default 2D geom sizes from the .xml file
        self.sim.model.geom_size[self.sim.model.geom_name2id('gpole')] = [self.pwidth + pole_dwidth, self.psemilength + pole_dlength, 0.0]
        self.sim.model.geom_size[self.sim.model.geom_name2id('gmass')] = [self.mradius + mass_dradius, self.mradius + mass_dradius, 0.0]
        # self.sim.model.geom_size[self.sim.model.geom_name2id('gcart')] = [0.1, 0.5, 0.0]
        # Local position of center of mass of the pole in cart (parent) frame
        self.sim.model.body_ipos[self.sim.model.body_name2id('pole')] = [0.0, 0.0, self.psemilength + pole_dlength]
        # Position offset wrt parent body
        self.sim.model.body_pos[self.sim.model.body_name2id('mass')] = [0.0, 0.0, 2 * (self.psemilength + pole_dlength)]
        # Update length for state calculation
        self.plength_cur = 2 * (self.psemilength + pole_dlength)

    def randomize_color(self):
        # Generate random index
        randidx = self.np_random.choice(a=self.ncolors)
        randjdx = self.np_random.choice(a=self.ncolors)
        if self.invisible_rod:
            rod_color = (0, 0, 0, 0)
        else:
            rod_color = self.color_options[randidx]
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('gpole')] = rod_color
        mass_color = self.color_options[randjdx]
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('gmass')] = mass_color
        # self.model.geom_rgba[self.sim.model.geom_name2id('gcart')] = (1, 0, 1, 1)

    def step(self, action: float):
        action = np.clip(action, -1.0, 1.0)
        # Getting state before simulation step and observed frame after is a work around to 1-step delayed obs frame
        state = self._get_state()
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        # Also terminate if the cart/slider joint goes out of the sim camera FOV
        out_of_view = np.abs(self.sim.data.qpos[0]) > 1.7 # Earlier tried 2.5
        # self.done is never set to True since there is no task
        done = out_of_view or self.done
        # dummy cost
        cost = 0.0
        return ob, -cost, done, {'success': self.done, 'state': state}

    # State is [x_cart, x_mass, y_mass, v_cart, theta_dot_mass]
    # x_mass and y_mass here are the (x, y (or rather z)) of the contact between the mass and the rope
    # Sign of x components is flipped so that the sign is consistent with a std. x-y coordinate plane,
    # though technically an x-z plane
    def _get_state(self):
        # Get current mass location in world frame
        cart_x = self.sim.data.qpos[0]
        mass_x = cart_x + self.plength_cur * np.sin(self.sim.data.qpos[1])
        cart_y = 0.0
        mass_y = cart_y + self.plength_cur * np.cos(self.sim.data.qpos[1])

        # - - - - - - - - - - -
        # # Find current angular location
        # theta_cur = np.arctan2(mass_x - cart_x, mass_y)
        theta_cur = self.sim.data.qpos[1]

        # Theta previous (wrap around taken care of by trig functions subsequently)
        # theta based angular velocity replaced with linear velocities in state representation
        theta_prev = theta_cur - self.sim.data.qvel[1] * self.dt
        # Location of cart in previous state
        prev_cart_x = cart_x - self.sim.data.qvel[0] * self.dt

        prev_mass_x = prev_cart_x + self.plength_cur * np.sin(theta_prev)
        cart_y = 0.0
        prev_mass_y = cart_y + self.plength_cur * np.cos(theta_prev)

        # We manually compute velocity this way since using tip speed with R*\dot{\theta} was giving problems
        # # - - - - - - - - - - -
        # Compute speed of end of pole in cartpole
        # cp_tip_speed = self.sim.data.qvel[1] * self.plength_cur
        # cpx = cp_tip_speed * np.sin(self.sim.data.qpos[1]) + self.sim.data.qvel[0]
        # cpx = cp_tip_speed * np.cos(self.sim.data.qpos[1])
        # cpy = cp_tip_speed * np.cos(self.sim.data.qpos[1])
        # Note: Making theta_dot part of state is avoided since the NN struggles to learn theta_dot
        vel_mass_x = (mass_x - prev_mass_x) / self.dt
        vel_mass_y = (mass_y - prev_mass_y) / self.dt

        if not self.for_ball:
            _st = np.array([
                self.sim.data.qpos[0],  # cart x pos
                self.sim.data.qpos[0] + self.plength_cur * np.sin(self.sim.data.qpos[1]),   # l*sin(theta) + cart_x
                0.0 + self.plength_cur * np.cos(self.sim.data.qpos[1]), # l*cos(theta)
                np.clip(self.sim.data.qvel[0], -20, 20),    # v_cart
                np.clip(vel_mass_x, -20, 20),
                np.clip(vel_mass_y, -20, 20)
            ])
        else:
            _st = np.concatenate([
                -self.sim.data.qpos[0] + self.plength_cur * np.sin(self.sim.data.qpos[1:] - np.pi),
                # l * sin(theta) + cart_x
                -self.plength_cur * np.cos(self.sim.data.qpos[1:] - np.pi)  # l*cos(theta)
            ]).ravel()
        return _st

    def _get_obs(self):
        size_ = self.seg_config.imsize
        return self.render(mode='rgb_array', width=size_, height=size_, camera_id=0)

    def reset(self):
        self.done = False
        if self.randomize:
            self.randomize_geometry()
            self.randomize_color()
        return self.reset_model()

    def reset_model(self):
        # Flip a coin to decide whether cartpole starts pointing up or down, can be replaced by uniform 360deg init
        coin = self.np_random.randint(0, 2)
        if coin:
            self.init_qpos[1] = np.pi
        else:
            self.init_qpos[1] = 0.0
        # Set joint angles and velocities
        self.set_state(
            np.concatenate((self.init_qpos[0] + self.np_random.uniform(low=-1.5, high=1.5, size=1),
                            self.init_qpos[1] + self.np_random.uniform(low=0.25 * -np.pi, high=0.25 * np.pi, size=1))),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()
