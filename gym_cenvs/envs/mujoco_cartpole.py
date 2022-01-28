import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class MujocoCartPoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_path = os.path.abspath('gym_cenvs/assets/cartpole.xml')

        # - - - - - Init Attributes Prior since some needed for model creation - - - - - -
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
        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip=1)
        utils.EzPickle.__init__(self)
        # Must come after model init
        self.reset_model()
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
        # Local position of center of mass of the pole in cart (parent) frame
        self.sim.model.body_ipos[self.sim.model.body_name2id('pole')] = [0.0, 0.0, self.psemilength + pole_dlength]
        # Position offset wrt parent body
        self.sim.model.body_pos[self.sim.model.body_name2id('mass')] = [0.0, 0.0, 2 * (self.psemilength + pole_dlength)]
        # Update length for state calculation
        self.plength_cur = 2 * (self.psemilength + pole_dlength)

    def step(self, action: float):
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        state = self._get_state()
        # Also terminate if the cart/slider joint goes out of the sim camera FOV
        out_of_view = np.abs(self.sim.data.qpos[0]) > 1.7 # Earlier tried 2.5
        # self.done is never set to True since there is no task
        done = out_of_view or self.done
        # dummy cost
        cost = 0.0
        return ob, -cost, done, {'success': self.done, 'state': state}

    def _get_obs(self):
        return self.render(mode='rgb_array', width=64, height=64, camera_id=0)

    # State is [x_cart, x_mass, y_mass, v_cart, theta_dot_mass]
    # x_mass and y_mass here are the (x, y (or rather z)) of the contact between the mass and the rope
    # Sign of x components is flipped so that the sign is consistent with a std. x-y coordinate plane,
    # though technically an x-z plane
    def _get_state(self):
        _st = np.concatenate([
            -self.sim.data.qpos[:1],  # cart x pos
            -self.sim.data.qpos[0] + self.plength_cur * np.sin(self.sim.data.qpos[1:] - np.pi), # l*sin(theta) + cart_x
            -self.plength_cur * np.cos(self.sim.data.qpos[1:] - np.pi), # l*cos(theta)
            np.clip(self.sim.data.qvel, -20, 20) * np.array([-1, 1]) # v_cart, theta_dot_mass
        ]).ravel()
        return _st

    def reset(self):
        self.done = False
        if self.randomize:
            self.randomize_geometry()
        return self.reset_model()

    def reset_model(self):
        # Set joint angles and velocities
        self.set_state(
            np.concatenate((self.init_qpos[0] + self.np_random.uniform(low=-1.5, high=1.5, size=1),
                            self.init_qpos[1] + self.np_random.uniform(low=0.25 * -np.pi, high=0.25 * np.pi, size=1))),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    # To be kept the same across simulated environments
    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
