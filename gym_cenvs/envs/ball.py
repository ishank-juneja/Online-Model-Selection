import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
from gym_cenvs.envs.base import MujocoBase


class Ball(mujoco_env.MujocoEnv, utils.EzPickle, MujocoBase):
    def __init__(self):
        xml_path = os.path.abspath('gym_cenvs/assets/ball.xml')

        # - - - - - Init Attributes Prior since some needed for model creation - - - - - -
        # Episode/task of a contiguous trajectory complete?
        self.done = False
        # Fixed initial ball radius
        self.bradius = 0.2
        self.bradius_max_add = 0.125
        self.bradius_max_sub = 0.085
        # Domain rand over geometry and color
        self.randomize = True
        # - - - - - - - - - - - - - - - -
        MujocoBase.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip=1)
        utils.EzPickle.__init__(self)
        # Must come after model init
        self.reset_model()
        # Create camera matrix for projection
        self.cam_matrix = self.get_cam_mat()

    def randomize_geometry(self):
        # zero state
        self.set_state(self.init_qpos, self.init_qvel)
        # Randomize changes via uniform delta perturbations
        ball_dradius = self.np_random.uniform(low=-self.bradius_max_sub, high=self.bradius_max_add)
        self.sim.model.geom_size[self.sim.model.geom_name2id('gball')] = [self.bradius + ball_dradius,
                                                                          self.bradius + ball_dradius, 0.0]

    def randomize_color(self):
        randidx = self.np_random.choice(a=self.ncolors)
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('gball')] = self.color_options[randidx]

    def step(self, action: float):
        # Getting state before simulation step and observed frame after is a work around to 1-step delayed obs frame
        state = self._get_state()
        # Unactuated freely falling ball model
        action = 0.0
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        # Terminate if the ball goes out of view
        out_of_view_x = np.abs(self.sim.data.qpos[0]) > 1.7 # Earlier tried 2.5
        # On sided ineq since always falls down
        out_of_view_z = self.sim.data.qpos[2] < -1.7
        out_of_view = out_of_view_x or out_of_view_z
        # self.done is never set to True since there is no task
        done = out_of_view or self.done
        # dummy cost
        cost = 0.0
        return ob, -cost, done, {'success': self.done, 'state': state}

    # State is [x_ball, z_ball]
    def _get_state(self):
        # x_ball, z_ball, Give perception coordinates and velocities
        _st = np.hstack((self.sim.data.qpos[0], self.sim.data.qpos[2], self.sim.data.qvel[0], self.sim.data.qvel[2]))
        return _st

    def reset(self):
        self.done = False
        if self.randomize:
            self.randomize_geometry()
            self.randomize_color()
        return self.reset_model()

    def reset_model(self):
        # No variation in y-position (depth)
        ball_x = self.np_random.uniform(low=-1.0, high=1.0)
        ball_y = 0.0
        ball_z = self.np_random.uniform(low=-1.0, high=1.0)
        ball_xyz = np.array([ball_x, ball_y, ball_z])
        # Sphere orientation does not matter
        ball_quat = np.hstack((1.0, np.zeros(3, dtype=np.float64)))
        ball_free_jnt_state = np.hstack((ball_xyz, ball_quat))
        # Reset ball velocity randomly in (x, y) dir and 0 for z and rotational
        ball_vx = self.np_random.uniform(low=-5.0, high=5.0)
        ball_vy = 0.0
        ball_vz = self.np_random.uniform(low=-1.0, high=1.0)
        ball_vxyz = np.array([ball_vx, ball_vy, ball_vz])
        # Set ball free joint velocity (aka ball velocity) with angular terms = 0
        ball_free_jnt_vel = np.hstack((ball_vxyz, np.zeros(3, dtype=np.float64)))
        self.set_state(ball_free_jnt_state, ball_free_jnt_vel)
        return self._get_obs()


