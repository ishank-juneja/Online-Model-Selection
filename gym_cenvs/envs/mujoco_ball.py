import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os


class MujocoBall(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_path = os.path.abspath('gym_cenvs/assets/ball.xml')

        # - - - - - Init Attributes Prior since some needed for model creation - - - - - -
        # Episode/task of a contiguous trajectory complete?
        self.done = False
        # Fixed initial ball radius
        self.bradius = 0.2
        self.bradius_max_add = 0.3
        self.bradius_max_sub = 0.15
        # Domain rand over geometry
        self.randomize = True
        # - - - - - - - - - - - - - - - -
        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip=1)
        utils.EzPickle.__init__(self)
        # Must come after model init
        self.reset_model()

    def randomize_geometry(self):
        # zero state
        self.set_state(self.init_qpos, self.init_qvel)
        # Randomize changes via uniform delta perturbations
        ball_dradius = self.np_random.uniform(low=-self.bradius_max_sub, high=self.bradius_max_add)
        self.sim.model.geom_size[self.sim.model.geom_name2id('gball')] = [self.bradius + ball_dradius,
                                                                          self.bradius + ball_dradius, 0.0]

    def step(self, action: float):
        # Unactuated freely falling ball model
        action = 0.0
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        state = self._get_state()
        # Terminate if the ball goes out of view
        # TODO: tune x and y components for out_of_view seperately
        out_of_view = np.min(np.abs(self.sim.data.qpos[:2])) > 1.7 # Earlier tried 2.5
        # self.done is never set to True since there is no task
        done = out_of_view or self.done
        # dummy cost
        cost = 0.0
        return ob, -cost, done, {'success': self.done, 'state': state}

    def _get_obs(self):
        return self.render(mode='rgb_array', width=64, height=64, camera_id=0)

    # State is [x_cart, x_mass, y_mass, v_cart, theta_dot_mass]
    def _get_state(self):
        _st = self.sim.data.qpos[:2]    # x_ball, y_ball
        return _st

    def reset(self):
        self.done = False
        if self.randomize:
            self.randomize_geometry()
        return self.reset_model()

    def reset_model(self):
        # No variation in z-position
        ball_xyz = np.hstack((self.np_random.uniform(low=-1.0, high=1.0, size=1),
                              self.np_random.uniform(low=-0.2, high=0.2, size=1), 0.0))
        ball_quat = np.hstack((1.0, np.zeros(3, dtype=np.float64)))
        ball_free_jnt_state = np.hstack((ball_xyz, ball_quat))
        # Reset ball velocity randomly in (x, y) dir and 0 for z and rotational
        ball_free_jnt_vel = np.hstack((self.np_random.uniform(low=-1.0, high=1.0, size=2),
                                       np.zeros(4, dtype=np.float64)))
        self.set_state(ball_free_jnt_state, ball_free_jnt_vel)
        return self._get_obs()

    # To be kept the same across simulated environments
    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
