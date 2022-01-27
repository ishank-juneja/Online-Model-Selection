import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py.generated import const


class MujocoCartPoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, fixed_environment=False):
        xml_path = os.path.join(os.path.dirname(__file__), '/home/ishank/Desktop/MM-LVSPC/gym_cenvs/assets/cartpole.xml')
        self.goal_x = 0
        self.goal_y = 0
        self.length = 1.0
        self.done = False
        print(xml_path)
        mujoco_env.MujocoEnv.__init__(self, xml_path, 1)
        utils.EzPickle.__init__(self)

        self.randomize = not fixed_environment
        self.reset_model()
        self.init_qpos[1] += np.pi
        self.init_qpos[0] += 1.00

    def randomize_geometry(self):
        # zero state
        self.set_state(
            self.init_qpos,
            self.init_qvel
        )
        # Randomize changes
        pole_dlength = self.np_random.uniform(low=-0.2, high=0.25)
        pole_dwidth = self.np_random.uniform(low=-0.05, high=0.05)
        mass_pole_diff = (pole_dwidth + 0.1) - 0.2
        mass_dradius = self.np_random.uniform(low=mass_pole_diff + 0.05, high=0.0)

        self.sim.model.geom_size[self.sim.model.geom_name2id('gpole')] = [0.1 + pole_dwidth, 0.5 + pole_dlength, 0.0]
        self.sim.model.geom_size[self.sim.model.geom_name2id('gmass')] = [0.2 + mass_dradius, 0.2 + mass_dradius, 0.]

        # Update positions
        self.sim.model.body_ipos[self.sim.model.body_name2id('pole')] = [0.0, 0.0, 0.5 + pole_dlength]
        self.sim.model.body_pos[self.sim.model.body_name2id('mass')] = [0.0, 0.0, 1 + 2 * pole_dlength]

        # Update length for state
        self.length = 1.0 + 2 * pole_dlength

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # action = 0.0
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        state = self._get_state()
        goal_cost, centre_cost = self.get_cost()
        link_collide = self.collision_check()

        self.done = self.done or (link_collide == 1)
        out_of_scope = (np.abs(self.sim.data.qpos[0]) > 2.5)
        done = out_of_scope or self.done or link_collide
        fail_cost = 100.0 if out_of_scope or (link_collide == 2) else 0.0
        return ob, -(goal_cost + centre_cost + fail_cost), done, {'success': self.done, 'state': state}

    def collision_check(self):
        # 0 for no collision
        # 1 for successful collision (target and mass)
        # 2 for unsuccessful collision (target and pole)
        success = 0
        fail = 0
        for ncon in range(self.data.ncon):
            con = self.sim.data.contact[ncon]
            g1 = self.sim.model.geom_id2name(con.geom1)
            g2 = self.sim.model.geom_id2name(con.geom2)

            geom_concat = g1 + g2

            if "target" in geom_concat:
                if "mass" in geom_concat:
                    success = 1
                if ("pole" in geom_concat) or ("cart" in geom_concat):
                    fail = 1

        if success:
            return 1
        if fail:
            return 2
        return 0

    def get_cost(self):
        x, y = self._get_state()[1:3]
        # Penalty to goal
        goal_cost = (x - self.goal_x)**2 + (y - self.goal_y) ** 2
        xx = self.sim.data.qpos[0]
        centre_cost = np.clip(np.abs(xx) - 1.0, 0.0, None)
        return goal_cost, centre_cost

    def get_goal(self):
        return [self.goal_x, self.goal_y]

    def _get_obs(self):
        return self.render(mode='rgb_array', width=64, height=64, camera_id=0)

    def _get_state(self):
        # State is [x_cart, x_mass, y_mass, v_cart, theta_dot_mass]
        return np.concatenate([
            -self.sim.data.qpos[:1],  # cart x pos
            -self.sim.data.qpos[0] + self.length * np.sin(self.sim.data.qpos[1:] - np.pi),
            -self.length * np.cos(self.sim.data.qpos[1:] - np.pi),
            np.clip(self.sim.data.qvel, -20, 20) * np.array([-1, 1]),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset(self):
        self.done = False
        if self.randomize:
            self.randomize_geometry()
        return self.reset_model()

    def reset_model(self):
        self.set_state(
            np.concatenate((self.init_qpos[0] + self.np_random.uniform(low=-1.5, high=1.5, size=1),
                            self.init_qpos[1] + self.np_random.uniform(low=0.25 * -np.pi, high=0.25 * np.pi, size=1))),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        initial_cost = 0.0
        # Now want the problem to be non trivial - can't start at goal, so will just
        # Rejection sample goal
        while initial_cost < .75:
            self.goal_x = np.random.uniform(low=0.0, high=1.5)
            self.goal_y = np.random.uniform(low=-1.0, high=.1)
            initial_cost, _ = self.get_cost()

        # self.sim.model.body_pos[self.sim.model.body_name2id('target')] = [-self.goal_x, 0, self.goal_y]
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
