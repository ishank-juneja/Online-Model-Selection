import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
from mujoco_py.generated import const


# TODO: document similar to cartpole
class ConkersEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, transparent_rope=False):
        self.goal_x = 0
        self.goal_y = 0
        self.done = False
        # Number of links in chain link approximation of rope
        self.nlinks = 10
        xml_path = os.path.abspath('gym_cenvs/assets/conkers.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 50)
        utils.EzPickle.__init__(self)
        self.reset_model()
        self.init_qpos[1] += np.pi
        self.init_qpos[0] = 1.0
        # Make rope transparent for rope free ball dataset
        if transparent_rope:
            # Set the color of each geom associated with rope link to transparent
            for idx in range(self.nlinks):
                gname = "gpole{0}".format(idx + 1)
                self.model.geom_rgba[self.sim.model.geom_name2id(gname)] = (1, 1, 1, 0)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        state = self._get_state()
        ob = self._get_obs()
        goal_cost, centre_cost = self.get_cost()
        collision = self.collision_check()
        #collision = 1 if goal_cost < 1e-1 else 0

        self.done = self.done or (collision == 1)
        out_of_scope = (np.abs(self.sim.data.qpos[0]) > 1.7)
        done = out_of_scope or self.done or (collision == 2)
        fail_cost = 100.0 if out_of_scope or (collision == 2) else 0.0

        if collision == 2:
            print('Failed due to collision')
        if out_of_scope:
            print('Failed, went offscreen')
        return ob, -(goal_cost + centre_cost + fail_cost), done, {'state': state, 'success': self.done}

    def _get_obs(self):
        return self.render(mode='rgb_array', width=64, height=64, camera_id=0)

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
        x, _, y = self.sim.data.site_xpos[0]
        # Penalty to goal
        goal_cost = (x + self.goal_x)**2 + (y - self.goal_y) ** 2
        xx = self.sim.data.qpos[0]
        centre_cost = np.clip(np.abs(xx) - 1.0, 0.0, None)
        return goal_cost, centre_cost

    def get_goal(self):
        return [self.goal_x, self.goal_y]

    def _get_state(self):
        # Old Tom version
        # custom_state = np.concatenate([
        #     self.sim.data.qpos,  # cart x pos
        #     np.clip(self.sim.data.qvel, -10, 10),
        #     np.clip(self.sim.data.qfrc_constraint, -10, 10)
        # ]).ravel()
        # Modified to return state of interest
        # Keep returning cart state needed to rejection sample conkers initializations too close to the goal
        # In addition return the sign flipped (for consistency) x cooord
        # and the z coord (y coord in our view) of the conker
        custom_state = np.concatenate([
            self.sim.data.qpos,  # cart x pos
            -1 * self.get_body_com('conker')[:1],
            self.get_body_com('conker')[2:]
        ]).ravel()
        return custom_state

    def reset(self):
        self.done = False
        return self.reset_model()

    def reset_model(self):
        rand_mask = np.zeros(self.model.nq)
        rand_mask[1] = 1
        # self.do_simulation([0], self.frame_skip)
        initial_collision = 2
        # Now want the problem to be non trivial - can't start at goal, so will just
        # Rejection sample goal
        while initial_collision:
            self.goal_x = np.random.uniform(low=0.0, high=1.5)
            self.goal_y = np.random.uniform(low=-1, high=.1)
            self.set_state(
                self.init_qpos + rand_mask * self.np_random.uniform(low=-0.25 * np.pi, high=0.25 * np.pi,
                                                                    size=self.model.nq),
                self.init_qvel + rand_mask * self.np_random.randn(self.model.nv) * .1
            )

            self.sim.model.body_pos[self.sim.model.body_name2id('target')] = [-self.goal_x, 0, self.goal_y]
            self.do_simulation([0], self.frame_skip)
            initial_collision = self.collision_check()
            d, _ = self.get_cost()
            # Check goal not too close to base
            goal = self.get_goal()
            base_x = self._get_state()[0]
            base_2_goal = np.linalg.norm(np.array([base_x, 0.0]) - np.asarray(goal))
            initial_collision = initial_collision or d < 1.0 or base_2_goal < 0.25

        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
