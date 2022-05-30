from gym import utils
from gym_cenvs.envs.base import MujocoBase
from gym.envs.mujoco import mujoco_env
import numpy as np
import os


class Catching(mujoco_env.MujocoEnv, utils.EzPickle, MujocoBase):
    def __init__(self):
        self.done = False
        xml_path = os.path.abspath('gym_cenvs/assets/catching.xml')
        MujocoBase.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path, 50)
        utils.EzPickle.__init__(self)
        # Create camera matrix for projection
        self.cam_matrix = self.get_cam_mat()
        # Init all joint positions and velocities to 0
        self.init_qpos[:] = 0.0
        self.init_qvel[:] = 0.0
        # Must come after model init
        self.reset_model()

    def step(self, action):
        # Clip action to respect force actuator control range
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        # Here both state and observation are being taken after the simulation step
        #  this is different from the simple model environments where I take state before and observation after so
        #  that they match, reason was that the observed frame was coming out delayed by about 1 time-step
        #  in those environments.
        #  Presumably this "off by 1" problem is present here as well, but it is hardly relevant here since
        #  the physics simulation time-scale is a lot shorter (50x via self.frameskip=50) than the simple model
        #  environments and the observation will basically be the same as the one that corresponds to the state
        state = self._get_state()
        ob = self._get_obs()
        # True environmental cost function (to act as reward for RL environment)
        goal_cost, centre_cost = self.get_cost()

        # Receive collision type if any
        collision_type = self.collision_check()

        print("Collision Type is {0}".format(collision_type))

        # Override to get rid of error
        collision_type = 0

        # type=1 is success
        self.done = self.done or (collision_type == 1)
        # Going outside of view
        out_of_view = (np.abs(self.sim.data.qpos[0]) > 1.7)
        # Need to reset the model if there was a type-2 failure collision or went out of view
        done = out_of_view or self.done
        # Considered failure if either of these happen
        fail_cost = 100.0 if out_of_view else 0.0

        if collision_type == 1:
            print('Task Successful')
        if out_of_view:
            print('Failed, went offscreen')

        # Note: Reward here was changed for the RL envs
        return ob, 0.0, done, {'state': state, 'success': self.done}

    def collision_check(self):
        """
        :return:
        0 for no collision
        1 for successful collision (cup base and ball)
        """
        success = False
        collision_pairs = []
        for ncon in range(self.data.ncon):
            con = self.sim.data.contact[ncon]
            g1 = self.sim.model.geom_id2name(con.geom1)
            g2 = self.sim.model.geom_id2name(con.geom2)

            collision_pair = g1 + '_' + g2
            collision_pairs.append(collision_pair)

            if "cup_collision_site" in collision_pair:
                if "gball" in collision_pair:
                    success = True

        print(collision_pairs)

        if success:
            return 1
        else:
            return 0

    def get_cost(self):
        # Get x and y coordinate of the site defined at cup base center
        xcup, _, ycup = self.sim.data.site_xpos[0]
        # Get x and y coordinate of the site defined at ball center
        xball, _, yball = self.sim.data.site_xpos[1]

        # Cost is distance between ball and cup
        goal_cost = (xcup + xball)**2 + (ycup - yball) ** 2

        # Cost on having to move cart?
        cartx = self.sim.data.qpos[0]
        centre_cost = np.clip(np.abs(cartx) - 1.0, 0.0, None)

        return goal_cost, centre_cost

    def _get_obs(self):
        size_ = self.seg_config.imsize
        return self.render(mode='rgb_array', width=size_, height=size_, camera_id=0)

    def _get_state(self):
        # Old Tom version
        custom_state = np.concatenate([
            self.sim.data.qpos,  # cart x pos
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).ravel()
        # Modified to return ball/mass coordinates
        # custom_state = np.concatenate([
        #     self.sim.data.qpos,  # cart x pos
        #     -1 * self.get_body_com('conker')[:1],
        #     self.get_body_com('conker')[2:]
        # ]).ravel()
        return custom_state

    def reset(self):
        self.done = False
        return self.reset_model()

    def reset_model(self):
        # Randomize cup position along x-axis
        cup_x = self.np_random.uniform(low=-1.7, high=1.7)

        # Randomize initial cup velocity
        cup_vx = self.np_random.uniform(low=-5.0, high=5.0)

        # Randomize ball x and y coordinates (y coordinate in viewing plane is z coordinate in space)
        #  Ball free joint quat. has no significance due to spherical symmetry
        ball_x = self.np_random.uniform(low=-1.0, high=1.0)
        ball_y = 0.0
        ball_z = self.np_random.uniform(low=1.0, high=1.7)
        ball_xyz = np.array([ball_x, ball_y, ball_z])
        # Sphere orientation does not matter
        ball_quat = np.hstack((1.0, np.zeros(3, dtype=np.float64)))

        # Reset ball velocity randomly in (x, y) dir and 0 for z and rotational
        ball_vx = self.np_random.uniform(low=-5.0, high=5.0)
        ball_vy = 0.0
        ball_vz = self.np_random.uniform(low=-1.0, high=1.0)
        ball_vxyz = np.array([ball_vx, ball_vy, ball_vz])
        # Set ball free joint velocity (aka ball velocity) with angular terms = 0
        ball_angular_speed = np.zeros(3, dtype=np.float64)

        # Concatenate together in the order joint appear in the XML
        my_qpos = np.hstack((cup_x, ball_xyz, ball_quat))
        my_qvel = np.hstack((cup_vx, ball_vxyz, ball_angular_speed))

        self.set_state(qpos=my_qpos, qvel=my_qvel)

        # Return the observation at which handing over env for stepping
        return self._get_obs()
