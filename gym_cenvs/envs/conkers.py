from gym import utils
from gym_cenvs.envs.base import MujocoBase
from gym.envs.mujoco import mujoco_env
import numpy as np
import os


class Conkers(mujoco_env.MujocoEnv, utils.EzPickle, MujocoBase):
    def __init__(self, transparent_rope=False):
        # Vars to hold goal location while playing conkers game (randomized via rest)
        self.goal_x = 0
        self.goal_y = 0
        # Episode terminated due to either failure or task completion
        self.done = False
        # Number of links in chain link approximation of rope
        self.nlinks = 10
        xml_path = os.path.abspath('gym_cenvs/assets/conkers.xml')
        MujocoBase.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip=50)
        utils.EzPickle.__init__(self)
        # Create camera matrix for projection
        self.cam_matrix = self.get_cam_mat()
        # The body/joint/ref parameter (joint reference position) doesn't seem to work for hinge joint
        #  specify += np.pi here so that it points downwards
        self.init_qpos[1] = np.pi
        self.init_qpos[0] = -1.0
        # Reset in preperation for new episode
        self.reset_model()
        # Make rope transparent
        if transparent_rope:
            # Set the color of each geom associated with rope link to transparent
            for idx in range(self.nlinks):
                gname = "gpole{0}".format(idx + 1)
                self.model.geom_rgba[self.sim.model.geom_name2id(gname)] = (1, 1, 1, 0)

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

        # type=1 is success
        self.done = self.done or (collision_type == 1)
        # Going outside of view
        out_of_view = (np.abs(self.sim.data.qpos[0]) > 1.7)
        # Need to reset the model if there was a type-2 failure collision or went out of view
        done = out_of_view or self.done or (collision_type == 2)
        # Considered failure if either of these happen
        fail_cost = 100.0 if out_of_view or (collision_type == 2) else 0.0

        #
        if collision_type == 1:
            print('Task Successful')
        if collision_type == 2:
            print('Failed due to collision')
        if out_of_view:
            print('Failed, went offscreen')

        # Note: Reward here was changed for the RL envs
        return ob, -(goal_cost + centre_cost + fail_cost), done, {'state': state, 'success': self.done}

    def collision_check(self):
        """
        :return:
        0 for no collision
        1 for successful collision (target and mass)
        2 for unsuccessful collision (target and pole)
        """
        success = False
        fail = False
        # Iterate over all the detected contact pairs held in simulator data
        for ncon in range(self.data.ncon):
            # Extract contact grouping
            con = self.sim.data.contact[ncon]
            # Extract the name of geom1 involved in contact
            g1 = self.sim.model.geom_id2name(con.geom1)
            # Extract the name of geom2 involved in contact
            g2 = self.sim.model.geom_id2name(con.geom2)
            # Search for contacts of our interest in the pair
            geom_concat = g1 + g2
            # Target location rigidly attached to world-body
            if "target" in geom_concat:
                # A target-mass collision is task aim
                if "mass" in geom_concat:
                    success = True
                # Return task failure if *any* pole segment or the cart comes in contact with the target site
                elif ("pole" in geom_concat) or ("cart" in geom_concat):
                    fail = True
                else:
                    # Collision of target with something else
                    pass
        if success:
            return 1
        elif fail:
            return 2
        else:
            return 0

    def get_cost(self):
        # Get x and y coordinate of the site defined at ball center
        x, _, y = self.sim.data.site_xpos[0]
        # Penalty to goal
        goal_cost = (x - self.goal_x)**2 + (y - self.goal_y)**2

        # penalty on how much cart_x position changed from initialization
        cartx = self.sim.data.qpos[0]
        centre_cost = np.clip(np.abs(cartx) - 1.0, 0.0, None)

        return goal_cost, centre_cost

    def get_goal(self):
        return [self.goal_x, self.goal_y]

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
        # Only randomize 2nd joint

        rand_mask = np.zeros(self.model.nq)
        rand_mask[1] = 1

        resample_env = True
        # Now want the problem to be non trivial - can't start at goal, so will just
        # Rejection sample goal
        while resample_env:
            # Uniform randomly set goal location to be reachable points in lower RHP
            self.goal_x = np.random.uniform(low=0.0, high=1.5)
            self.goal_y = np.random.uniform(low=-1.0, high=0.1)

            # Initialize state with small random perturbation
            self.set_state(
                self.init_qpos + rand_mask * self.np_random.uniform(low=-0.25 * np.pi, high=0.25 * np.pi,
                                                                    size=self.model.nq),
                self.init_qvel + rand_mask * self.np_random.randn(self.model.nv) * .1
            )
            # Set target location
            self.sim.model.body_pos[self.sim.model.body_name2id('target')] = [self.goal_x, 0, self.goal_y]
            # Take 1 step to see if starting in collision
            self.do_simulation([0], self.frame_skip)
            initial_collision = self.collision_check()

            # Check goal not to close to start pos of conker
            d, _ = self.get_cost()
            conker_near_goal = True if d < 1.0 else False

            # Check if goal not too close to init cart position
            goal = self.get_goal()
            base_x = self._get_state()[0]
            base_2_goal = np.linalg.norm(np.array([base_x, 0.0]) - np.asarray(goal))
            cart_near_goal = True if base_2_goal < 0.25 else False

            resample_env = initial_collision or conker_near_goal or cart_near_goal

        # Retirn the ibservation at which handing over env for stepping
        return self._get_obs()
