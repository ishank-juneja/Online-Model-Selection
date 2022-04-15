import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym_cenvs.envs.base import MujocoBase


class Dubins(mujoco_env.MujocoEnv, utils.EzPickle, MujocoBase):
    def __init__(self):
        xml_path = os.path.abspath('gym_cenvs/assets/dubins.xml')

        # - - - - - Init Attributes Prior since some needed for model creation - - - - - -
        # Episode/task of a contiguous trajectory complete?
        self.done = False
        # Domain rand over geometry and color
        self.randomize = True
        # Variable to hold the static component of state [x (m), y (m), theta (rad)]
        self.myqpos = np.array([0.0, 0.0, 0.0])
        # Variable to hold vel parts of state [V (translational speed, m/s), theta_dot (rad/s)]
        #  The velocity at time `t` takes the model from its qpos at time t-1 to its qpos at time t
        self.myqvel = np.array([0.0, 0.0])
        # - - - - - - - - - - - - - - - -
        MujocoBase.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip=1)
        utils.EzPickle.__init__(self)
        # Must come after model init
        self.reset_model()
        # Create camera matrix for projection
        self.cam_matrix = self.get_cam_mat()

    def randomize_geom(self):
        # Randomize changes within desired range
        # Default car length is 0.2
        car_length = self.np_random.uniform(low=(0.3 - 0.1), high=(0.3 + 0.1))
        # Keep the width noticably smaller than the width to disambiguate heading
        car_width = self.np_random.uniform(low=0.1, high=0.6 * car_length)
        # Height of car, even though we view head on onto the x-y plane
        #  heights effects the shadow of the car
        car_height = self.np_random.uniform(low=0.03, high=0.07)
        # [x, y, z] Dimensions of rectangle (box in 3D) geometry
        self.sim.model.geom_size[self.sim.model.geom_name2id('gdubins')] = [car_length, car_width, car_height]

    def randomize_color(self):
        randidx = self.np_random.choice(a=self.ncolors)
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('gdubins')] = self.color_options[randidx]

    def step(self, action: np.ndarray):
        """
        Single action u = theta^{.}
        :param action: Angular velocity in rad/s
        :return: Return template for simple models
        """
        # state = self._get_state()
        # We compute the state here within the class definition and don't rely on simulator to update state

        # Set theta_dot that took me from state before taking action (t-1) to state after taking action
        # (returned, state at time t)
        #  Multiplied action \in [-1, 1] for more pronounced rotation
        self.myqvel[1] = 3 * action

        # Update x and y positions based on heading at t-1 and transational velocity
        # Update current heading
        del_theta = self.myqvel[1] * self.dt
        # Note: This is theta at time t
        theta = self.myqpos[2]
        self.myqpos[2] += del_theta
        self.myqpos[0] += self.myqvel[0] * self.dt * np.cos(theta) # x
        self.myqpos[1] += self.myqvel[0] * self.dt * np.sin(theta) # y

        # do simulation is necessary for updating the observation, we manually update state after running
        self.do_simulation(0.0, self.frame_skip)
        self.set_state(qpos=self.myqpos, qvel=np.zeros_like(self.myqpos))

        ob = self._get_obs()

        # Terminate if the car center goes out of view
        out_of_view = np.max(np.abs(self.sim.data.qpos[0:2])) > 1.7

        # out_of_view = out_of_view_x or out_of_view_y
        # self.done is never set to True since there is no task
        done = out_of_view or self.done

        # dummy cost for simple models
        cost = 0.0
        # Return cos(theta), sin(theta) as state instead of just theta
        cos_theta = np.array([np.cos(theta)])
        sin_theta = np.array([np.sin(theta)])
        return ob, -cost, done, {'success': self.done, 'state': np.concatenate((self.myqpos[:2], cos_theta, sin_theta,
                                                                                self.myqvel[:1]))}

    # State is [x_ball, z_ball]
    def _get_state(self):
        # x_car, y_car, theta_car obatined using the qpos of the x, y slide joints, and theta hinge
        _st = self.sim.data.qpos[:].copy()
        return _st

    def reset(self):
        self.done = False
        if self.randomize:
            self.randomize_geom()
            self.randomize_color()
        return self.reset_model()

    def reset_model(self):
        """
        Re-Initialize the model at a random location, with a random heading,
        random transational velocity within reasonable range (transational vel constant for a trajectory)
        and initialized with 0 angular turning rate (\dot{\theta} = 0)
        :return:
        """
        # Init the car at a location around the center of the arena
        car_x = self.np_random.uniform(low=-1.0, high=1.0)
        car_y = self.np_random.uniform(low=-1.0, high=1.0)

        # Select a random init orientation
        theta = self.np_random.uniform(low=-np.pi, high=np.pi)

        # Initialize the position and heading of car
        self.myqpos[0] = car_x
        self.myqpos[1] = car_y
        self.myqpos[2] = theta

        # Pick transational velocity from episode from range
        self.myqvel[0] = np.random.uniform(low=0.5, high=3.0)
        # Fix initial theta_dot to 0, this won't have an effect since we update/override states
        #  manually, internally
        self.myqvel[1] = 0.0

        # Set simulator state, manually push zeros for velocities
        self.set_state(self.myqpos, np.zeros_like(self.myqpos))
        return self._get_obs()
