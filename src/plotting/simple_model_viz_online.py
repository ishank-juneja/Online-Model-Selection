import matplotlib.pyplot as plt
import numpy as np
import os
from src.plotting.simple_model_viz import SimpleModViz
from typing import List
STDDEV_SCALE = 1.0


class SMVOnline(SimpleModViz):
    """
    Simple Model Viz functionality with methods for doing offline perception tests
    """
    def __init__(self, simp_model: str, vel_as_color: bool = False):
        super(SMVOnline, self).__init__(simp_model, vel_as_color)

    def overlay_ball_state(self, img_axis, observed_state, alpha=1.0, color='g', display_t_only: bool = False):
        """
        Override overlay function for cartpole to account for augmented state containing rob_state
         in first 2 dimensions
        :param img_axis:
        :param observed_state:
        :param alpha:
        :param color:
        :param display_t_only:
        :return:
        """
        observed_state = observed_state[2:]
        super(SMVOnline, self).overlay_ball_state(img_axis, observed_state, alpha, color, display_t_only)

    def overlay_cartpole_state(self, img_axis, observed_state, alpha=1.0, color='g', display_t_only: bool = False):
        """
        Override overlay function for cartpole to account for different state representation
        :param img_axis:
        :param observed_state:
        :param alpha:
        :param color:
        :param display_t_only:
        :return:
        """
        # Cam matrix for projecting points from world to pixel space
        cam_mat = np.load(self.config.cam_mat_path)
        # Homogenous world coordinates for cart and mass
        self.carty = 0.0
        cart_world = np.array([observed_state[0], 0.0, self.carty, 1.0])
        mass_world = np.array([observed_state[2], 0.0, observed_state[3], 1.0])
        cart_pixel = cam_mat @ cart_world
        mass_pixel = cam_mat @ mass_world
        # Divide by 2D homogenous scaling term
        cart_pixel = self.homogenous_to_regular_coordinates(cart_pixel)
        mass_pixel = self.homogenous_to_regular_coordinates(mass_pixel)
        # Perform extra steps needed when dealing with 2 side by side frames as opposed to 1 frame
        if self.nframes == 2 and (not display_t_only):
            # Check if dt has been set, else can't deal with 2 frames
            if self.dt is None:
                raise ValueError("Cannot perform overlay on two states when dt has not been set")
            # Augment the x-axis pixel coordinates to correspond to the right half of 2-stacked together frames
            cart_pixel[0] += self.config.imsize
            mass_pixel[0] += self.config.imsize
            # Compute the static portion (cart and mass coords) of the previous state based on current velocity
            #  formula for velocity is (x_t - x_{t-1})/delta_t
            # Determine cart position at t-1
            prev_cart_world_x = observed_state[0] - observed_state[1] * self.dt
            prev_cart_world = np.array([prev_cart_world_x, 0.0, self.carty, 1.0])
            prev_cart_pixel = cam_mat @ prev_cart_world
            prev_cart_pixel = self.homogenous_to_regular_coordinates(prev_cart_pixel)
            # Compute length of the pole in meter, Use plen and angular velocity to determine mass position at {t-1}
            plen = self.euclidean_distance(cart_world, mass_world)

            # - - - - - - - - - - -
            # # Find current angular location
            # theta_cur = atan2(observed_state[1] - observed_state[0], observed_state[2])
            # # Theta previous (wrap around taken care of by trig functions subsequently)
            # theta based angular velocity replaced with linear velocities in state representation
            # theta_prev = theta_cur - observed_state[4] * self.dt
            # If using angular velocity can bring back above instead
            # theta_prev = theta_cur - theta_dot * self.dt
            # prev_mass_world = np.array([prev_cart_world_x + plen * sin(theta_prev), 0.0, self.carty + plen * cos(theta_prev), 1.0])
            # # - - - - - - - - - - -

            prev_mass_worldx = mass_world[0] - observed_state[4] * self.dt
            prev_mass_worldy = mass_world[2] - observed_state[5] * self.dt
            # prev_mass_worldx = observed_state[4]
            # prev_mass_worldy = observed_state[5]
            prev_mass_world = [prev_mass_worldx, 0.0, prev_mass_worldy, 1.0]

            prev_mass_pixel = cam_mat @ prev_mass_world
            prev_mass_pixel = self.homogenous_to_regular_coordinates(prev_mass_pixel)
            self.plot_dumbel(img_axis, (prev_cart_pixel[0], prev_cart_pixel[1]),
                             (prev_mass_pixel[0], prev_mass_pixel[1]), color=color, alpha=alpha)
        self.plot_dumbel(img_axis, (cart_pixel[0], cart_pixel[1]), (mass_pixel[0], mass_pixel[1]),
                         color=color, alpha=alpha)
        return

