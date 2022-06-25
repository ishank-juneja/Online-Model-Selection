import matplotlib.pyplot as plt
import numpy as np
import os
from src.plotting.simple_model_viz import SimpleModViz
from typing import List


class SMVOnline(SimpleModViz):
    """
    Simple Model Viz functionality with methods for doing offline perception tests
    """
    def __init__(self, simp_model: str, cam_mat: np.ndarray = None, vel_as_color: bool = False):
        super(SMVOnline, self).__init__(simp_model, vel_as_color)
        # - - - - - - - - - - - - - - - - - - - - -
        # Params related to visualizing of robot state
        # y coordinate of robot position in 2D plane
        self.rob_y_pos = 0.0
        self.rob_y_vel = 0.0
        self.rob_dims = 4
        # Color of dot representing rob_state
        self.rob_color = 'r'
        # - - - - - - - - - - - - - - - - - - - - -
        # Override cam_mat set in Base Class
        if cam_mat is not None:
            self.cam_mat = cam_mat

    def overlay_rob_state(self, img_axis, online_state, alpha=1.0, color='g', display_t_only: bool = False):
        """
        Overlay the GT robot state in the same way we overlay ball state as a point in 2D
        :param img_axis:
        :param online_state: Assumed to be size 2 1D np array with pos and vel
        :param alpha:
        :param color:
        :param display_t_only:
        :return:
        """
        # Make rob_state into a full 2D point [x, y, v_x, v_y] for projecting
        rob_state = np.zeros(self.rob_dims, dtype=np.float64)
        rob_state[0] = online_state[0]
        rob_state[1] = self.rob_y_pos
        rob_state[2] = online_state[1]
        rob_state[3] = self.rob_y_vel
        # Use overlay_ball_state of SMViv to overlay point in 2D space
        #  Overlay ball state is essentially overlay a 2D point ....
        super(SMVOnline, self).overlay_ball_state(img_axis, rob_state, alpha, color, display_t_only)
        return

    def overlay_ball_state(self, img_axis, online_state, alpha=1.0, color='g', display_t_only: bool = False):
        """
        Override overlay function for cartpole to account for augmented state containing rob_state
         in first 2 dimensions
        :param img_axis:
        :param online_state:
        :param alpha:
        :param color:
        :param display_t_only:
        :return:
        """
        offline_state = online_state[2:]
        super(SMVOnline, self).overlay_ball_state(img_axis, offline_state, alpha, color, display_t_only)
        # Overlay rob state ...
        rob_state = online_state[:2]
        self.overlay_rob_state(img_axis, rob_state, alpha, color=self.rob_color, display_t_only=display_t_only)

    def overlay_cartpole_state(self, img_axis, online_state, alpha=1.0, color='g', display_t_only: bool = False):
        """
        Override overlay function for cartpole to account for different state representation
        :param img_axis:
        :param online_state: Cartpole state rep: [rob_x, rob_v, x_mass, y_mass, v_cart, vx_mass, vy_mass]
        :param alpha:
        :param color:
        :param display_t_only:
        :return:
        """
        # Scramble observed state in format specified in paper into the old offline used
        #  cartpole state format of [x_cart, x_mass, y_mass, v_cart, vx_mass, vy_mass]
        # Allocate container to hold unscrambled online state, same dims as offline
        offline_state = np.copy(online_state)
        offline_state[1] = online_state[2]
        offline_state[2] = online_state[3]
        offline_state[3] = online_state[1]
        super(SMVOnline, self).overlay_cartpole_state(img_axis, offline_state, alpha, color, display_t_only)
        # Overlay rob state ...
        rob_state = online_state[:2]
        self.overlay_rob_state(img_axis, rob_state, alpha, color=self.rob_color, display_t_only=display_t_only)
        return
