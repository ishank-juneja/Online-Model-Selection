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

    def __call__(self, kwargs):
        self.viz_trajectory(**kwargs)

    def viz_trajectory(self, masked_frames: List[np.ndarray], traj_mu: List[np.ndarray], traj_stddev: np.ndarray, traj_mu_full: np.ndarray, save_dir: str):
        self.data_keys = ["mu_y",
                          "sigma_y",
                          "mu_z",
                          "sigma_z",
                          "param_mu",
                          "param_sigma",
                          "seg_conf",
                          "masked_frame"
                          ]
        """
        Save animation online execution frames as image files, key word arguments use the same keys
        as the data_keys of SimpleModBook
        :param masked_frames: The frames after segmentation associated with a particular simple model
        :param traj_mu: List
        :param traj_stddev: stddevs is a concatenated array of all 3 (total, alea, epi) kinds of uncertanties
        :param traj_mu_full: mu_np: Individual means of the ensemble members
        :param save_dir: Directory into which to save results
        :return:
        """
        # Fill this in based on what the book save_viz method asks for

