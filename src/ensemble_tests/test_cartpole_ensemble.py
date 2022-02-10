import torch.cuda
from src.models.UKVAE import UnscentedKalmanVariationalAutoencoder
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import re
from math import pi, cos, sin, atan2, sqrt
from src.results_dir_manager import ResultDirManager
dir_manager = ResultDirManager()

parser = argparse.ArgumentParser()
# Name of model being tested
parser.add_argument("--cnn-name", type=str)
# Whether to combine together uncertainties
parser.add_argument("--combine", action="store_true")
args = parser.parse_args()


# Plots a horizontal error bar along length of entire image for perceived cart/robot position
def plot_horizontal_error_bar(xpos, std_dev):
    xpos_plot = sim2plot1Dx(xpos)
    std_dev_plot = std_dev * 21
    # print(std_dev_plot)
    plt.axvline(xpos_plot, ymin=0, ymax=64, color='r')
    plt.axvspan(xpos_plot - std_dev_plot, xpos_plot + std_dev_plot, ymin=0, ymax=64, alpha=0.6, color='g')
    return

# Takes the covariance matrix Sigma as an input
def plot_ellipse(Sigma, num_sigmas=1):
    # Perform eigen value decomposition of Sigma
    e_vals, e_vecs = np.linalg.eig(Sigma)
    # Get the eigen vector which is the direction of major axis
    evec1 = e_vecs[:, 0]
    # Get the minor axis
    evec2 = e_vecs[:, 1]
    # Find the angle that ellipse needs to be rotated to get the right ellipse contour
    theta = atan2(evec1[1], evec1[0])
    # Get length of semi-major and semi-minor axes, use 1-sigma contour
    a = num_sigmas * sqrt(e_vals[0])
    b = num_sigmas * sqrt(e_vals[1])
    # Generate a Standard Ellipse
    t = np.linspace(0, 2*pi, 100)
    Ell = np.array([a*np.cos(t), b*np.sin(t)])
    # 2-D rotation matrix
    R_rot = np.array([[cos(theta), -sin(theta)],
                  [sin(theta), cos(theta)]])
    # Rotated Ellipse
    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])
    return Ell_rot

def sim2plot1Dx(pos: float):
    return int(pos * 21) + 32

def sim2plot1Dy(pos: float):
    return -int(pos * 21) + 32

# Coordinate transform a point from sim to plot
def sim2plot(point_2d_array):
    return np.array([sim2plot1Dx(point_2d_array[0]), sim2plot1Dy(point_2d_array[1])])

def sim2plot_ellipse(ellipse_pts):
    npts = ellipse_pts.shape[1]
    new_ell_pts = np.zeros_like(ellipse_pts)
    for idx in range(npts):
        new_ell_pts[:, idx] = sim2plot(ellipse_pts[:, idx])
    return new_ell_pts

# Plot 2D mean/std-dev
def plot_state_estimate2D(mu, std_dev):
    mu_plot = sim2plot(mu)
    # Arrange std_devs into covariance matrix
    std_dev = std_dev * 21
    sigma_plot = np.array([[std_dev[0] ** 2, 0.0], [0.0, std_dev[1] ** 2]])
    # Obtain Ellipse Contour
    ell_pts = plot_ellipse(sigma_plot)
    # print(ell_pts)
    # Plot Ellipse contour
    plt.fill(mu_plot[0] + ell_pts[0, :], mu_plot[1] + ell_pts[1, :], alpha=0.6, color='g', ec=None)
    # Plot ellipse center
    plt.scatter(mu_plot[0], mu_plot[1], color='r', s=10)

def plot_true_state_label(full_state):
    obs_state = full_state[:3]
    mass_pos = obs_state[1:]
    cart_pos = obs_state[0]
    plt.axvline(sim2plot1Dx(cart_pos), ymin=0, ymax=64)
    mass_pos_plot = sim2plot(mass_pos)
    plt.scatter(mass_pos_plot[0], mass_pos_plot[1], s=10)

# Path to test_data_frames
# test_path = "data/MujocoBall-v0/test_traj_22/observation_step_1.npy"
# test_path = "data-archive/Conkers-v0/all_test_observations.npy"
# test_path = "data-archive/Conkers-v0/all_test_observations.npy"
for idx in range(100):
    for jdx in range(30):
        test_path = "data-archive/Conkers-v0/all_test_observations.npy"
        # test_path = "data/Kendama-v0/test_traj_{0}/observation_step_8.npy".format(idx + 1)
        state_path = "data/MujocoBall-v0/test_traj_{0}/traj_states.npy".format(idx + 1)
        # test_path = "data-archive/MujocoBall-v0/all_test_observations.npy"

        # create a model for testing
        model_name = args.cnn_name
        # model_name = 'model_conkers_Feb05_13-33-50'
        from src.pendulum_analogy_config import Config
        test_states = np.load(state_path)
        if 'Conkers' in test_path:
            test_data = np.load(test_path)[idx, jdx, :, :, :]
        else:
            test_data = np.load(test_path)
        # Create pendulum analogy config object
        config = Config()

        # N, T, _ = test_states.shape
        nstates = config.observation_dimension
        myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name=model_name)
        if args.combine:
            myensemble.test = True
        else:
            myensemble.test = False
        # Use CPU at test time
        # config.device = 'cpu'
        myensemble.encoder.cuda()

        mu, stddev = myensemble.encode_single_observation(test_data)

        mu = mu[0].cpu().detach().numpy()
        stddev = stddev[0].cpu().detach().numpy()

        print(mu)
        print(stddev)

        plt.imshow(test_data)
        plot_state_estimate2D(mu[1:], stddev[1:])
        plot_horizontal_error_bar(mu[0], stddev[0])
        # plot_true_state_label(test_states[0])
        plt.title(str(np.around(stddev, 4)), color='k', size=18)
        dir_manager.add_location('results', 'results/')
        fig_path = dir_manager.next_path('results', prefix='tom_cartpole_perception_', postfix='OOD_conkers%s.png')
        plt.savefig(fig_path)
        plt.clf()
        plt.cla()
        plt.close()
