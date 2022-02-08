import torch.cuda
from models.UKVAE import UnscentedKalmanVariationalAutoencoder
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import re
from math import pi, cos, sin, atan2, sqrt

parser = argparse.ArgumentParser()
# Name of model being tested
parser.add_argument("--cnn-name", type=str)
# Whether to combine together uncertainties
parser.add_argument("--combine", action="store_true")
args = parser.parse_args()


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

# Coordinate transform a point from sim to plot
def sim2plot(point_2d_array):
    return np.array([int(point_2d_array[0] * 21) + 32, -int(point_2d_array[1] * 21) + 32])

def sim2plot_ellipse(ellipse_pts):
    npts = ellipse_pts.shape[1]
    new_ell_pts = np.zeros_like(ellipse_pts)
    for idx in range(npts):
        new_ell_pts[:, idx] = sim2plot(ellipse_pts[:, idx])
    return new_ell_pts

# Plot 2D mean/std-dev
def plot_state_estimate2D(mu, std_dev):
    # Plot ellipse center
    mu_plot = sim2plot(mu.cpu().detach().numpy()[0])
    std_dev = std_dev.cpu().detach().numpy()[0]
    plt.scatter(mu_plot[0], mu_plot[1], color='g')
    # Arrange std_devs into covariance matrix
    sigma = np.array([[std_dev[0] ** 2, 0.0], [0.0, std_dev[1] ** 2]])
    # Obtain Ellipse Contour
    ell_pts = plot_ellipse(sigma)
    ell_pts = sim2plot_ellipse(ell_pts)
    # Plot Ellipse contour
    plt.fill(mu_plot[0] + ell_pts[0, :] - 32, mu_plot[1] + ell_pts[1, :] - 32, alpha=0.7, color='g', ec=None)


# Path for test data frame
# test_path = "data/MujocoBall-v0/test_traj_22/observation_step_1.npy"
test_path = "data-archive/Conkers-v0/all_test_observations.npy"
# test_path =  "data/MujocoCartpole-v0/test_traj_22/observation_step_1.npy"
state_path = "data/MujocoBall-v0/test_traj_22/traj_states.npy"
# test_path = "data-archive/MujocoBall-v0/all_test_observations.npy"

# create a model for testing
model_name = args.cnn_name
# model_name = 'model_conkers_Feb05_13-33-50'
if 'conkers' in model_name:
    from src.pendulum_analogy_config import Config
    test_data = np.load(test_path)[57, 10, :, :, :]
    # test_data = np.load(test_path)
    test_states = np.load(state_path)
elif 'ball' in model_name:
    from src.ball_config import Config
    test_data = np.load(test_path)
    test_states = np.load(state_path)
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

if 'conkers' in model_name:
    mu = mu[:, 1:]
    stddev = stddev[:, 1:]

print(mu)
print(stddev)

plt.imshow(test_data)
plot_state_estimate2D(mu, stddev)
plt.text(10, 10, str(np.around(stddev[0].cpu().detach().numpy(), 4)), color='r', size=14)
plt.savefig('results/tmp.png')




# avg_stddev = 0.0
# avg_rmse = 0.0
#
# # Use fewer trajectories
# N = 5
# for idx in range(N):
#     print("Starting traj {0}".format(idx + 1))
#     for jdx in range(T):
#         test_obs = test_data[idx, jdx, :, :, :]
#         state_label = test_states[idx, jdx, :]
#         mu, stddev = myensemble.encode_single_observation(test_obs)
#         # avg_rmse += np.sqrt(np.sum(np.square(mu.detach().numpy() - state_label[:3])))
#         if not args.combine:
#             print(mu.reshape(10, nstates))
#             print(stddev.reshape(10, nstates))
#         else:
#             print(mu)
#             print(stddev)
#         # avg_stddev += stddev.sum()
#         plt.imshow(test_obs)
#         plt.show()
# print("Average RMSE between observable part of label and prediction {0}".format(avg_rmse/(N*T)))
# print("Average std dev for observations {0}".format(avg_stddev/(N*T)))

