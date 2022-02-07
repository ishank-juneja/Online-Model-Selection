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

# Path for test data frame
# test_path = "data/MujocoBall-v0/test_traj_22/observation_step_1.npy"
# test_path = "data-archive/Conkers-v0/all_test_observations.npy"
test_path =  "data/MujocoCartpole-v0/test_traj_22/observation_step_1.npy"
state_path = "data/MujocoBall-v0/test_traj_22/traj_states.npy"
# test_path = "data-archive/MujocoBall-v0/all_test_observations.npy"

# create a model for testing
model_name = args.cnn_name
# model_name = 'model_conkers_Feb05_13-33-50'
if 'conkers' in model_name:
    from src.pendulum_analogy_config import Config
    # test_data = np.load(test_path)[57, 10, :, :, :]
    test_data = np.load(test_path)
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

