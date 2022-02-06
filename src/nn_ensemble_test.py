import torch.cuda
from models.UKVAE import UnscentedKalmanVariationalAutoencoder
import numpy as np
import os
import matplotlib.pyplot as plt


import re

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# Paths for test data

# states: 100 x 30 x 2
conkers_test_for_ball_obs = 'data/Conkers-v0/all_test_observations.npy'
conkers_test_for_ball_states = 'data/Conkers-v0/all_test_states.npy'
# states: 100 x 30 x 2
conkers_no_rope_test_for_ball_obs = 'data/Conkers-v1/all_test_observations.npy'
conkers_no_rope_test_for_ball_states = 'data/Conkers-v1/all_test_states.npy'
# Regular ball test data (i.d. to train data)
# states: 2000 x 15 x 2
ball_test_observations = 'data/MujocoBall-v0/all_test_observations.npy'
ball_test_states = 'data/MujocoBall-v0/all_test_states.npy'
# Regular cartpole test data
cartpole_test_observations = 'data/MujocoCartpole-v0-25/all_test_observations.npy'
cartpole_test_states = 'data/MujocoCartpole-v0-25/all_test_states.npy'


# create a model for testing
model_name = 'model_ball_Feb05_12-46-00.pt'
# model_name = 'model_conkers_Feb05_13-33-50'
if 'conkers' in model_name:
    from src.pendulum_analogy_config import Config
    test_data = np.load(cartpole_test_observations)
    test_states = np.load(cartpole_test_states)
elif 'ball' in model_name:
    from src.ball_config import Config
    test_data = np.load(ball_test_observations)
    test_states = np.load(ball_test_states)
# Create pendulum analogy config object
config = Config()

N, T, nstates = test_states.shape
myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name=model_name)
myensemble.test = True
# Use CPU at test time
# config.device = 'cpu'
myensemble.encoder.cuda()

avg_stddev = 0.0
avg_rmse = 0.0

# Use fewer trajectories
N = 5
for idx in range(N):
    print("Starting traj {0}".format(idx + 1))
    for jdx in range(T):
        test_obs = test_data[idx, jdx, :, :, :]
        state_label = test_states[idx, jdx, :]
        mu, stddev = myensemble.encode_single_observation(test_obs)
        # avg_rmse += np.sqrt(np.sum(np.square(mu.detach().numpy() - state_label[:3])))
        # print(mu.reshape(10, 3))
        # print(stddev.reshape(10, 3))
        print(mu)
        print(stddev)
        # avg_stddev += stddev.sum()
        plt.imshow(test_obs)
        plt.show()
print("Average RMSE between observable part of label and prediction {0}".format(avg_rmse/(N*T)))
print("Average std dev for observations {0}".format(avg_stddev/(N*T)))

