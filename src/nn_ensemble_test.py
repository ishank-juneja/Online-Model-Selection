import torch.cuda
from src.pendulum_analogy_config import Config
from src.models.UKVAE import UnscentedKalmanVariationalAutoencoder
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

test_data = np.load(cartpole_test_observations)
test_states = np.load(cartpole_test_states)
N, T, nstates = test_states.shape

# Create pendulum analogy config object
config = Config()
# Use CPU at test time
config.device = 'cpu'
# create a model for testing
# myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name='model_conkers_Feb03_20-40-20')
# myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name='model_conkers_Feb03_13-56-07')
# myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name='model_conkers_Feb02_13-29-51')
# myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name='model_conkers_Feb03_22-14-08')
# myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name='model_conkers_Feb03_23-44-36')
# myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name='model_conkers_Feb04_00-20-40')
myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name='model_conkers_Feb04_00-20-40')
myensemble.test = True
# myensemble.encoder.cuda()

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
        avg_rmse += np.sqrt(np.sum(np.square(mu.detach().numpy() - state_label[:3])))
        avg_stddev += stddev.sum()
print("Average RMSE between observable part of label and prediction {0}".format(avg_rmse/(N*T)))
print("Average std dev for observations {0}".format(avg_stddev/(N*T)))

