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


if __name__ == '__main__':
    # Create pendulum analogy config object
    config = Config()
    # create a model for testing
    myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name='model_conkers_Feb02_13-29-51')
    myensemble.test = True
    myensemble.encoder.cuda()
    conkers_frame = np.load('/home/ishank/Desktop/MM-LVSPC/data/Conkers-v0/test_traj_num_70/test_observation_9.npy')
    plt.imshow(conkers_frame)
    plt.show()
    mu, stddev = myensemble.encode_single_observation(conkers_frame)
    print(stddev)
    # Get a frame from cartpole dataset
    # cartpole_test_obs = np.load('data/MujocoCartpole-v0-Ishank/all_test_observations.npy')
    # N, T, _, _, _ = cartpole_test_obs.shape
    # cartpole_stddev = 0
    # for idx in range(N//10):
    #     for jdx in range(T):
    #         frame = cartpole_test_obs[idx, jdx, :, :, :]
    #         mu, stddev = myensemble.encode_single_observation(frame)
    #         cartpole_stddev += stddev.sum()
    # torch.cuda.empty_cache()
    # print("Cartpole: {0}".format(10*cartpole_stddev/(N*T)))
    # conkers_test_obs = np.load('data/Conkers-v0/all_test_observations.npy')
    # N, T, _, _, _ = conkers_test_obs.shape
    # conkers_stddev = 0
    # for idx in range(N):
    #     for jdx in range(T):
    #         frame = conkers_test_obs[idx, jdx, :, :, :]
    #         mu, stddev = myensemble.encode_single_observation(frame)
    #         conkers_stddev += stddev.sum()
    # print("Conkers: {0}".format(conkers_stddev/(N*T)))


