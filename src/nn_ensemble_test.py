import torch
from torch import nn
from src.pendulum_analogy_config import Config
from src.models.base_model import BaseModel
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
    # create a base model for testing
    myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name='model_conkers_Feb01_08-49-58')
    # Get a frame from cartpole dataset
    frame_file_names = os.listdir('data/MujocoCartpole-v0/test_traj_num_214/')
    frame_file_names = natural_sort(frame_file_names)
    for filename in frame_file_names:
        if 'observation' in filename:
            frame = np.load(os.path.join('data/MujocoCartpole-v0/test_traj_num_214/', filename))
            plt.imshow(frame)
            plt.show()
            myensemble.encoder.cuda()
            mu, std = myensemble.encode_single_observation(frame)
            print(mu.reshape(10, 3).mean(axis=0))
