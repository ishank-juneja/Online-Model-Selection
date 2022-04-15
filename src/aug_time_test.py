import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from src.training.enc_training_augs import ImageTrajectoryAugmenter


dom_rand = ImageTrajectoryAugmenter()

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

start = time.time()
for idx in range(5000):
    traj_tensor = npy_loader("data/Ball-v0/train_traj_343/traj_observations.npy")
    traj_np = traj_tensor.detach().cpu().numpy()
    plt.imshow(traj_np.reshape(64*10, 64, 3))
    plt.show()
    bg_changed = dom_rand.apply_random_augmentation(traj_tensor).detach().cpu().numpy()
    plt.imshow(bg_changed.reshape(64*10, 64, 3))
    plt.show()
end = time.time()

print("Time taken = {0}s".format(end-start))

