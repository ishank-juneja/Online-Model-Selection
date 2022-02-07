import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())

frame_location = 'data/MujocoCartpole-v0/train_traj_1883/observation_step_2.npy'
states_location = 'data/MujocoCartpole-v0/train_traj_1883/traj_states.npy'

frame = np.load(frame_location)
states = np.load(states_location)
state = states[1]

print(state)

plt.imshow(frame)
shift = 2.0
plt.scatter(int(state[1] * 21) + 32, -int(state[2] * 21) + 32)
plt.show()

print(state)

