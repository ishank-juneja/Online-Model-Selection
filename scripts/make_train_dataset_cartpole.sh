#!/bin/bash

pip install -e .
python3 src/run_env.py --env 'MujocoCartpole-v0' --name-traj train --ntraj 5000 --len 20 --seed 0
