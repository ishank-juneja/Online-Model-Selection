#!/bin/bash

pip install -e .
python3 src/run_env.py --env 'MujocoCartpole-v0' --name-traj test --ntraj 1000 --len 30 --seed 1
