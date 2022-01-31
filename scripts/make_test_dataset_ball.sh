#!/bin/bash

pip install -e .
python3 src/run_env.py --env 'MujocoBall-v0' --name-traj test --ntraj 2000 --len 15 --seed 1
