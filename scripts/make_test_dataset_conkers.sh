#!/bin/bash

pip install -e .
python3 src/run_env.py --env 'Conkers-v0' --name-traj test --ntraj 100 --len 20 --seed 0
