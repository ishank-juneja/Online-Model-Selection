#!/bin/bash

pip install -e .
python3 src/run_env.py --env 'MujocoBall-v0' --name-traj train --ntraj 10000 --len 10 --seed 0
