#!/bin/bash

pip install -e .
python3 src/cup_test.py --env 'Kendama-v0' --name-traj test --ntraj 100 --len 20 --seed 0 --show
