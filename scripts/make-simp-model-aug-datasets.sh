#!/bin/bash

#Always run script from parent folder of scripts
cd ../
# venv assumed to be present in parent directory
source venv/bin/activate
# Export outermnost project directory to PYTHONPATH for imports to work properly
export PYTHONPATH=$PYTHONPATH:/home/ishank/Desktop/simple-model-perception
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ishank/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_KEY_PATH=$MUJOCO_KEY_PATH:/home/ishank/.mujoco
export MUJOCO_GL=egl

python3 src/aug_datasets/make_simp_mod_aug_dataset.py --folder ball_aug
python3 src/aug_datasets/make_simp_mod_aug_dataset.py --folder cartpole_aug
python3 src/aug_datasets/make_simp_mod_aug_dataset.py --folder dcartpole_aug
python3 src/aug_datasets/make_simp_mod_aug_dataset.py --folder dubins_aug
