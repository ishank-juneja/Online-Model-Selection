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

#python3 src/simp_mod_datasets/make_seg_dataset.py --remake --folder ball_seg_1frame --datasets train test --nframes 1000 100 --seed 0
#python3 src/simp_mod_datasets/make_seg_dataset.py --remake --folder cartpole_seg_1frame --datasets train test --nframes 1000 100 --seed 0
python3 src/simp_mod_datasets/make_seg_dataset.py --remake --folder dcartpole_seg_1frame --datasets train test --nframes 1000 100 --seed 0
#python3 src/simp_mod_datasets/make_seg_dataset.py --remake --folder dubins_seg_1frame --datasets train test --nframes 1000 100 --seed 0
