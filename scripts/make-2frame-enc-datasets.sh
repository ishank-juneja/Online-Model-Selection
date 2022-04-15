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
# This is needed for mixing dm_control and mujoco_py imports ...
# https://github.com/deepmind/dm_control/issues/266
export MUJOCO_GL=egl

# Ball can't have long trajectories so lengths kept shorter at 10 and 15
python3 src/simp_mod_datasets/make_enc_dataset.py --save --folder ball_enc_2frame --ntraj 10000 1000 --len 10 15 --datasets train test # Slow
#python3 src/simp_mod_datasets/make_enc_dataset.py --save --folder ball_enc_2frame --ntraj 10000 1000 --len 10 15 --datasets train test --save-traj-viz --overlay
python3 src/simp_mod_datasets/make_enc_dataset.py --save --folder cartpole_enc_2frame --ntraj 5000 500 --len 20 25 --datasets train test # Quick
#python3 src/simp_mod_datasets/make_enc_dataset.py --save --folder cartpole_enc_2frame --ntraj 5000 500 --len 20 25 --datasets train test  --save-traj-viz --overlay
python3 src/simp_mod_datasets/make_enc_dataset.py --save --folder dcartpole_enc_2frame --ntraj 5000 500 --len 20 25 --datasets train test # Very Very Slow
#python3 src/simp_mod_datasets/make_enc_dataset.py --save --folder dcartpole_enc_2frame --ntraj 5000 500 --len 20 25 --datasets train test  --save-traj-viz --overlay
python3 src/simp_mod_datasets/make_enc_dataset.py --save --folder dubins_enc_2frame --ntraj 5000 500 --len 20 25 --datasets train test # Very Quick
#python3 src/simp_mod_datasets/make_enc_dataset.py --save --folder dubins_enc_2frame --ntraj 5000 500 --len 20 25 --datasets train test  --save-traj-viz --overlay
