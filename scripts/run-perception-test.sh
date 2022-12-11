#!/bin/bash

#Always run script from parent folder of scripts
cd ../
# venv assumed to be present in parent directory
source venv/bin/activate
# Export outermnost project directory to PYTHONPATH for imports to work properly
export PYTHONPATH=$PYTHONPATH:/home/ishank/Desktop/OMS
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ishank/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_KEY_PATH=$MUJOCO_KEY_PATH:/home/ishank/.mujoco
export MUJOCO_GL=egl

python3 src/perception_tests/run_test.py --enc-model-name model_ball_enc_1frame_Sep24_11-11-40
#python3 src/perception_tests/run_test.py --enc-model-name model_dubins_enc_2frame_Apr16_23-35-43
