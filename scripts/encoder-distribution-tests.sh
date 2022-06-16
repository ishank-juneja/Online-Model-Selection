#!/bin/bash

#Always run script from parent folder of scripts
cd ../
# venv assumed to be present in parent directory
source venv/bin/activate
# Export outermnost project directory to PYTHONPATH for imports to work properly
export PYTHONPATH=$PYTHONPATH:/home/ishank/Desktop/MM-LVSPC

# Format for folder: {simp-model}_enc_nframe, n \in {1, 2}
#  Augmentations to be excluded are specified: Possible exclusions to pass: no_bg_simp_model no_bg_imgnet \
#   no_fg_texture no_bg_shape no_noise
#  Remeber to adjust self.obs_dim in the config file based on number of frames being provided
python3 src/perception_tests/encoder-distribution-test.py --enc-model-name "model_ball_enc_2frame_Apr16_07-30-33"
