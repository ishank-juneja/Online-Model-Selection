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

#python3 src/training/train_encoder.py --folder ball_enc_1frame --log --augs no_bg_simp_model no_bg_imgnet no_fg_texture no_bg_shape no_noise
#python3 src/training/train_encoder.py --folder ball_enc_2frame --log --augs no_bg_simp_model no_bg_imgnet no_fg_texture no_bg_shape no_noise
#
#python3 src/training/train_encoder.py --folder cartpole_enc_1frame --log --augs no_bg_simp_model no_bg_imgnet no_fg_texture no_bg_shape no_noise
#python3 src/training/train_encoder.py --folder cartpole_enc_2frame --log --augs no_bg_simp_model no_bg_imgnet no_fg_texture no_bg_shape no_noise

python3 src/training/train_encoder.py --folder dcartpole_enc_1frame --log --augs no_bg_simp_model no_bg_imgnet no_fg_texture no_bg_shape no_noise
python3 src/training/train_encoder.py --folder dcartpole_enc_2frame --log --augs no_bg_simp_model no_bg_imgnet no_fg_texture no_bg_shape no_noise

python3 src/training/train_encoder.py --folder dubins_enc_1frame --log --augs no_bg_simp_model no_bg_imgnet no_fg_texture no_bg_shape no_noise
python3 src/training/train_encoder.py --folder dubins_enc_2frame --log --augs no_bg_simp_model no_bg_imgnet no_fg_texture no_bg_shape no_noise
