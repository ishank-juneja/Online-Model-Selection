#!/bin/bash

#Always run script from parent folder of scripts
cd ../
# venv assumed to be present in parent directory
source venv/bin/activate
# Export outermnost project directory to PYTHONPATH for imports to work properly
export PYTHONPATH=$PYTHONPATH:/home/ishank/Desktop/MM-LVSPC

# Format for folder: {simp-model}_seg_nframe{m-mask}, n \in {1, 2}, m-mask \in {"", 1mask, 2mask}
#  Augmentations to be excluded are specified: Possible exclusions to pass: no_bg_simp_model no_bg_imgnet \
#   no_fg_texture no_bg_shape no_noise
#  Traning config in src/config/seg_config.py
python3 src/training/train_seg.py --arch MRCNN --folder ball_seg_1frame --viz-output --augs no_bg_simp_model --pretrained
