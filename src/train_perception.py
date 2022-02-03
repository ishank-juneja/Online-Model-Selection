import torch
import argparse
from arm_pytorch_utilities.rand import seed
from torch.utils.tensorboard import SummaryWriter
import importlib
import shutil
import os
from datetime import datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Simple model for which training being performed
    parser.add_argument("--simple-model", type=str, choices=['cartpole', 'ball', 'double-cartpole', 'dubins'])
    # Path to train data for this simple model
    parser.add_argument("--train-data", type=str)
    # Path to test data for this simple model
    parser.add_argument("--test-data", type=str)
    # Path to save/load trained perception networks
    parser.add_argument("--nn-path", type=str, default='models/CNN')
    # Load pretrained model for testing only
    parser.add_argument("--load-name", type=str)
    # Log training via TB
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args("")
    # Import the appropriate simple-model config file
    Config = getattr(importlib.import_module("src.config.{0}".format(args.simple_model)), 'Config')
    config = Config()
    # If loading in a pre-trained model
    if args.load_name:
        nn_name = args.load_name
    # Otherwise assign model name with simple-model type and current-time
    else:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        nn_name = "{0}_model_{1}".format(args.simple_model, current_time)
        # Create directory for saving the config with which NN is trained
        model_name = nn_name.replace('-', '_')
        model_name = model_name.replace('.', '_')
        save_dir_pth = os.path.join(args.nn_path, model_name)
        os.makedirs(save_dir_pth, exist_ok=True)
        # Copy config file used for creating this model
        shutil.copyfile('src/config/cartpole.py', '{0}/config.py'.format(save_dir_pth))
    # Setup TB writer
    mywriter = None
    if args.log:
        mywriter = SummaryWriter(flush_secs=20, log_dir='runs/{0}'.format(nn_name))
    # Retrieve GPU device name
    device = config.device
    # Random seed for making model training reproducable
    seed(config.seed)

    if args.train_data is not None:
        print("Training model {0}".format(nn_name))

