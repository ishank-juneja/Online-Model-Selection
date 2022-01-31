from src.agents.trainer import Trainer
from tensorboardX import SummaryWriter
from datetime import datetime
import signal
import sys
import argparse
from image_trajectory_dataset import dataset_builder
from torch.utils.data import DataLoader, RandomSampler
from arm_pytorch_utilities.rand import seed
import os


def sigint_handler(signal, frame):
    print('\nEarly exit')
    print('Saving model to data/experiments/{}'.format(name))
    if args.save and args.train:
        trainer.save()
    sys.exit(0)


def print_memory_usage():
    pass
    #print(torch.cuda.memory_allocated() / 1e9)


signal.signal(signal.SIGINT, sigint_handler)

parser = argparse.ArgumentParser()
parser.add_argument("--load", action="store_true")
parser.add_argument("--load-name", help="path to model for saving or loading")
parser.add_argument("--save", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--viz", action="store_true")
parser.add_argument("--viz-src", choices=["decoder", "gym"], default="decoder")
parser.add_argument("--data-dir", default="../data/trajectories/mujoco_cartpole")
parser.add_argument("--log", action="store_true")
parser.add_argument("--box", action="store_true")
parser.add_argument("--do-control", action="store_true")
parser.add_argument("--experiment", choices=["conkers", "kendama"])
args = parser.parse_args()

if args.experiment == 'conkers':
    from pendulum_analogy_config import Config
else:
    raise ValueError("Invalid experiment choice")

# Load config file
config = Config()

# If loading in a pre-trained model
if args.load:
    name = args.load_name
else:
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    name = "model_{}_{}".format(args.experiment, current_time)
    # Create directory for saving
    model_name = name.replace('-', '_')
    model_name = model_name.replace('.', '_')
    fig_dir = 'models/CNN/{}'.format(model_name)
    os.makedirs(fig_dir, exist_ok=True)
    # Copy config file used for creating model
    import shutil
    shutil.copyfile('src/pendulum_analogy_config.py', '{}/config.py'.format(fig_dir))

print('Model name: {}'.format(name))
# set seed
seed(randseed=config.seed)
writer = None
# Use TensorBoardX object to log
if args.log:
    writer = SummaryWriter(flush_secs=20, log_dir='runs/{}'.format(name))
device = config.device

if args.train or args.test or args.viz:
    print('Loading data...')
    train_loader = None
    test_loader = None

    # Get data
    if args.box:
        # X, A, S = load_box_data(args.data_dir)
        raise ValueError('No dataloader implemented for box data')
    else:
        if args.train:
            train_dataset = dataset_builder(config.data_config, 'train')
            print(len(train_dataset))
            train_sampler = RandomSampler(train_dataset)
            train_loader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      drop_last=True)

        if args.test or args.viz:
            test_dataset = dataset_builder(config.data_config, 'test')
            test_sampler = RandomSampler(test_dataset)
            test_loader = DataLoader(test_dataset, sampler=None,
                                     batch_size=25,
                                     num_workers=config.num_workers,
                                     drop_last=True)

    trainer = Trainer(name, config)
    trainer.model.train()
    if args.load:
        print('Loading Model...')
        trainer.model.load_model(name)
        #import torch
        #torch.save(trainer.model.transition.state_dict(), '../notebooks/{}'.format('rope_dynamics'))

    if args.train:
        print("Training...")
        trainer.run(train_loader, test_loader)
        test_loss = trainer.test(train_loader)
        print("Final train loss: {}".format(test_loss))

    if args.test:
        test_loss = trainer.test(test_loader)
        print("Test loss: {}".format(test_loss))


    if args.viz:
        print(args.viz_src)
        trainer.viz_rollout(test_loader)
