from arm_pytorch_utilities.rand import seed
from datetime import datetime
import signal
import sys
import shutil
import argparse
from src.training import MyDatasetBuilder, Trainer
from src.utils import EncDataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import os


def sigint_handler(model_name, model_trainer, signal, frame):
    print('Early exit from trainer, attempting to save current model {0} ...'.format(model_name))
    model_trainer.save()
    sys.exit(0)


def main(args):
    # Attempt to create an EncDataset object with the passed folder name
    enc_dataset = EncDataset(args.folder)

    # Retrieve name of simple model
    simp_model = enc_dataset.get_simp_model()

    # Retrieve number of stacked together frames we want to train this model for
    nframes = enc_dataset.get_nframe()

    # Retrieve python enc config object
    config = enc_dataset.get_enc_cfg()

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    model_name = "model_{0}_enc_{1}frame_{2}".format(simp_model, nframes, current_time)
    # Create a directory for storing model hparams/config
    model_dir_path = "models/encoder/{0}".format(model_name)
    # Copy config file used for creating model to model-dir
    os.makedirs(model_dir_path, exist_ok=True)
    shutil.copyfile('src/config/cartpole_config.py', '{0}/config.py'.format(model_dir_path))

    print('Training Model: {}'.format(model_name))
    # set seed
    seed(randseed=config.seed)
    mywriter = None
    # Use TensorBoardX object to log
    if args.log:
        mywriter = SummaryWriter(flush_secs=20, log_dir='runs/{0}'.format(model_name))
    device = config.device

    print('Making dataloader ...')

    dataset_builder = MyDatasetBuilder(config=config, excluded_augs=args.excluded_augs)

    train_dataset = dataset_builder.get_dataset(dataset_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.batch_size,
                              num_workers=config.num_workers, drop_last=True)

    test_dataset = dataset_builder.get_dataset(dataset_type='test')
    test_sampler = RandomSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=config.batch_size,
                             num_workers=config.num_workers, drop_last=True)

    trainer = Trainer(model_name, config, mywriter)
    # SIGINT handler for early exit to traning, pycharm complains incorrectly
    signal.signal(signal.SIGINT, partial(sigint_handler, model_name, trainer))
    # Train Model
    trainer.model.train()

    print("Training...")
    trainer.run(train_loader, test_loader)
    test_loss = trainer.test(train_loader)
    print("Final train loss: {}".format(test_loss))

    # Prints final test loss
    test_loss = trainer.test(test_loader)
    print("Test loss: {}".format(test_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dcartpole = double-cartpole, dubins = dubins car
    parser.add_argument("--folder",
                        action='store',
                        type=str,
                        help="Name of the folder from which to take train data, "
                             "should follow format of EncFolder even if not saving generated frames to disk",
                        metavar="folder")

    parser.add_argument("--log",
                        action='store_true',
                        help="log traning session with TensorBoard",
                        dest="log")

    parser.add_argument("--augs",
                        action='store',
                        nargs='*',
                        choices=["no_fg_texture", "no_bg_simp_model", "no_bg_shape", "no_bg_imgnet", "no_noise"],
                        type=str,
                        help="Kinds of augmentations to be excluded from training: "
                             "fg_texture=Apply random FG textures"
                             "bg_simp_model=Have other simple models in background as distractors"
                             "bg_shape: Have simulator-esque random regular polygon shapes in the bg"
                             "bg_imgnet: Randomize the background with images from the imgnet dataset",
                        dest="excluded_augs",
                        metavar="excluded_augs")

    args = parser.parse_args()

    main(args)
