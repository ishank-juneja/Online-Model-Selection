import argparse
from arm_pytorch_utilities.rand import seed
from datetime import datetime
import logging
import numpy as np
import os
from src.agents import CatchingAgent, ConkersAgent, KendamaAgent
from src.plotting import GIFMaker, SimpModLibViz
from src.utils import ResultDirManager, setup_logging
import torch


def main(args):
    task_name = args.task

    # - - - - - - - - - - - - - - - - - - - - - - - - -
    # Setup logging to console and file handler
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    dir_manager = ResultDirManager()

    # Folder for all things generated in this run
    run_folder_name = "online_run_{0}_{1}".format(task_name, current_time)
    run_folder_path = "runs/online/{0}".format(run_folder_name)
    dir_manager.add_location("run_log_root", run_folder_path)
    # Within root path make folders for certain kinds of data
    data_folder_path = run_folder_path + "/data"
    videos_folder_path = run_folder_path + "/videos"
    # tmp location to store animation frames
    tmp_loc_path = videos_folder_path + "/tmp"
    dir_manager.add_location("run_data", data_folder_path)
    dir_manager.add_location("video_data", videos_folder_path)
    dir_manager.add_location("tmp", tmp_loc_path)
    # Path to log file
    log_file_path = os.path.join(run_folder_path, "run.log")

    if not setup_logging(console_log_output="stdout", console_log_level="info", console_log_color=True,
                         logfile_file=log_file_path, logfile_log_level="debug", logfile_log_color=False,
                         log_line_template="%(color_on)s[%(created)d] [%(threadName)s] [%(levelname)-8s] %(message)s%(color_off)s"):
        print("Failed to setup logging, aborting.")
        exit()
    # - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - - - - - - - - - - - - - - - - - - - - - -
    # Setup video/GIF visualizations
    viz = SimpModLibViz(task_name=task_name, model_names=args.models)
    # - - - - - - - - - - - - - - - - - - - - - - - - -

    # set seed
    seed(randseed=args.seed)

    # Make an agent to solve task
    # TODO: Remove hard-coding of device
    #  Instead have the same torch device for an entire program the way Johnson does it ...
    mydevice = 'cuda:0'

    # Prepare key word args for agent
    kwargs = {'smodel_list': args.models,
              'device': mydevice,
              'dir_manager': dir_manager}
    if task_name == 'catching':
        agent = CatchingAgent(**kwargs)
    elif task_name == 'conkers':
        agent = ConkersAgent(**kwargs)
    elif task_name == 'kendama':
        agent = KendamaAgent(**kwargs)
    else:
        raise NotImplementedError
    logging.info("Created agent for task {0}".format(task_name))

    # Reset all online estimated parameters associated with agent
    agent.reset_trial()

    agent.do_episode()
    # Save episode data to disk
    ep_file_name = "test_episode.npz"
    ep_file_path = dir_manager.get_abs_path('run_data') + '/' + ep_file_name
    agent.save_episode_data(ep_file_path)
    agent_ep_history, smodels_ep_histories = agent.load_episode_data(ep_file_path)
    # Assemble plotting function kwargs
    kwargs = {'viz_name': 'filtered_state_on_frames',
              'agent_ep_history': agent_ep_history,
              'model_ep_histories': smodels_ep_histories,
              'save_dir': dir_manager.get_abs_path('tmp'),
              }
    viz(**kwargs)

    gif_maker = GIFMaker(delay=35)

    gif_path = videos_folder_path + "/result.gif"
    gif_maker.make_gif(gif_path=gif_path, frames_dir=tmp_loc_path)

    viz.cleanup_frames(tmp_loc_path)

    # Expect tmp dir to be empty after run, try and remove
    try:
        os.rmdir(tmp_loc_path)
    except OSError:
        logging.error("tmp location {0} not empty on exit ...".format(tmp_loc_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task",
                        action='store',
                        type=str,
                        choices=["conkers", "catching", "kendama"],
                        help="Short name of task being performed by MM-LVSPC",
                        metavar="task",
                        dest="task")

    parser.add_argument("--ntrials",
                        action='store',
                        type=int,
                        default=3,
                        help="Number of independent trials for mm-lvspc",
                        dest="ntrials",
                        metavar="ntrials")

    parser.add_argument("--nepisodes",
                        action='store',
                        type=int,
                        default=20,
                        help="Number of episodes per trial",
                        dest="nepisodes",
                        metavar="nepisodes")

    parser.add_argument("--models",
                        action='store',
                        nargs='+',
                        choices=['cartpole', 'ball', 'dcartpole', 'dubins'],
                        type=str,
                        help="Models available in simple model library",
                        dest="models",
                        metavar="models-in-lib")

    parser.add_argument("--seed",
                        action='store',
                        type=int,
                        default=0,
                        help="seed np, torch and random libs",
                        dest="seed",
                        metavar="seed")

    parser.add_argument("--do-online-learning",
                        action='store_true',
                        help="Whether to update transition model online. Update -> Use GP, Don't Update -> Don't use GP",
                        dest="do_online_learning")

    args = parser.parse_args()

    main(args)
