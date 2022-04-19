import argparse
from arm_pytorch_utilities.rand import seed
from datetime import datetime
import logging
import numpy as np
import os
from src.agents import ConkersAgent
from src.simp_mod_library.simp_mod_lib import SimpModLib
from src.utils import ResultDirManager, setup_logging
import torch


def main(args):
    task_name = args.task

    # Setup logging to console and file handler
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    dir_manager = ResultDirManager()

    # Folder for all things this run
    run_folder_name = "online_run_{0}_{1}".format(task_name, current_time)
    run_folder_path = "runs/online/{0}".format(run_folder_name)
    dir_manager.add_location('run_log', run_folder_path)

    # Path to log file
    log_file_path = os.path.join(run_folder_path, "run.log")

    if not setup_logging(console_log_output="stdout", console_log_level="debug", console_log_color=True,
                         logfile_file=log_file_path, logfile_log_level="debug", logfile_log_color=False,
                         log_line_template="%(color_on)s[%(created)d] [%(threadName)s] [%(levelname)-8s] %(message)s%(color_off)s"):
        print("Failed to setup logging, aborting.")
        exit()

    # set seed
    seed(randseed=args.seed)

    if task_name == 'conkers':
        agent = ConkersAgent(smodel_list=args.models)
    else:
        raise NotImplementedError
    logging.info("Created agent for task {0}".format(task_name))

    train_success = []
    test_success = []
    test_times = []
    train_times = []
    trial = 0

    while trial < args.ntrials:
        # Reset the learnable params in model library
        agent.model_lib.reset()

        ep = 0
        episode_train_times = []
        episode_test_times = []
        episode_train_success = []
        episode_test_success = []

        if trial > 0:
            train_success_rate = np.mean(np.asarray(train_success), axis=0)
            logging.info("Trial: {0} training success rate {1}".format(trial, train_success_rate))

        t_success = []
        t_times = []

        logging.info("Testing before trial {0}".format(trial))
        for test in range(20):
            fail, t = agent.do_episode()
            t_success.append(0. if (fail or t == agent.episode_T - 1) else 1.)
            t_times.append(t)
            print(test, not fail, t)
            # if config.save_episode_data:
            #     agent.save_episode_data('{}_trial_{}_ep_{}_test_{}'.format(fname_pre, trial, ep, test))

        print(np.mean(t_success))
        episode_test_times.append(t_times)
        episode_test_success.append(t_success)

        while ep < args.nepisodes:
            fail, t = agent.do_episode(action_noise=True)

            if args.do_online_learning:
                agent.store_episode_data()

            if t < 5:
                continue

            #if config.use_online_GP:
            ##    agent.model.reset_model()

            ep += 1

            # if config.save_episode_data:
            #     agent.save_episode_data('{}_trial_{}_ep_{}'.format(fname_pre, trial, ep))

            episode_train_success.append(0. if (fail or t == agent.episode_T - 1) else 1.)
            print(episode_train_success[-1])

            print(ep)

            # Retrain online learned transition distribution
            train_interval = 1
            if (ep <= args.nepisodes) and (ep % train_interval == 0):
                if args.do_online_learning:
                    try:
                        agent.train_on_episode()
                    except Exception as e:
                        print(e)
                        print('failed training... retrying episode')
                        episode_train_success = episode_train_success[:-1]
                        ep -= 1
                        continue

                t_success = []
                for test in range(20):
                    fail, t = agent.do_episode()
                    t_success.append(0. if (fail or t == agent.episode_T-1) else 1.)
                    t_times.append(t)
                    if args.do_online_learning:
                        print(test, not fail, t)
                    # if config.save_episode_data:
                    #     agent.save_episode_data('{}_trial_{}_ep_{}_test_{}'.format(fname_pre, trial, ep, test))

                print(np.mean(t_success))
                episode_test_success.append(t_success)
                episode_test_times.append(t_times)

                torch.cuda.empty_cache()

        test_success.append(episode_test_success)
        train_success.append(episode_train_success)
        test_times.append(episode_test_times)
        trial += 1
        agent.model_lib['cartpole'].saved_data = None
        agent.model_lib['cartpole'].trained = False

        train_success_npy = np.asarray(train_success)
        test_success_npy = np.asarray(test_success)
        test_times_npy = np.asarray(test_times)
        #print(np.mean(test_success, axis=0))
        # results = dict()
        # from scipy.io import savemat
        # results['train_s'] = train_success_npy
        # results['test_s'] = test_success_npy
        # results['test_t'] = test_times_npy
        # savemat('{}_trial_{}.mat'.format(config.episode_fname, trial), results)

        agent.model_lib['cartpole'].trans_dist.save_model('model_w_GP')
        print('Mean training sucess rate over trials')
        print(np.mean(train_success_npy, axis=0))
        print('Mean testing success rate over trials')
        print(np.mean(np.mean(test_success_npy, axis=0), axis=1))
        print('Testing std rate over trials')
        print(np.std(np.mean(test_success_npy, axis=0), axis=1))
        print(test_times_npy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task",
                        action='store',
                        type=str,
                        choices=["conkers"],
                        help="Name of the folder into which to put dataset",
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
                        help="Models to be included in simple model library",
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
                        help="Whether to update transition models online",
                        dest="do_online_learning")

    args = parser.parse_args()

    main(args)
