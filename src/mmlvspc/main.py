import argparse
from arm_pytorch_utilities.rand import seed
from datetime import datetime
import logging
import numpy as np
import os
from src.agents import ConkersAgent
from src.config import SimpModLibConfig
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

    # Create lib config object
    lib_cfg = SimpModLibConfig()

    # Create list of dict containing loadable pt model names of used models using lib config
    load_name_lst = []
    for smodel in args.models:
        load_name_lst.append(lib_cfg[smodel])

    mylib = SimpModLib(load_name_lst)

    if task_name == 'conkers':
        agent = ConkersAgent(smodel_lib=mylib)
    else:
        raise NotImplementedError

    logging.info("Created agent for task {0}".format(task_name))

    train_success = []
    test_success = []
    test_times = []
    train_times = []
    trial = 0

    while trial < args.ntrials:
        agent.model.transition.reset_params()
        agent.CostFn.iter = 0
        if config.use_online_GP:
            agent.model.reset_model()
            agent.CostFn.uncertainty_cost = 0.0

        #agent.load_data()
        #agent.train_on_episode()

        ep = 0
        episode_train_times = []
        episode_test_times = []
        episode_train_success = []
        episode_test_success = []

        if trial > 0:
            print('trial: {} '.format(trial))
            print(np.mean(np.asarray(train_success), axis=0))

        t_success = []
        t_times = []
        if config.test_first:
            for test in range(config.num_tests):
                print_memory_usage()
                fail, t = agent.do_episode()
                t_success.append(0. if (fail or t == config.episode_T - 1) else 1.)
                t_times.append(t)
                print(test, not fail, t)
                if config.save_episode_data:
                    agent.save_episode_data('{}_trial_{}_ep_{}_test_{}'.format(fname_pre, trial, ep, test))

        print(np.mean(t_success))
        episode_test_times.append(t_times)
        episode_test_success.append(t_success)

        while ep < config.num_episodes:
            print_memory_usage()
            fail, t = agent.do_episode(action_noise=True)
            if config.do_online_learning:
                agent.store_episode_data()
            if t < 5:
                continue

            #if config.use_online_GP:
            ##    agent.model.reset_model()

            ep += 1

            if config.save_episode_data:
                agent.save_episode_data('{}_trial_{}_ep_{}'.format(fname_pre, trial, ep))

            episode_train_success.append(0. if (fail or t == config.episode_T-1) else 1.)
            print(episode_train_success[-1])
            print_memory_usage()
            print(ep)
            if (ep <= config.num_episodes) and (ep % config.train_interval == 0):
                print_memory_usage()
                if config.do_online_learning:
                    try:
                        agent.train_on_episode()
                    except Exception as e:
                        print(e)
                        print('failed training... retrying episode')
                        episode_train_success = episode_train_success[:-1]
                        ep -= 1
                        continue

                t_success = []
                for test in range(config.num_tests):
                    print_memory_usage()
                    fail, t = agent.do_episode()
                    t_success.append(0. if (fail or t == config.episode_T-1) else 1.)
                    t_times.append(t)
                    if config.do_online_learning:
                        print(test, not fail, t)
                    if config.save_episode_data:
                        agent.save_episode_data('{}_trial_{}_ep_{}_test_{}'.format(fname_pre, trial, ep, test))

                print(np.mean(t_success))
                episode_test_success.append(t_success)
                episode_test_times.append(t_times)

                torch.cuda.empty_cache()

        test_success.append(episode_test_success)
        train_success.append(episode_train_success)
        test_times.append(episode_test_times)
        trial += 1
        agent.model.saved_data = None
        agent.model.trained = False

        train_success_npy = np.asarray(train_success)
        test_success_npy = np.asarray(test_success)
        test_times_npy = np.asarray(test_times)
        #print(np.mean(test_success, axis=0))
        results = dict()
        from scipy.io import savemat
        results['train_s'] = train_success_npy
        results['test_s'] = test_success_npy
        results['test_t'] = test_times_npy
        savemat('{}_trial_{}.mat'.format(config.episode_fname, trial), results)

        agent.model.save_model('model_w_GP')
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
                        help="Number of trials for mm-lvspc",
                        dest="seed",
                        metavar="seed")

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

    args = parser.parse_args()

    main(args)
