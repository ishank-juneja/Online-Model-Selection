import random
from src.models.UKVAE import UnscentedKalmanVariationalAutoencoder
import numpy as np
from src.results_dir_manager import ResultDirManager
import matplotlib.patches as patches
import argparse
from src.pendulum_analogy_config import Config
from matplotlib import pyplot as plt
from random import sample
from arm_pytorch_utilities.rand import seed
seed(0)


def get_aleatoric_uncertainty(z_std):
    return np.mean(z_std, axis=0)


# def get_epistemic_uncertainty(z_mu):
#     return np.sqrt(np.sum(np.square(z_mu - np.mean(z_mu, axis=0)), axis=0))
def get_epistemic_uncertainty(z_mu):
    return np.std(z_mu, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Name of model being tested
    parser.add_argument("--cnn-name", type=str)
    # Whether to combine together uncertainties
    args = parser.parse_args()
    # Get folder names of test trajectories
    dir_manager = ResultDirManager()
    dir_manager.add_location('cartpole_data', 'data/MujocoCartpole-v0/')
    dir_manager.add_location('fixed_cartpole_data', 'data/MujocoCartpole-v0_fixed_geom/')
    dir_manager.add_location('conkers_test_time_data', 'data/LVSPC-tests/')
    cartpole_test_traj_folders = dir_manager.scrape_loc_for_prefix('cartpole_data', 'test_traj_*')
    fixed_cartpole_test_traj_folders = dir_manager.scrape_loc_for_prefix('fixed_cartpole_data', 'test_traj_*')
    conkers_test_traj_folders = dir_manager.scrape_loc_for_prefix('conkers_test_time_data', 'test_traj_*')

    # create a model for testing
    model_name = args.cnn_name
    # model_name = 'model_conkers_Feb05_13-33-50'
    # Create pendulum analogy config object
    config = Config()
    myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name=model_name)

    myensemble.encoder.cuda()

    fig1 = plt.figure(figsize=(15, 10))
    ax1 = fig1.subplots(2, 3)

    # Cartpole Data
    # cartpole_epistemic = []
    # cartpole_aleatoric = []
    # cartpole_total = []
    # for idx, dir_path in enumerate(cartpole_test_traj_folders):
    #     print("Processing Test Traj {0}".format(idx + 1))
    #     obs_frame_file_names = dir_manager.list_dir_objects(dir_path, pattern='observation_step_*.npy',
    #                                                         return_sorted=True)
    #     for frame_file in obs_frame_file_names:
    #         test_frame = np.load(frame_file)
    #         mu, stddev = myensemble.encode_single_observation(test_frame)
    #         mu_np = mu.cpu().detach().numpy()[0].reshape(10, 3)
    #         stddev_np = stddev.cpu().detach().numpy()[0].reshape(10, 3)
    #         alea = get_aleatoric_uncertainty(stddev_np)
    #         epi = get_epistemic_uncertainty(mu_np)
    #         total = np.sqrt(np.square(alea) + np.square(epi))
    #         cartpole_aleatoric.append(alea.sum())
    #         cartpole_epistemic.append(epi.sum())
    #         cartpole_total.append(total.sum())
    # # plt.hist(cartpole_epistemic, 100)
    # # plt.hist(cartpole_aleatoric, 100)
    # ax[0].hist(cartpole_total, 100, density=True, alpha=0.35)
    # ax[1].hist(cartpole_aleatoric, 100, density=True, alpha=0.35)
    # ax[2].hist(cartpole_epistemic, 100, density=True, alpha=0.35)

    # Fixed Cartpole Data
    cartpole_epistemic = []
    cartpole_aleatoric = []
    cartpole_total = []
    # For a quick viz
    # fixed_cartpole_test_traj_folders = random.sample(fixed_cartpole_test_traj_folders, 100)
    for idx, dir_path in enumerate(fixed_cartpole_test_traj_folders):
        print("Processing Test Traj {0}".format(idx + 1))
        obs_frame_file_names = dir_manager.list_dir_objects(dir_path, pattern='observation_step_*.npy',
                                                            return_sorted=True)
        for frame_file in obs_frame_file_names:
            test_frame = np.load(frame_file)
            mu, stddev = myensemble.encode_single_observation(test_frame)
            mu_np = mu.cpu().detach().numpy()[0].reshape(10, 3)
            stddev_np = stddev.cpu().detach().numpy()[0].reshape(10, 3)
            alea = get_aleatoric_uncertainty(stddev_np)
            epi = get_epistemic_uncertainty(mu_np)
            total = np.sqrt(np.square(alea) + np.square(epi))
            cartpole_aleatoric.append(alea.sum())
            cartpole_epistemic.append(epi.sum())
            cartpole_total.append(total.sum())
    # plt.hist(cartpole_epistemic, 100)
    # plt.hist(cartpole_aleatoric, 100)
    ax1[0][0].hist(cartpole_total, 100, density=True, alpha=0.35)
    ax1[0][1].hist(cartpole_aleatoric, 100, density=True, alpha=0.35)
    ax1[0][2].hist(cartpole_epistemic, 100, density=True, alpha=0.35)
    ax1[1][0].hist(cartpole_total, 50, alpha=0.35, density=True,  range=(0.4, 1.25))
    ax1[1][1].hist(cartpole_aleatoric, 50, alpha=0.35, density=True,  range=(0.4, 1.25))
    ax1[1][2].hist(cartpole_epistemic, 50, alpha=0.35, density=True,  range=(0.4, 1.25))
    print(max(cartpole_total))
    # Conkers Data
    conkers_epistemic = []
    conkers_aleatoric = []
    conkers_total = []
    # Sample from folders to reduce data-points
    conkers_test_traj_folders = sample(conkers_test_traj_folders, len(fixed_cartpole_test_traj_folders))
    for idx, dir_path in enumerate(conkers_test_traj_folders):
        print("Processing Test Traj {0}".format(idx + 1))
        obs_frame_file_names = dir_manager.list_dir_objects(dir_path, pattern='observation_*.npy',
                                                            return_sorted=True)
        for frame_file in obs_frame_file_names:
            test_frame = np.load(frame_file)
            mu, stddev = myensemble.encode_single_observation(test_frame)
            mu_np = mu.cpu().detach().numpy()[0].reshape(10, 3)
            stddev_np = stddev.cpu().detach().numpy()[0].reshape(10, 3)
            alea = get_aleatoric_uncertainty(stddev_np)
            epi = get_epistemic_uncertainty(mu_np)
            total = np.sqrt(np.square(alea) + np.square(epi))
            conkers_aleatoric.append(alea.sum())
            conkers_epistemic.append(epi.sum())
            conkers_total.append(total.sum())
    # plt.hist(conkers_epistemic, 100)
    # plt.hist(conkers_aleatoric, 100)
    ax1[0][0].hist(conkers_total, 100, density=True, alpha=0.35)
    ax1[0][1].hist(conkers_aleatoric, 100, density=True, alpha=0.35)
    ax1[0][2].hist(conkers_epistemic, 100, density=True, alpha=0.35)
    ax1[1][0].hist(conkers_total, 50, alpha=0.35, density=True,  range=(0.4, 1.25))
    ax1[1][1].hist(conkers_aleatoric, 50, alpha=0.35, density=True,  range=(0.4, 1.25))
    ax1[1][2].hist(conkers_epistemic, 50, alpha=0.35, density=True,  range=(0.4, 1.25))
    print(max(conkers_total))

    fig1.legend(['Cartpole', 'Conkers'])
    fig1.suptitle('Perception Uncertainty Distributions as Histograms')
    for idx in range(3):
        # Need to create a fresh Rectangle patch for every subplot
        rect = patches.Rectangle((0.4, 0), 0.85, 1, linewidth=1, edgecolor='r', facecolor='none')
        ax1[0][idx].add_patch(rect)

    for idx in range(2):
        ax1[idx][0].set_title('Total Uncertainty PDF')
        ax1[idx][1].set_title('Aleatoric PDF')
        ax1[idx][2].set_title('Epistemic PDF')
    # ax1[0].set_title('Total Uncertainty PDF')
    # ax1[1].set_title('Aleatoric PDF')
    # ax1[2].set_title('Epistemic PDF')
    fig1.savefig('results/cartpole_perception_histogram.png')
    # fig2.savefig('results/cartpole_perception_histogram_tail.png')
    # plt.show()
    
    # # Conkers test time data
    # cartpole_stddev_values = []
    # for dir_path in conkers_test_traj_folders:
    #     obs_frame_file_names = dir_manager.list_dir_objects(dir_path, pattern='observation_*.npy',
    #                                                         return_sorted=True)
    #     for frame_file in obs_frame_file_names:
    #         frame = np.load(frame_file)
    #         print(frame.shape)


    # # create a model for testing
    # model_name = args.cnn_name
    # # model_name = 'model_conkers_Feb05_13-33-50'
    #
    # test_data = np.load(cartpole_test_observations)
    # test_states = np.load(cartpole_test_states)
    #
    #
    # test_data = np.load(ball_test_observations)
    # test_states = np.load(ball_test_states)
    #
    # # Create pendulum analogy config object
    # config = Config()
    #
    # N, T, _ = test_states.shape
    # nstates = config.observation_dimension
    # myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name=model_name)
    # if args.combine:
    #     myensemble.test = True
    # else:
    #     myensemble.test = False
    # # Use CPU at test time
    # myensemble.encoder.cuda()
    #
    # avg_stddev = 0.0
    # avg_rmse = 0.0
    #
    # # Preallocate arrays of the size of datasets
    #
    #
    # # Use fewer trajectories
    # N = 5
    # for idx in range(N):
    #     print("Starting traj {0}".format(idx + 1))
    #     for jdx in range(T):
    #         test_obs = test_data[idx, jdx, :, :, :]
    #         state_label = test_states[idx, jdx, :]
    #         mu, stddev = myensemble.encode_single_observation(test_obs)
    #         avg_rmse += np.sqrt(np.square(mu.cpu().detach().numpy()[0] - state_label))
    #         if not args.combine:
    #             print(mu.reshape(10, nstates))
    #             print(stddev.reshape(10, nstates))
    #         else:
    #             print(mu)
    #             print(stddev)
    #         avg_stddev += stddev.sum()
    #         # plt.imshow(test_obs)
    #         # plt.show()
    # print("Average RMSE between observable part of label and prediction {0}".format(avg_rmse / (N * T)))
    # print("Average std dev for observations {0}".format(avg_stddev / (N * T)))
