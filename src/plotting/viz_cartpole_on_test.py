import numpy as np
import matplotlib.pyplot as plt
import glob
import subprocess
import os
import argparse
from src.models.UKVAE import UnscentedKalmanVariationalAutoencoder
from src.results_dir_manager import ResultDirManager
from src.pendulum_analogy_config import Config
STDDEV_SCALE = 2


def get_aleatoric_uncertainty(z_std):
    return np.mean(z_std, axis=0)


def get_epistemic_uncertainty(z_mu):
    return np.std(z_mu, axis=0)


def cleanup_frames_from_disk(path_where_frames_dumped):
    # Get rid of temp png files
    for file_name in glob.glob(os.path.join(path_where_frames_dumped, '*.png')):
        os.remove(file_name)
    return

# stddevs is a concatenated array of all 3 kinds of uncertanties
def save_animation_frames(imgs, means, stddevs, save_dir):
    fig = plt.figure(figsize=(20, 5))
    ax = fig.subplots(1, 4)
    # Plot a separate plot for every frame in the trajectory
    for idx in range(len(imgs)):
        # For iterating over the 3 mean + uncertainty plots
        for jdx in range(1, 4):
            # Imshow clears axes automatically
            ax[jdx].plot(means[:, 0])
            ax[jdx].plot(means[:, 1])
            ax[jdx].plot(means[:, 2])
            std_dev_scale = STDDEV_SCALE
            if jdx == 3:
                std_dev_scale = 2 * STDDEV_SCALE
            ax[jdx].fill_between(np.arange(len(means)), means[:, 0] - std_dev_scale * stddevs[:, jdx - 1, 0],
                               means[:, 0] + std_dev_scale * stddevs[:, jdx - 1, 0], alpha=0.4)
            ax[jdx].fill_between(np.arange(len(means)), means[:, 1] - std_dev_scale * stddevs[:, jdx - 1, 1],
                               means[:, 1] + std_dev_scale * stddevs[:, jdx - 1, 1], alpha=0.4)
            ax[jdx].fill_between(np.arange(len(means)), means[:, 2] - std_dev_scale * stddevs[:, jdx - 1, 2],
                               means[:, 2] + std_dev_scale * stddevs[:, jdx - 1, 2], alpha=0.4)
            ax[jdx].axvline(idx, color='r')
            ax[jdx].set_ylim([-1.7, 1.7])
            ax[jdx].set_aspect(1.0 / ax[jdx].get_data_ratio(), adjustable='box')
        ax[1].set_title('Total Uncertainty')
        ax[2].set_title('Aleatoric')
        ax[3].set_title('Epistemic')
        # The image frame goes in the first column
        ax[0].imshow(imgs[idx])
        fig.legend(['xcart', 'xmass', 'ymass'], loc='upper right')
        fig.suptitle('Uncertainty Evolution for Cartpole Perception', size=18)
        # Save temporary png file frames in home folder
        fig.savefig(os.path.join(save_dir, "file{0:02d}.png".format(idx + 1)))
        # Clear all axes for next frame
        for jdx in range(4):
            ax[jdx].cla()
    fig.clear()
    plt.close()


def save_video(video_path, frames_dir):
    frames_path_pattern = os.path.join(frames_dir, 'file%02d.png')
    subprocess.call([
        'ffmpeg', '-framerate', '2', '-i', frames_path_pattern, '-r', '30', '-pix_fmt', 'yuv420p',
        video_path
    ])
    cleanup_frames_from_disk(frames_dir)


def save_gif(gif_path, frames_dir):
    print("Baking your GIF ... ")
    frames_path_pattern = os.path.join(frames_dir, '*.png')
    subprocess.call([
        'convert', '-delay', '60', '-loop', '0', frames_path_pattern, gif_path
    ])
    cleanup_frames_from_disk(frames_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-images", action="store_true",
                        help="save frames encountered from test cases at a location as .npy files")
    parser.add_argument("--data-loc", type=str, default='../MM-LVSPC/data/LVSPC-tests')
    parser.add_argument("--make-gif", action="store_true")
    parser.add_argument("--make-mp4", action="store_true")
    parser.add_argument("--cnn-name", type=str)
    args = parser.parse_args()
    cnn_name = args.cnn_name.split('.')[0]

    dir_manager = ResultDirManager()
    dir_manager.add_location('test_data', 'data/online/')
    dir_manager.add_location('vid_results', 'results/videos/{0}'.format(cnn_name))

    if args.save_images:
        dir_manager.add_location('dataset', args.data_loc)

    # Glob .npz paths with matching prefixes
    episode_data_files = dir_manager.scrape_loc_for_prefix('test_data', 'test_with_gp_2_trial_*.npz')

    # create a model for testing
    model_name = args.cnn_name
    # model_name = 'model_conkers_Feb05_13-33-50'
    # Create pendulum analogy config object
    config = Config()
    myensemble = UnscentedKalmanVariationalAutoencoder(config, load_name=model_name)
    myensemble.encoder.cuda()

    # Find predicted mean/uncertainty for loaded perception model for all these image trajectories that were
    # generated online
    if len(episode_data_files) > 100:
        first_few = episode_data_files[0:50]
        last_few = episode_data_files[0:50]
        episode_data_files = first_few + last_few
    for idx, file_path in enumerate(episode_data_files):
        episode_data = np.load(file_path)
        # Extract frames from loaded file
        frames = episode_data['arr_3']
        # Allocate arrays to hold mean and stddev predictions
        traj_mu = np.zeros((len(frames), 3), dtype=np.float64)
        # 3 kinds of uncertainties: total, alea, api
        traj_stddev = np.zeros((len(frames), 3, 3), dtype=np.float64)
        for idx in range(len(frames)):
            mu, stddev = myensemble.encode_single_observation(frames[idx])
            # Reshape predicted mu and stddev into right format
            mu_np = mu.cpu().detach().numpy()[0].reshape(10, 3)
            traj_mu[idx, :] = mu_np.mean(axis=0)
            traj_stddev[idx, 2, :] = get_epistemic_uncertainty(mu_np)
            stddev_np = stddev.cpu().detach().numpy()[0].reshape(10, 3)
            traj_stddev[idx, 1, :] = get_aleatoric_uncertainty(stddev_np)
            traj_stddev[idx, 0, :] = np.sqrt(np.square(traj_stddev[idx, 1, :]) + np.square(traj_stddev[idx, 2, :]))
        dir_save = dir_manager.get_abs_path('vid_results')
        # Get path to .mp4 file
        save_animation_frames(frames, traj_mu, traj_stddev, dir_save)
        if args.make_mp4:
            vid_path = dir_manager.next_path('vid_results', '{0}_test'.format(cnn_name), postfix='%s.mp4')
            save_video(vid_path, dir_save)
        if args.make_gif:
            gif_path = dir_manager.next_path('vid_results', '{0}_test'.format(cnn_name), postfix='%s.gif')
            save_gif(gif_path, dir_save)
        if args.save_images:
            traj_dir = dir_manager.make_fresh_dir('dataset', 'test_traj_{0}'.format(idx + 1))
            for jdx in range(len(frames)):
                frame_path = os.path.join(traj_dir, 'observation_{0}.npy'.format(jdx + 1))
                np.save(frame_path, frames[jdx, :, :, :])
