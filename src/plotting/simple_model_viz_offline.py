import matplotlib.pyplot as plt
import numpy as np
import os
from src.plotting.simple_model_viz import SimpleModViz
from typing import Callable, List
STDDEV_SCALE = 1.0


class SMZOffline(SimpleModViz):
    """
    Simple Model Viz functionality with methods for doing offline perception tests
    """
    def __init__(self, simp_model: str, vel_as_color: bool = False):
        super(SMZOffline, self).__init__(simp_model, vel_as_color)

    @staticmethod
    def save_frames(frames: list, save_dir: str):
        """
        Simply saves the passed frames as is
        Used for plotting raw GIFs
        :param frames: List of np.ndarrays containing images
        :param save_dir: location where to save as png files
        :return:
        """
        fig = plt.figure(figsize=(5, 5))
        ax = fig.subplots(1, 1)
        # Plot a separate plot for every frame in the trajectory
        for idx in range(len(frames)):
            # The image frame goes in the first column
            ax.imshow(frames[idx])
            # fig.suptitle('Trajectory', size=16)
            # Save temporary png file frames in home folder
            fig.savefig(os.path.join(save_dir, "file{0:03d}.png".format(idx + 1)))
            ax.clear()
        fig.clear()
        plt.close()
        return

    def save_train_animation_frames(self, imgs: list, save_dir: str, true_states: list = None,
                                    overlay_states: bool = False):
        """
        True states label and overlay are optional, can also be used to convert sequence of frames to GIF
        :param imgs:
        :param save_dir:
        :param true_states:
        :param overlay_states:
        :return:
        """
        fig = plt.figure(figsize=(5, 5))
        ax = fig.subplots(1, 1)
        # Plot a separate plot for every frame in the trajectory
        for idx in range(len(imgs)):
            # The image frame goes in the first column
            ax.imshow(imgs[idx])
            if overlay_states:
                self.overlay_state(img_axis=ax, observed_state=true_states[idx], alpha=1.0, color='g')
            fig.suptitle('{0} Trajectory'.format(self.simp_model.capitalize()), size=16)
            # Save temporary png file frames in home folder
            fig.savefig(os.path.join(save_dir, "file{0:03d}.png".format(idx + 1)))
            ax.clear()
        fig.clear()
        plt.close()
        return

    def get_viz_function(self, ver: str) -> Callable:
        if ver == 'ver1':
            return self.save_test_animation_frames_ver1
        elif ver == 'ver2':
            return self.save_test_animation_frames_ver2
        elif ver == 'ver3':
            return self.save_test_animation_frames_ver3
        elif ver == 'ver4':
            return self.save_test_animation_frames_ver4
        else:
            raise NotImplementedError

    def save_test_animation_frames_ver4(self, masked_frames: List[np.ndarray], traj_mu: np.ndarray,
                                        traj_stddev: np.ndarray, traj_mu_full: np.ndarray, save_dir: str):
        """
        Save animation test frames as image files
        Identical to ver3 except only the frame at time t is plotted and frame at t-1 though in masked_frames
        is ommited
        ** Only intended for use with stacked frames **
        :param masked_frames:
        :param traj_mu: Combined mean of the encoder ensemble
        :param traj_stddev: stddevs is a concatenated array of all 3 (total, alea, epi) kinds of uncertanties
        :param traj_mu_full: mu_np: Individual means of the ensemble members
        :param save_dir: Directory into which to save results
        :return:
        """
        if self.nframes != 2:
            raise ValueError("ver4 is not intended for use with single frames")

        # Number of axes in plot
        naxes = 3
        fig = plt.figure(figsize=(5 * naxes, 5))
        ax = fig.subplots(1, naxes)

        # Get number of position related states for current simple model
        nqpos = self.get_nqpos_states()

        # Scale up factor while plotting standard deviation to better see the variation
        std_dev_scale = STDDEV_SCALE

        # Plot a separate plot for every frame in the trajectory
        #  While plotting use axis 0 for images and axes 1 and 2 for state + state uncertainty
        for idx in range(len(masked_frames)):
            # First make the plot for position related states
            for kdx in range(traj_mu.shape[1]):
                if kdx < nqpos:
                    jdx = 1
                else:
                    jdx = 2
                # Imshow clears axes automatically
                ax[jdx].plot(traj_mu[:, kdx])
                # Only the first column of stddev (which is total epi + alea) is used rest are ignored in ver3
                ax[jdx].fill_between(np.arange(len(traj_mu)), traj_mu[:, kdx] - std_dev_scale * traj_stddev[:, 0, kdx],
                                     traj_mu[:, kdx] + std_dev_scale * traj_stddev[:, 0, kdx], alpha=0.4,
                                     label='_nolegend_')

            # Both axis 1 and 2 have a moving red carot line and both need their aspect ration set the same way
            for jdx in range(1, 3):
                ax[jdx].axvline(idx, color='r')
                # ax[jdx].set_aspect(1.0 / ax[jdx].get_data_ratio(), adjustable='box')

            # Limits for the position related plots are smaller than those for the velocity related plots
            ax[1].set_ylim([-2.0, 2.0])
            ax[2].set_ylim([-8.0, 8.0])

            # Plot only the image at time t in the first column ommiting the image at time t-1
            ax[0].imshow(masked_frames[idx][:, self.config.imsize:, :])

            # We plot the prediction by every ensemble member separately with low alpha and red color
            # Depends on number of num ensembles
            for jdx in range(traj_mu_full.shape[1]):
                self.overlay_state(ax[0], traj_mu_full[idx][jdx], alpha=0.2, color='r', display_t_only=True)

            # More prominent overlay with alpha=1.0 and green for their combined result
            self.overlay_state(ax[0], traj_mu[idx], display_t_only=True)

            # Axis 1 gets legend for position
            self.add_legend_axis(ax[1], ispos=True)
            # Axis 2 gets legend for velocity
            self.add_legend_axis(ax[2], ispos=False)

            fig.suptitle("{0} Perception".format(self.simp_model.capitalize()), size=18)

            # Save temporary png file frames in designated folder
            fig.savefig(os.path.join(save_dir, "file{0:03d}.png".format(idx + 1)))

            # Clear all axes for next frame
            for jdx in range(naxes):
                ax[jdx].cla()

        fig.clear()
        plt.close()

    def save_test_animation_frames_ver3(self, masked_frames: List[np.ndarray], traj_mu: np.ndarray,
                                        traj_stddev: np.ndarray, traj_mu_full: np.ndarray, save_dir: str):
        """
        Save animation test frames as image files
        In this version position and velocity related states are plotted in separate plots
        **Can handle both 2 Stacked frames and single frames**
        :param masked_frames:
        :param traj_mu: Combined mean of the encoder ensemble
        :param traj_stddev: stddevs is a concatenated array of all 3 (total, alea, epi) kinds of uncertanties
        :param traj_mu_full: mu_np: Individual means of the ensemble members
        :param save_dir: Directory into which to save results
        :return:
        """
        # Number of axes in plot
        naxes = 3
        fig = plt.figure(figsize=(5 * naxes, 5))
        ax = fig.subplots(1, naxes)

        # get number of position related states for current simple model
        nqpos = self.get_nqpos_states()

        # Scale up factor while plotting standard deviation to better see the variation
        std_dev_scale = STDDEV_SCALE

        # Plot a separate plot for every frame in the trajectory
        #  While plotting use axis 0 for images and axes 1 and 2 for state + state uncertainty
        for idx in range(len(masked_frames)):
            # First make the plot for position related states
            for kdx in range(traj_mu.shape[1]):
                if kdx < nqpos:
                    jdx = 1
                else:
                    jdx = 2
                # Imshow clears axes automatically
                ax[jdx].plot(traj_mu[:, kdx])
                # Only the first column of stddev (which is total epi + alea) is used rest are ignored in ver3
                ax[jdx].fill_between(np.arange(len(traj_mu)), traj_mu[:, kdx] - std_dev_scale * traj_stddev[:, 0, kdx],
                                     traj_mu[:, kdx] + std_dev_scale * traj_stddev[:, 0, kdx], alpha=0.4,
                                     label='_nolegend_')

            # Both axis 1 and 2 have a moving red carot line and both need their aspect ration set the same way
            for jdx in range(1, 3):
                ax[jdx].axvline(idx, color='r')
                # ax[jdx].set_aspect(1.0 / ax[jdx].get_data_ratio(), adjustable='box')

            # Limits for the position related plots are smaller than those for the velocity related plots
            ax[1].set_ylim([-2.0, 2.0])
            ax[2].set_ylim([-8.0, 8.0])

            # The image frame and masked image go in the first column
            ax[0].imshow(masked_frames[idx])

            # We plot the prediction by every ensemble member separately with low alpha and red color
            # Depends on number of num ensembles
            for jdx in range(traj_mu_full.shape[1]):
                self.overlay_state(ax[0], traj_mu_full[idx][jdx], alpha=0.2, color='r')

            # More prominent overlay with alpha=1.0 and green for their combined result
            self.overlay_state(ax[0], traj_mu[idx])

            # Axis 1 gets legend for position
            self.add_legend_axis(ax[1], ispos=True)
            # Axis 2 gets legend for velocity
            self.add_legend_axis(ax[2], ispos=False)

            fig.suptitle("{0} Perception".format(self.simp_model.capitalize()), size=18)

            # Save temporary png file frames in designated folder
            fig.savefig(os.path.join(save_dir, "file{0:03d}.png".format(idx + 1)))

            # Clear all axes for next frame
            for jdx in range(naxes):
                ax[jdx].cla()

        fig.clear()
        plt.close()

    def save_test_animation_frames_ver2(self, masked_frames: List[np.ndarray], traj_mu: np.ndarray,
                                        traj_stddev: np.ndarray, traj_mu_full: np.ndarray, save_dir: str):
        """
        Save animation test frames as image (png) files using mpl
        :param masked_frames: List of masked frames (np.ndarrays)
        :param traj_mu: Combined mean of the encoder ensemble
        :param traj_stddev: An array of total uncertainty only
        :param traj_mu_full: mu_np: Individual means of the ensemble members
        :param save_dir: Directory into which to save results
        :return:
        """
        # Number of axes in plot
        naxes = 2
        fig = plt.figure(figsize=(5 * naxes, 5))
        ax = fig.subplots(1, naxes)
        # Indices 0, 1 are for images, and 2-4 are for state data
        # Plot a separate plot for every frame in the trajectory
        for idx in range(len(masked_frames)):
            std_dev_scale = STDDEV_SCALE
            for kdx in range(traj_mu.shape[1]):
                # Imshow clears axes automatically
                ax[1].plot(traj_mu[:, kdx])
                ax[1].fill_between(np.arange(len(traj_mu)), traj_mu[:, kdx] - std_dev_scale * traj_stddev[:, kdx],
                                   traj_mu[:, kdx] + std_dev_scale * traj_stddev[:, kdx], alpha=0.4)
            ax[1].axvline(idx, color='r')
            ax[1].set_ylim([-1.7, 1.7])
            ax[1].set_aspect(1.0 / ax[1].get_data_ratio(), adjustable='box')
            # The image frame and masked image go in the first column
            ax[0].imshow(masked_frames[idx])
            # Depends on number of num ensembles
            for jdx in range(traj_mu_full.shape[1]):
                self.overlay_state(ax[0], traj_mu_full[idx][jdx], alpha=0.2, color='r')
            # More prominent overlay for their combined result
            self.overlay_state(ax[0], traj_mu[idx])
            self.add_legend_fig(fig)
            fig.suptitle('Uncertainty Evolution for Perception', size=18)
            # Save temporary png file frames in home folder
            fig.savefig(os.path.join(save_dir, "file{0:03d}.png".format(idx + 1)))
            # Clear all axes for next frame
            for jdx in range(naxes):
                ax[jdx].cla()
        fig.clear()
        plt.close()

    def save_test_animation_frames_ver1(self, masked_frames: list, traj_mu: np.ndarray,
                                        traj_stddev: np.ndarray, traj_mu_full: np.ndarray, save_dir: str):
        """
        Save animation test frames as image files
        :param masked_frames:
        :param traj_mu: Combined mean of the encoder ensemble
        :param traj_stddev: stddevs is a concatenated array of all 3 (total, alea, epi) kinds of uncertanties
        :param traj_mu_full: mu_np: Individual means of the ensemble members
        :param save_dir: Directory into which to save results
        :return:
        """
        # Number of axes in plot
        naxes = 4
        fig = plt.figure(figsize=(5 * naxes, 5))
        ax = fig.subplots(1, naxes)
        # Indices 0, 1 are for images, and 2-4 are for state data
        # Plot a separate plot for every frame in the trajectory
        for idx in range(len(masked_frames)):
            # For iterating over the 3 mean + uncertainty plots
            for jdx in range(1, naxes):
                std_dev_scale = STDDEV_SCALE
                # Last axis is for epistemic uncertainty which is scaled up for viz
                if jdx == naxes - 1:
                    std_dev_scale = 2 * STDDEV_SCALE
                for kdx in range(traj_mu.shape[1]):
                    # Imshow clears axes automatically
                    ax[jdx].plot(traj_mu[:, kdx])
                    ax[jdx].fill_between(np.arange(len(traj_mu)),
                                         traj_mu[:, kdx] - std_dev_scale * traj_stddev[:, jdx - 2, kdx],
                                         traj_mu[:, kdx] + std_dev_scale * traj_stddev[:, jdx - 2, kdx], alpha=0.4,
                                         label='_nolegend_')
                ax[jdx].axvline(idx, color='r')
                ax[jdx].set_ylim([-1.7, 1.7])
                ax[jdx].set_aspect(1.0 / ax[jdx].get_data_ratio(), adjustable='box')
            # The image frame and masked image go in the first column
            ax[0].imshow(masked_frames[idx])
            # Depends on number of num ensembles
            for jdx in range(traj_mu_full.shape[1]):
                self.overlay_state(ax[0], traj_mu_full[idx][jdx], alpha=0.2, color='r')
            # More prominent overlay for their combined result
            self.overlay_state(ax[0], traj_mu[idx])
            self.add_legend_fig(fig)
            fig.suptitle('Uncertainty Evolution for Perception', size=18)
            # Save temporary png file frames in home folder
            fig.savefig(os.path.join(save_dir, "file{0:03d}.png".format(idx + 1)))
            # Clear all axes for next frame
            for jdx in range(naxes):
                ax[jdx].cla()
        fig.clear()
        plt.close()

    @staticmethod
    def save_animation_frames_seg(self, before, after, save_dir: str):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.subplots(1, 2)
        # Plot a separate plot for every frame in the trajectory
        for idx in range(len(before)):
            ax[0].set_title('Before')
            ax[1].set_title('After')
            # The image frame goes in the first column
            ax[0].imshow(before[idx])
            ax[1].imshow(after[idx])
            fig.suptitle('Segmentation for Cartpoles on Kendama', size=18)
            # Save temporary png file frames in home folder
            fig.savefig(os.path.join(save_dir, "file{0:03d}.png".format(idx + 1)))
            # Clear all axes for next frame
            for jdx in range(2):
                ax[jdx].cla()
        fig.clear()
        plt.close()
