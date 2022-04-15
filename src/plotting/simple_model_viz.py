from math import atan2, sin, cos, pi
import matplotlib.pyplot as plt
import numpy as np
import os
from src.config import CartpoleConfig, BallConfig, DcartpoleConfig, DubinsConfig
from typing import Callable, List, Tuple
STDDEV_SCALE = 1.0


class SimpleModViz:
    """
    Visualization only simple model without gym environment
    """
    def __init__(self, simp_model: str):
        self.simp_model = simp_model
        if self.simp_model == 'ball':
            self.overlay_state = self.overlay_ball_state
            self.config = BallConfig()
        elif self.simp_model == 'cartpole':
            self.overlay_state = self.overlay_cartpole_state
            self.config = CartpoleConfig()
        elif self.simp_model == 'dcartpole':
            self.overlay_state = self.overlay_dcartpole_state
            self.config = DcartpoleConfig()
        elif self.simp_model == 'dubins':
            self.overlay_state = self.overlay_dubins_state
            self.config = DubinsConfig()
        else:
            raise NotImplementedError("Viz not implemented for simple model {0}".format(self.simp_model))
        # We deal with 1D actuate simple models that have a fixed cart/robot position in the screen
        # Cart y-position in world coordinates, manually carried over from the mujoco_carpole.xml file line
        #     <body name="cart" pos="0 0 0">
        self.carty = 0.0

        # Whether the viz works with single or stacked frames
        self.nframes = None

        # Delta t between consecutuve frames
        self.dt = None

    def set_nframes(self, nframes: int):
        """
        Number of stacked frames viz deals with
        :param nframes:
        :return:
        """
        self.nframes = nframes

    def set_delta_t(self, dt: float):
        """
        Set the simulation time step dt
        :param dt:
        :return:
        """
        self.dt = dt

    @staticmethod
    def save_frames(frames: list, save_dir: str):
        """
        Most boring of all the viz functions simple saves the frames as is
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

    def save_train_animation_frames(self, imgs: list, save_dir: str, true_states: list = None, overlay_states: bool = False):
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

    def get_nqpos_states(self) -> int:
        """
        Returns the number of static states (joint config positions) in the estimated state
        Static states then can be indexed as state[:nqpos], and
        velocity states (joint config velocities) can be indexed as [nqpos:]
        :return: nqpos
        """
        return self.config.nqpos

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
                    ax[jdx].fill_between(np.arange(len(traj_mu)), traj_mu[:, kdx] - std_dev_scale * traj_stddev[:, jdx - 2, kdx],
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

    def add_legend_axis(self, axis, ispos: bool):
        """
        Similar to add_legend_fig but works on a single axis of a subplot as opposed to entire axis
        :param axis: mpl axis
        :param ispos: Whether legend is for pos states or for vel states, True->pos, False->vel
        :return:
        """
        if self.simp_model == 'ball':
            if ispos:
                legend_str = ['x', 'y']
            else:
                legend_str = ['v_x', 'v_y']
        elif self.simp_model == 'cartpole':
            if ispos:
                legend_str = ['xcart', 'xmass', 'ymass']
            else:
                legend_str = ['vcart', 'vxmass', 'vymass']
        elif self.simp_model == 'dcartpole':
            if ispos:
                legend_str = ['xcart', 'xhinge', 'yhinge', 'xmass', 'ymass']
            else:
                legend_str = ['vcart', 'vxhinge', 'vyhinge', 'vxmass', 'vymass']
        elif self.simp_model == 'dubins':
            if ispos:
                legend_str = ['xcar', 'ycar', 'cos(theta)', 'sin(theta)']
            else:
                legend_str = ['Vcar']
        else:
            raise NotImplementedError("Legend for simp model {0} is not implemented".format(self.simp_model))
        axis.legend(legend_str, loc='upper right')

    # - - - - -  Depracated - - - -
    def add_legend_fig(self, fig):
        """
        Picks the appropriate legend string for an entire figure depending on what needs to be plotted
        :param fig: Full axis onto which plot figure
        :return:
        """
        if self.simp_model == 'ball':
            if self.nframes < 2:
                legend_str = ['x', 'y']
            else:
                legend_str = ['x', 'y', 'v_x', 'v_y']
        elif self.simp_model == 'cartpole':
            if self.nframes < 2:
                legend_str = ['xcart', 'xmass', 'ymass']
            else:
                legend_str = ['xcart', 'xmass', 'ymass', 'vcart', 'theta_dot']
        else:
            raise NotImplementedError("Legend for simp model {0} is not implemented".format(self.simp_model))
        fig.legend(legend_str, loc='upper right')
    # - - - - -  Depracated - - - -

    # stddevs is a concatenated array of all 3 kinds of uncertainties
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

    def overlay_ball_state(self, img_axis, observed_state, alpha=1.0, color='g', display_t_only: bool = False):
        """
        :param img_axis: mpl axis obj onto which to perform the overlay
        :param observed_state: current observed state
        :param alpha: opacity of state overlay
        :param color: color of state overlay
        :param display_t_only: Whether to only display the frame at time t (foregoing velocity prediction viz on images)
        :return:
        """
        # Cam matrix for projecting points from world to pixel space
        cam_mat = np.load(self.config.cam_mat_path)
        # Homogenous world coordinates for ball
        ball_world = np.array([observed_state[0], 0.0, observed_state[1], 1.0])
        ball_pixel = cam_mat @ ball_world
        # Divide by 2D homogenous scaling term
        ball_pixel = self.homogenous_to_regular_coordinates(ball_pixel)
        # Perform extra steps if concatenating 2 frames to overlay the full state with velocity
        if self.nframes == 2 and (not display_t_only):
            # Check if dt has been set, else can't deal with 2 frames
            if self.dt is None:
                raise ValueError("Cannot perform overlay on two states when dt has not been set")
            # Augment the x-axis pixel coordinates to correspond to the right half of 2-stacked together frames
            ball_pixel[0] += self.config.imsize
            # Compute the static portion (cart and mass coords) of the previous state based on current velocity
            #  formula for velocity is (x_t - x_{t-1}) / delta_t
            # Determine ball position at t-1
            prev_ball_world = np.array([observed_state[0] - observed_state[2] * self.dt, 0.0,
                                        observed_state[1] - observed_state[3] * self.dt, 1.0])
            prev_ball_pixel = cam_mat @ prev_ball_world
            prev_ball_pixel = self.homogenous_to_regular_coordinates(prev_ball_pixel)
            img_axis.scatter(prev_ball_pixel[0], prev_ball_pixel[1], marker='o', s=100, color=color, label='_nolegend_',
                             alpha=alpha)
        img_axis.scatter(ball_pixel[0], ball_pixel[1], marker='o', s=100, color=color, label='_nolegend_', alpha=alpha)
        return

    def overlay_cartpole_state(self, img_axis, observed_state, alpha=1.0, color='g', display_t_only: bool = False):
        """
        :param img_axis: mpl axis obj onto which to perform the overlay
        :param observed_state: current observed state
        :param alpha: opacity of state overlay
        :param color: color of state overlay
        :param display_t_only: Whether to only display the frame at time t (foregoing velocity prediction viz on images)
        :return:
        """
        # Cam matrix for projecting points from world to pixel space
        cam_mat = np.load(self.config.cam_mat_path)
        # Homogenous world coordinates for cart and mass
        self.carty = 0.0
        cart_world = np.array([observed_state[0], 0.0, self.carty, 1.0])
        mass_world = np.array([observed_state[1], 0.0, observed_state[2], 1.0])
        cart_pixel = cam_mat @ cart_world
        mass_pixel = cam_mat @ mass_world
        # Divide by 2D homogenous scaling term
        cart_pixel = self.homogenous_to_regular_coordinates(cart_pixel)
        mass_pixel = self.homogenous_to_regular_coordinates(mass_pixel)
        # Perform extra steps needed when dealing with 2 side by side frames as opposed to 1 frame
        if self.nframes == 2 and (not display_t_only):
            # Check if dt has been set, else can't deal with 2 frames
            if self.dt is None:
                raise ValueError("Cannot perform overlay on two states when dt has not been set")
            # Augment the x-axis pixel coordinates to correspond to the right half of 2-stacked together frames
            cart_pixel[0] += self.config.imsize
            mass_pixel[0] += self.config.imsize
            # Compute the static portion (cart and mass coords) of the previous state based on current velocity
            #  formula for velocity is (x_t - x_{t-1})/delta_t
            # Determine cart position at t-1
            prev_cart_world_x = observed_state[0] - observed_state[3] * self.dt
            prev_cart_world = np.array([prev_cart_world_x, 0.0, self.carty, 1.0])
            prev_cart_pixel = cam_mat @ prev_cart_world
            prev_cart_pixel = self.homogenous_to_regular_coordinates(prev_cart_pixel)
            # Compute length of the pole in meter, Use plen and angular velocity to determine mass position at {t-1}
            plen = self.euclidean_distance(cart_world, mass_world)

            # - - - - - - - - - - -
            # # Find current angular location
            # theta_cur = atan2(observed_state[1] - observed_state[0], observed_state[2])
            # # Theta previous (wrap around taken care of by trig functions subsequently)
            # theta based angular velocity replaced with linear velocities in state representation
            # theta_prev = theta_cur - observed_state[4] * self.dt
            # If using angular velocity can bring back above instead
            # theta_prev = theta_cur - theta_dot * self.dt
            # prev_mass_world = np.array([prev_cart_world_x + plen * sin(theta_prev), 0.0, self.carty + plen * cos(theta_prev), 1.0])
            # # - - - - - - - - - - -

            prev_mass_worldx = mass_world[0] - observed_state[4] * self.dt
            prev_mass_worldy = mass_world[2] - observed_state[5] * self.dt
            # prev_mass_worldx = observed_state[4]
            # prev_mass_worldy = observed_state[5]
            prev_mass_world = [prev_mass_worldx, 0.0, prev_mass_worldy, 1.0]

            prev_mass_pixel = cam_mat @ prev_mass_world
            prev_mass_pixel = self.homogenous_to_regular_coordinates(prev_mass_pixel)
            self.plot_dumbel(img_axis, (prev_cart_pixel[0], prev_cart_pixel[1]),
                             (prev_mass_pixel[0], prev_mass_pixel[1]), color=color, alpha=alpha)
        self.plot_dumbel(img_axis, (cart_pixel[0], cart_pixel[1]), (mass_pixel[0], mass_pixel[1]),
                         color=color, alpha=alpha)
        return

    def overlay_dcartpole_state(self, img_axis, observed_state, alpha=1.0, color='g', display_t_only: bool = False):
        """
        :param img_axis: mpl axis obj onto which to perform the overlay
        :param observed_state: current observed state
        :param alpha: opacity of state overlay
        :param color: color of state overlay
        :param display_t_only: Whether to only display the frame at time t (foregoing velocity prediction viz on images)
        :return:
        """
        # Cam matrix for projecting points from world to pixel space
        cam_mat = np.load(self.config.cam_mat_path)
        # Homogenous world coordinates for cart and mass
        self.carty = 0.0
        cart_world = np.array([observed_state[0], 0.0, self.carty, 1.0])
        hinge2_world = np.array([observed_state[1], 0.0, observed_state[2], 1.0])
        mass_world = np.array([observed_state[3], 0.0, observed_state[4], 1.0])
        cart_pixel = cam_mat @ cart_world
        hinge2_pixel = cam_mat @ hinge2_world
        mass_pixel = cam_mat @ mass_world
        # Divide by 2D homogenous scaling term
        cart_pixel = self.homogenous_to_regular_coordinates(cart_pixel)
        hinge2_pixel = self.homogenous_to_regular_coordinates(hinge2_pixel)
        mass_pixel = self.homogenous_to_regular_coordinates(mass_pixel)

        # Perform extra steps needed when dealing with 2 side by side frames as opposed to 1 frame
        if self.nframes == 2 and (not display_t_only):
            # Check if dt has been set, else can't deal with 2 frames
            if self.dt is None:
                raise ValueError("Cannot perform overlay on two states when dt has not been set")
            # Augment the x-axis pixel coordinates to correspond to the right half of 2-stacked together frames
            cart_pixel[0] += self.config.imsize
            hinge2_pixel[0] += self.config.imsize
            mass_pixel[0] += self.config.imsize

            # Compute the static portion (cart and mass coords) of the previous state based on current velocity
            #  formula for velocity is (x_t - x_{t-1})/delta_t
            # Determine cart position at t-1
            prev_cart_world_x = observed_state[0] - observed_state[5] * self.dt
            prev_cart_world = np.array([prev_cart_world_x, 0.0, self.carty, 1.0])
            prev_cart_pixel = cam_mat @ prev_cart_world
            prev_cart_pixel = self.homogenous_to_regular_coordinates(prev_cart_pixel)

            prev_hinge2_worldx = hinge2_world[0] - observed_state[6] * self.dt
            prev_hinge2_worldy = hinge2_world[2] - observed_state[7] * self.dt

            prev_hinge2_world = [prev_hinge2_worldx, 0.0, prev_hinge2_worldy, 1.0]
            prev_hinge2_pixel = cam_mat @ prev_hinge2_world
            prev_hinge2_pixel = self.homogenous_to_regular_coordinates(prev_hinge2_pixel)

            prev_mass_x = mass_world[0] - observed_state[8] * self.dt
            prev_mass_y = mass_world[2] - observed_state[9] * self.dt
            prev_mass_world = [prev_mass_x, 0.0, prev_mass_y, 1.0]
            prev_mass_pixel = cam_mat @ prev_mass_world
            prev_mass_pixel = self.homogenous_to_regular_coordinates(prev_mass_pixel)

            # Plot dumbel for link 1
            self.plot_dumbel(img_axis, (prev_cart_pixel[0], prev_cart_pixel[1]), (prev_hinge2_pixel[0], prev_hinge2_pixel[1]),
                             color=color, alpha=alpha)
            # Plot dumbel for link 2
            self.plot_dumbel(img_axis, (prev_hinge2_pixel[0], prev_hinge2_pixel[1]), (prev_mass_pixel[0], prev_mass_pixel[1]),
                             color=color, alpha=alpha)
        # Plot dumbel for link 1
        self.plot_dumbel(img_axis, (cart_pixel[0], cart_pixel[1]), (hinge2_pixel[0], hinge2_pixel[1]),
                         color=color, alpha=alpha)
        # Plot dumbel for link 2
        self.plot_dumbel(img_axis, (hinge2_pixel[0], hinge2_pixel[1]), (mass_pixel[0], mass_pixel[1]),
                         color=color, alpha=alpha)
        return

    def overlay_dubins_state(self, img_axis, observed_state, alpha=1.0, color='g', display_t_only: bool = False):
        """
        :param img_axis: mpl axis obj onto which to perform the overlay
        :param observed_state: current observed state
        :param alpha: opacity of state overlay
        :param color: color of state overlay
        :return:
        """
        # Cam matrix for projecting points from world to pixel space
        cam_mat = np.load(self.config.cam_mat_path)

        # Car geometric center, car moves in x-y plane
        car_x = observed_state[0]
        car_y = observed_state[1]

        car_world = np.array([car_x, car_y, 0.0, 1.0])
        car_pixel = cam_mat @ car_world
        # Divide by 2D homogenous scaling term
        car_pixel = self.homogenous_to_regular_coordinates(car_pixel)

        # Only visualize theta if it is available as part of the received state
        #  because rectangular geom of dubins car does not allow theta to be disambiguated from a single frame
        #  so the state from single images will not include theta even though is it a "static/position" state
        # This is kept separate since the angle is a "special state" that can be tested by overlaying onto the single
        #  frame at time t, however it can only be inferred from a pair of images
        #  So we might want to plot the dumbel pointing in the direction of the heading to visualize traning labels
        #  on a single frame (Unlike for other simple models where labels for quantities that need 2 frames to be
        #  estimated also need 2 frames to be able to visualize the labels)
        if len(observed_state) > 2:
            # Note: Since cos and sin are being produced by a NN, there is no guarantee that
            #  sin^2 + cos^2 = 1, so no guarantee that length of the visualized stick will remain constant
            car_cos_theta = observed_state[2]
            car_sin_theta = observed_state[3]
            # Visualize decoded/label heading of the car as a dumbel oriented in the direction of heading
            dlen = 0.3

            # Find a point lying in the direction of current heading, a distance dlen away from geometric center
            angle_pt_world = np.array([car_x + dlen * car_cos_theta, car_y + dlen * car_sin_theta,
                                       0.0, 1.0])
            angle_pt_pixel = cam_mat @ angle_pt_world
            angle_pt_pixel = self.homogenous_to_regular_coordinates(angle_pt_pixel)

        # Perform extra steps if concatenating 2 frames to overlay the full state with velocity
        if self.nframes == 2 and (not display_t_only):
            if len(observed_state) < 3:
                raise ValueError("2 frame Dubins car encoder model predicting less than 3 outputs, check config file"
                                 "encoder was trained with")

            # Check if dt has been set, else can't deal with 2 frames
            if self.dt is None:
                raise ValueError("Cannot perform overlay on two states when dt has not been set")

            # Augment the x-axis pixel coordinates to correspond to the right half of 2-stacked together frames
            car_pixel[0] += self.config.imsize

            # The Value error check at the start of this block ensures below will be defined even if pycharm complains
            angle_pt_pixel[0] += self.config.imsize

            # Compute the static portion (cart and mass coords) of the previous state based on current velocity
            #  formula for velocity is (x_t - x_{t-1}) / delta_t

            cos_theta_cur = observed_state[2]
            sin_theta_cur = observed_state[3]

            # Determine car location at t-1
            Vcar = observed_state[4]
            xcar_prev = observed_state[0] - Vcar * cos_theta_cur * self.dt
            ycar_prev = observed_state[1] - Vcar * sin_theta_cur * self.dt

            prev_car_world = np.array([xcar_prev, ycar_prev, 0.0, 1.0])

            prev_car_pixel = cam_mat @ prev_car_world
            prev_car_pixel = self.homogenous_to_regular_coordinates(prev_car_pixel)
            img_axis.scatter(prev_car_pixel[0], prev_car_pixel[1], marker='o', s=50, color=color, label='_nolegend_',
                             alpha=alpha)
        if len(observed_state) > 2:
            self.plot_dumbel(img_axis, (car_pixel[0], car_pixel[1]), (angle_pt_pixel[0], angle_pt_pixel[1]),
                             color=color, alpha=alpha, marker_size=50)
        else:
            img_axis.scatter(car_pixel[0], car_pixel[1], marker='o', s=50, color=color, label='_nolegend_',
                             alpha=alpha)
        return

    def plot_dumbel(self, img_axis, point1: Tuple[float, float], point2: Tuple[float, float], color, alpha,
                    marker1: str = 'o', marker2: str = 'o', marker_size: float = 100):
        """
        Plots a dumbel like shape (2 circles and a line connecting them)
        Intended for overlaying the state of simple models onto images of said simple models
        :param img_axis: mpl axis onto which performing overlay
        :param point1:
        :param point2:
        :param color:
        :param alpha:
        :param marker1: marker style for point 1
        :param marker2: marker style for point 2
        :param marker_size: marker size for both point1 and point2
        :return:
        """
        img_axis.scatter(point1[0], point1[1], marker=marker1, s=marker_size // self.nframes, color=color, label='_nolegend_',
                         alpha=alpha)
        img_axis.scatter(point2[0], point2[1], marker=marker2, s=marker_size // self.nframes, color=color, label='_nolegend_',
                         alpha=alpha)
        img_axis.plot((point1[0], point2[0]), (point1[1], point2[1]), color=color, label='_nolegend_', alpha=alpha)

    @staticmethod
    def euclidean_distance(p1: np.ndarray, p2: np.ndarray):
        return np.sqrt(np.sum(np.square(p1 - p2)))

    @staticmethod
    def homogenous_to_regular_coordinates(array: np.ndarray):
        return array[:-1] / array[-1]
