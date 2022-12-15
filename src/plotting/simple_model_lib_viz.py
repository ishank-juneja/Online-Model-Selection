import glob
import matplotlib.pyplot as plt
from math import pi, cos, sin, atan2, sqrt
import numpy as np
import os
from pathlib import Path
from src.plotting import FigureSaver, GIFMaker, VideoMaker
from src.plotting.simple_model_viz_online import SMVOnline
from typing import List, Literal, Dict


class SimpModLibViz:
    """
    Solution to plotting data cached during a single episode ...
    Code repetition due to subtle differences between the plotting of different data_keys
    """
    def __init__(self, task_name: str, model_names: List[str]):
        """
        :param task_name: Task for which this viz is created
        :param model_names: Simp Model names to index into smodel_lib
        """
        self.task_name = task_name
        self.model_names = model_names
        self.nmodels = len(model_names)

        # Create an online viz method and get the right overlay function for every simple model in the library
        self.vizs = {}
        self.overlay_fns = {}
        for smodel in self.model_names:
            self.vizs[smodel] = SMVOnline(smodel)
            self.vizs[smodel].set_nframes(nframes=1)
            # Retrieve reference to the right viz function from a SMViz inited using the right config file
            self.overlay_fns[smodel] = self.vizs[smodel].overlay_state

        # - - - - - - - - - - - - - - - - - -
        # Figure saving related
        self.gif_maker = GIFMaker(delay=15)
        self.video_maker = VideoMaker(frame_rate=2)
        self.figure_saver = FigureSaver()
        # - - - - - - - - - - - - - - - - - -

    def __call__(self, save_dir: str, viz_name: Literal["filtered_state_on_frames", "predicted_state_on_frames"],
                 agent_ep_history: Dict, model_ep_histories: Dict[str, Dict]):
        """
        :param save_dir:
        :param viz_type:
        :param agent_ep_history:
        :param model_ep_histories:
        :return:
        """
        viz_fn = self.__getattribute__(viz_name)
        # Save frames to disk
        viz_fn(save_dir, agent_ep_history, model_ep_histories)
        # Stitch together frames into gif
        # Path object for save_dir
        save_dir_obj = Path(save_dir)
        out_dir = save_dir_obj.parent.__str__()
        # gif_path = out_dir + "/run.gif"
        mp4_path = out_dir + "/run.mp4"
        # self.gif_maker.make_gif(gif_path=gif_path, frames_dir=save_dir)
        # self.video_maker.make_video(video_path=mp4_path, frames_dir=save_dir)
        # Remove frames from tmp folder
        # self.cleanup_frames(save_dir)

    def filtered_state_on_frames(self, save_dir: str, agent_ep_history: Dict, model_ep_histories: Dict):
        """
        Plot gt_frame (from agent), and filtered simple model states over-layed on their segmented images side by side
        :param save_dir: Dir to save results in
        :param agent_ep_history: Contains data for keys defined in BaseAgent
        :param model_ep_histories: data for keys defined in SimpModBook
        :return:
        """
        # Number of axes in plot
        naxes = self.nmodels + 1
        fig = plt.figure(figsize=(5 * naxes, 5))
        ax = fig.subplots(1, naxes)
        # First frame flag, only apply tight_layout to 1 frame
        first_frame = True

        # Infer number of time-steps in episode from data
        nsteps = len(agent_ep_history['action'])

        # Fetch rob states from agent episode data, shape: nsteps x 1 x rob_dim
        rob_states = agent_ep_history['gt_rob_state']

        for idx in range(nsteps):
            # Plot raw gt_frames on first axis
            ax[0].imshow(agent_ep_history['gt_frame'][idx])
            # Turn off the mpl px units imshow axes
            ax[0].axis('off')
            ax[0].set_title('{0} frame'.format(self.task_name.capitalize()))
            # Fetch GT rob_state at time idx
            rob_state = rob_states[idx][0, :]

            # Make a window and state overlay for every model
            for jdx, smodel in enumerate(self.model_names):
                smodel_aug_state = np.hstack((rob_state, model_ep_histories[smodel]['mu_z'][idx][0, :]))
                smodel_aug_state_pre = np.hstack((rob_state, model_ep_histories[smodel]['mu_z_pre'][idx][0, :]))
                smodel_frame = model_ep_histories[smodel]['masked_frame'][idx]
                # Display the masked frame for this smodel on an image axis
                ax[jdx + 1].imshow(smodel_frame)
                # Turn off mpl px units axes
                ax[jdx + 1].axis('off')
                ax[jdx + 1].set_title('{0} smodel state'.format(smodel.capitalize()), size=12)
                # Overlay smodel state on imag axis, only with a single frame at time t
                #  and no frame at time t-1
                self.overlay_fns[smodel](ax[jdx + 1], smodel_aug_state, display_t_only=True)
                # Blank sup-title for right spacing
                fig.suptitle("".format(self.task_name.capitalize()), size=14)

            # Must invoke tight_layout after putting everything on mpl canvas
            if first_frame:
                fig.tight_layout()
                first_frame = False

            # Save temporary png file frames in designated folder
            self.figure_saver.save_fig(fig, save_dir, nsaves=1)

            # Clear all axes for next frame
            for kdx in range(1, naxes):
                ax[kdx].cla()

        fig.clear()
        plt.close()
        return

    def observed_state_on_frames(self, save_dir: str, agent_ep_history: Dict, model_ep_histories: Dict):
        """
        Plot gt_frame (from agent), and filtered simple model states over-layed on their segmented images side by side
        :param save_dir: Dir to save results in
        :param agent_ep_history: Contains data for keys defined in BaseAgent
        :param model_ep_histories: data for keys defined in SimpModBook
        :return:
        """
        # Number of axes in plot
        naxes = self.nmodels + 1
        fig = plt.figure(figsize=(5 * naxes, 5))
        ax = fig.subplots(1, naxes)
        # First frame flag, only apply tight_layout to 1 frame
        first_frame = True

        # Infer number of time-steps in episode from data
        nsteps = len(agent_ep_history['action'])

        # Fetch rob states from agent episode data, shape: nsteps x 1 x rob_dim
        rob_states = agent_ep_history['gt_rob_state']

        for idx in range(nsteps):
            # Plot raw gt_frames on first axis
            ax[0].imshow(agent_ep_history['gt_frame'][idx])
            # Turn off the mpl px units imshow axes
            ax[0].axis('off')
            ax[0].set_title('{0} frame'.format(self.task_name.capitalize()))
            # Fetch GT rob_state at time idx
            rob_state = rob_states[idx][0, :]

            # Make a window and state overlay for every model
            for jdx, smodel in enumerate(self.model_names):
                smodel_state = model_ep_histories[smodel]['mu_y'][idx][0, :]
                smodel_aug_state = np.hstack((rob_state, smodel_state))
                smodel_frame = model_ep_histories[smodel]['masked_frame'][idx]
                # Display the masked frame for this smodel on an image axis
                ax[jdx + 1].imshow(smodel_frame)
                # Turn off mpl px units axes
                ax[jdx + 1].axis('off')
                ax[jdx + 1].set_title('{0} smodel state'.format(smodel.capitalize()), size=12)
                # Overlay smodel state on imag axis, only with a single frame at time t
                #  and no frame at time t-1
                self.overlay_fns[smodel](ax[jdx + 1], smodel_aug_state, display_t_only=True)
                # Blank sup-title for right spacing
                fig.suptitle("".format(self.task_name.capitalize()), size=14)

            # Must invoke tight_layout after putting everything on mpl canvas
            if first_frame:
                fig.tight_layout()
                first_frame = False

            # Save temporary png file frames in designated folder
            self.figure_saver.save_fig(fig, save_dir, nsaves=1)

            # Clear all axes for next frame
            for kdx in range(1, naxes):
                ax[kdx].cla()

        fig.clear()
        plt.close()
        return

    def predicted_state_on_frames(self, save_dir: str, agent_ep_history: Dict, model_ep_histories: Dict):
        """
        Plot gt_frame (from agent), and predicted simple model states over-layed on their segmented images side by side
        :param save_dir: Dir to save results in
        :param agent_ep_history: Contains data for keys defined in BaseAgent
        :param model_ep_histories: data for keys defined in SimpModBook
        :return:
        """
        # Number of axes in plot
        naxes = self.nmodels + 1
        fig = plt.figure(figsize=(5 * naxes, 5))
        ax = fig.subplots(1, naxes)
        # First frame flag, only apply tight_layout to 1 frame
        first_frame = True

        # Infer number of time-steps in episode from data
        nsteps = len(agent_ep_history['action'])

        # Fetch rob states from agent episode data, shape: nsteps x 1 x rob_dim
        rob_states = agent_ep_history['gt_rob_state']

        for idx in range(nsteps - 1):
            # Plot raw gt_frames on first axis
            ax[0].imshow(agent_ep_history['gt_frame'][idx + 1])
            # Turn off the mpl px units imshow axes
            ax[0].axis('off')
            ax[0].set_title('{0} frame'.format(self.task_name.capitalize()))
            # Fetch GT rob_state at time idx
            rob_state = rob_states[idx + 1][0, :]

            # Make a window and state overlay for every model
            for jdx, smodel in enumerate(self.model_names):
                smodel_aug_state = np.hstack((rob_state, model_ep_histories[smodel]['mu_z_pre'][idx][0, :]))
                smodel_frame = model_ep_histories[smodel]['masked_frame'][idx + 1]
                # Display the masked frame for this smodel on an image axis
                ax[jdx + 1].imshow(smodel_frame)
                # Turn off mpl px units axes
                ax[jdx + 1].axis('off')
                ax[jdx + 1].set_title('{0} smodel state'.format(smodel.capitalize()), size=12)
                # Overlay smodel state on imag axis, only with a single frame at time t
                #  and no frame at time t-1
                self.overlay_fns[smodel](ax[jdx + 1], smodel_aug_state, display_t_only=True)
                # Blank sup-title for right spacing
                fig.suptitle("".format(self.task_name.capitalize()), size=14)

            # Must invoke tight_layout after putting everything on mpl canvas
            if first_frame:
                fig.tight_layout()
                first_frame = False

            # Save temporary png file frames in designated folder
            self.figure_saver.save_fig(fig, save_dir, nsaves=1)

            # Clear all axes for next frame
            for kdx in range(1, naxes):
                ax[kdx].cla()

        fig.clear()
        plt.close()
        return

    @staticmethod
    def cleanup_frames(tmp_frames_dir):
        """
        Cleanup the frames from the tmp location once done making GIF
        :param tmp_frames_dir: Location to clear frames for to prepare for next call
        :return:
        """
        # Get rid of temp png files
        for file_name in glob.glob(os.path.join(tmp_frames_dir, '*.png')):
            os.remove(file_name)
        return
