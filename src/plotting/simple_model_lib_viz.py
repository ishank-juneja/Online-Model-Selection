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
    Serves two purposes:
    1. Act like an interface between SML and visualizers of individual simple models
    2. Define methods for viz quantities that are common to all models (ex: Robot State)
    """
    def __init__(self, task_name: str, model_names: List[str]):
        """
        :param task_name: Task for which this viz is created
        :param model_names: Simp Model names to index into smodel_lib
        """
        self.task_name = task_name
        self.model_names = model_names
        self.nmodels = len(model_names)

        self.smodel_vizs = {}

        for smodel in self.model_names:
            self.smodel_vizs[smodel] = SMVOnline(smodel)
            self.smodel_vizs[smodel].set_nframes(nframes=1)

        # - - - - - - - - - - - - - - - - - -
        # Figure saving related
        self.gif_maker = GIFMaker(delay=15)
        self.video_maker = VideoMaker(frame_rate=2)
        self.figure_saver = FigureSaver()
        # - - - - - - - - - - - - - - - - - -

    def __call__(self, save_dir: str, viz_type: Literal["frames_only"], agent_ep_history: Dict, model_ep_histories: Dict[str, Dict]):
        """
        :param save_dir:
        :param viz_type:
        :param agent_ep_history:
        :param model_ep_histories:
        :return:
        """
        # Pick the visualization function for saving png frames to disk
        if viz_type == "smodel_frames":
            # Plot frames only with no uncertainty visualization
            viz_fn = self.save_frames_only
        elif viz_type == "smodel_frames_and_plot":
            # Plot frames only but with 2 different frames for predicted and corrected
            #  filter states
            raise NotImplementedError
        else:
            raise NotImplementedError
        # Save frames to disk
        viz_fn(save_dir, agent_ep_history, model_ep_histories)
        # Stitch together frames into gif
        # Path object for save_dir
        save_dir_obj = Path(save_dir)
        out_dir = save_dir_obj.parent.__str__()
        # gif_path = out_dir + "/run.gif"
        mp4_path = out_dir + "/run.mp4"
        # self.gif_maker.make_gif(gif_path=gif_path, frames_dir=save_dir)
        self.video_maker.make_video(video_path=mp4_path, frames_dir=save_dir)
        # Remove frames from tmp folder
        self.cleanup_frames(save_dir)

    def save_frames_only(self, save_dir: str, agent_ep_history: Dict, model_ep_histories: Dict):
        """
        Plot gt_frame (from agent), and simple model states over-layed on their segmented images side by side
        :param save_dir: Dir to save results in
        :param agent_ep_history: Contains data for keys defined in BaseAgent
        :param model_ep_histories: ... SimpModBook
        :return:
        """
        # Number of axes in plot
        naxes = self.nmodels + 1
        fig = plt.figure(figsize=(5 * naxes, 5))
        ax = fig.subplots(1, naxes)
        # First frame flag
        first_frame = True

        # Infer number of time-steps in episode from data
        nsteps = len(agent_ep_history['action'])

        # Fetch rob states from agent episode data, shape: nsteps x 1 x rob_dim
        rob_states = agent_ep_history['gt_rob_state']
        # Create dict of overlay functions and create rob state augmented smodel states
        overlay_fn_dict = {}
        for smodel in self.model_names:
            overlay_fn_dict[smodel] = self.smodel_vizs[smodel].overlay_state
            # Fetch smodel states for active model from SimPModBook episode data
            smodel_states = model_ep_histories[smodel]['mu_z']
            # Predicted smodel states, one row less than smodel_states
            smodel_states_pred = model_ep_histories[smodel]['mu_z_pre']
            # Concatenate robot state and filtered simple model state
            smodel_aug_states = np.dstack((rob_states, smodel_states))
            # Repeat for pred_states
            smodel_aug_pred_states = np.dstack((rob_states[1:], smodel_states_pred))

        idx = 0
        corrected_done = False
        while idx < nsteps:
            for state_type in ["corrected", "predicted"]:
                # Plot raw gt_frames on first axis
                ax[0].imshow(agent_ep_history['gt_frame'][idx])
                # Turn off the mpl px units imshow axes
                ax[0].axis('off')
                ax[0].set_title('{0} frame'.format(self.task_name.capitalize()))
                # Fetch GT rob_state at time idx
                rob_state = rob_states[idx][0, :]
                # Make a window and state overlay for every model
                for jdx, smodel in enumerate(self.model_names):
                    # Decide whether to plot corrected state or predicted state in this iteration
                    if state_type == 'corrected' or idx == 0:
                        # Augmented state with rob_state added
                        smodel_aug_state = smodel_aug_states[idx][0, :]
                        corrected_done = True
                    elif state_type == 'predicted' and idx > 0:
                        smodel_aug_state = smodel_aug_pred_states[idx][0, :]
                        corrected_done = False
                    smodel_frame = model_ep_histories[smodel]['masked_frame'][idx]
                    # Display the masked frame for this smodel on an image axis
                    ax[jdx + 1].imshow(smodel_frame)
                    # Turn off mpl px units axes
                    ax[jdx + 1].axis('off')
                    ax[jdx + 1].set_title('{0} smodel state'.format(smodel.capitalize()), size=12)
                    # Overlay smodel state on imag axis, only with a single frame at time t
                    #  and no frame at time t-1
                    overlay_fn_dict[smodel](ax[jdx + 1], smodel_aug_state, display_t_only=True)
                    # Overlay robot state as a Red dot on image
                    self.smodel_vizs[smodel].overlay_rob_state(ax[jdx + 1], rob_state, color='r', display_t_only=True)
                # Blank sup-title for right spacing
                fig.suptitle("".format(self.task_name.capitalize()), size=14)

                # Must invoke tight_layout after putting everything on mpl canvas
                if first_frame:
                    fig.tight_layout()
                    first_frame = False
                # Save temporary png file frames in designated folder
                self.figure_saver.save_fig(fig, save_dir, nsaves=1)

                # Clear all axes for next frame
                for jdx in range(naxes):
                    ax[jdx].cla()
            if corrected_done:
                idx += 1

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
