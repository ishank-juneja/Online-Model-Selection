import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from src.plotting import GIFmaker
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

        self.gif_maker = GIFmaker(delay=35)

    def __call__(self, save_dir: str, viz_type: Literal["frames_only"], agent_ep_history: Dict, model_ep_histories: Dict[str, Dict]):
        """
        :param save_dir:
        :param viz_type:
        :param agent_ep_history:
        :param model_ep_histories:
        :return:
        """
        # Pick the visualization function for saving png frames to disk
        if viz_type == "frames_only":
            # Plot frames only with no uncertainty visualization
            viz_fn = self.save_frames_only
        # TODO: Add a method here that plots 2 filtered states (predicted and corrected) for every frame
        else:
            raise NotImplementedError
        # Save frames to disk
        viz_fn(save_dir, agent_ep_history, model_ep_histories)
        # Stitch together frames into gif
        # Path object for save_dir
        save_dir_obj = Path(save_dir)
        gif_dir = save_dir_obj.parent.__str__()
        gif_path = gif_dir + "/run.gif"
        self.gif_maker.make_gif(gif_path, save_dir)
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

        # Infer number of time-steps in episode from data
        nsteps = len(agent_ep_history['action'])

        # Create dict of overlay functions
        overlay_fn_dict = {}
        for smodel in self.model_names:
            overlay_fn_dict[smodel] = self.smodel_vizs[smodel].overlay_state

        for idx in range(nsteps):
            # Plot raw gt_frames on first axis
            ax[0].imshow(agent_ep_history['gt_frame'][idx])
            # Fetch GT rob_state at time idx
            rob_state = agent_ep_history['gt_rob_state'][idx][0, :]
            # Make a window and state overlay for every model
            for jdx, smodel in enumerate(self.model_names):
                smodel_state = model_ep_histories[smodel]['mu_z'][idx][0, :]
                # Augmented state with rob_state added
                smodel_state_aug = np.hstack((rob_state, smodel_state))
                smodel_frame = model_ep_histories[smodel]['masked_frame'][idx]
                # Display the masked frame for this smodel on an image axis
                ax[jdx + 1].imshow(smodel_frame)
                # Overlay smodel state on imag axis, only with a single frame at time t
                #  and no frame at time t-1
                overlay_fn_dict[smodel](ax[jdx + 1], smodel_state_aug, display_t_only=True)

            fig.suptitle("{0} Filtered S-model States".format(self.task_name.capitalize()), size=18)

            # Save temporary png file frames in designated folder
            # plt.show()
            fig.savefig(os.path.join(save_dir, "file{0:03d}.png".format(idx + 1)))

            # Clear all axes for next frame
            for jdx in range(naxes):
                ax[jdx].cla()

        fig.clear()
        plt.close()

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
