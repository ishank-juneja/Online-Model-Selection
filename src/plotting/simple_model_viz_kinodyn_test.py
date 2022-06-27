import matplotlib.pyplot as plt
import numpy as np
import os
from src.plotting import FigureSaver
from src.plotting.simple_model_viz_online import SMVOnline
from typing import List


class SMVKDTest(SMVOnline):
    """
    Use SMVOnline state format for viz and add some methods for KD testing (plot 2 states on the same frame)
    """
    def __init__(self, simp_model: str, cam_mat: np.ndarray = None, vel_as_color: bool = False):
        super(SMVKDTest, self).__init__(simp_model, cam_mat, vel_as_color)
        self.figure_saver = FigureSaver()

    def save_frames_with_current_and_predicted_frames(self, frames: np.ndarray, cur_sim_states: np.ndarray,
                                                      next_pred_states: np.ndarray, save_dir: str):
        """
        Overlay both cur_sim_states and next_pred_states onto frames
        :param frames: Frames from a simple model env. or an augmented simple model env like catching is the env for the
        freely falling ball simple model env
        :param cur_sim_states: State at time t from the simulator
        :param next_pred_states: State at time t+1 predicted using the kino-dynamic function being tested
        :param save_dir: Dir location where to save overlayed frames
        :return:
        """
        # Number of axes in plot
        naxes = 1
        fig = plt.figure(figsize=(5 * naxes, 5))
        ax = fig.subplots(1, 1) # A single non-subscriptable axis
        # First frame flag, only apply tight_layout to 1 frame
        first_frame = True
        # Infer number of time-steps in episode from data
        nsteps = len(frames) - 1

        for idx in range(nsteps):
            cur_gt_state = cur_sim_states[idx + 1][0, :]
            pred_state = next_pred_states[idx][0, :]
            frame = frames[idx + 1][::512//64, ::512//64, :]

            # Display the masked frame for this smodel on an image axis
            ax.imshow(frame)
            # Turn off mpl px units axes
            ax.axis('off')
            # Overlay smodel state on imag axis, only with a single frame at time t
            #  and no frame at time t-1
            self.overlay_state(ax, cur_gt_state, display_t_only=True, color='g', alpha=0.3)
            self.overlay_state(ax, pred_state, display_t_only=True, color='orange', alpha=0.3)

            # Blank sup-title for right spacing
            fig.suptitle('{0} dynamics'.format(self.simp_model.capitalize(), size=14))

            # Must invoke tight_layout after putting everything on mpl canvas
            if first_frame:
                fig.tight_layout()
                first_frame = False

            # Save temporary png file frames in designated folder
            self.figure_saver.save_fig(fig, save_dir, nsaves=1)

            # Clear axes for next frame
            ax.cla()

        fig.clear()
        plt.close()
        return
