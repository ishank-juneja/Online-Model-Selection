from copy import deepcopy
import matplotlib.pyplot as plt
import os


class FigureSaver:
    """
    A wrapper class around the savefig() method of matplotlib to ensure
     frames are stitched in an intended order
    """
    def __init__(self):
        # Index of frame to be saved next
        self.frame_idx = 0

    def save_fig(self, fig, save_dir: str, nsaves: int = 1):
        """
        :param fig: mpl fig object
        :param save_dir: Location where fig is to be saved as a png file
        :param nsaves: Number of times this frame is saved to disk
        :return:
        """
        # Reference to a deep copy of fig to save figure multiple times
        for idx in range(nsaves):
            # next_fig = deepcopy(fig)
            fig.savefig(os.path.join(save_dir, "file{0:05d}.png".format(self.frame_idx + 1)))
            # fig = next_fig
            self.frame_idx += 1

