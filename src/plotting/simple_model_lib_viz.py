import matplotlib.pyplot as plt
from typing import List, Dict


class SimpModLibViz:
    """
    Serves two purposes:
    1. Act like an interface between SML and visualizers of individual simple models
    2. Define methods for viz quantities that are common to all models (ex: Robot State)
    """
    def __init__(self, model_names: List[str], smodel_lib: Dict):
        """
        :param model_names: Simp Model names to index into smodel_lib
        :param smodel_lib: Index into this dict to access the viz of the underlying models
        """
        self.model_names = model_names
        self.model_lib = smodel_lib

        return

    # TODO: Add a method that visualizes the live filtered state and plots live filtered uncertainty
    #  side by side for all the smodels