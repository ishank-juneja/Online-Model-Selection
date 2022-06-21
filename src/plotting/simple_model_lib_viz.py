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

    def save_episode_frames(self, agent_history: Dict, model_histories: Dict[Dict]):
        """
        Invoke the viz methods of books in model_lib to display current filtered state
         of all simple models side by side
        :param agent_history: History params specific to agent as defined in __init__ of BaseAgent
        :param model_histories: Use the model_names to index into the model histories of individual simple models
        :return:
        """


