import numpy as np
from src.simp_mod_library.simp_mod_book import SimpModBook
from typing import List, Union


class SimpModLib:
    """
    Class to encapsulate everything needed for the interaction between the agent and the library of simple model
    dynamics priors. Lib is a tool used by BaseAgent.
    Can be thought of as a sophisticated wrapper around a list of SimpModBooks (SMB)
    """
    def __init__(self, model_names: List[str], device: str):
        """
        param model_names: List of names of simple models in library
        :param device: cpu/gpu
        """
        # Sort the model names lexicographically for deterministic indexing of books in lib.
        model_names.sort()
        # Have used setter ...
        self.nmodels = len(model_names)

        # The key data element of this Library
        #  A lib is a dictionary ... irony ?
        self.lib = {}

        # Create a separate Simple Model Book object for every
        for mod_name in model_names:
            smodel_book = SimpModBook(simp_mod=mod_name, device=device)
            self.lib[mod_name] = smodel_book

        # List of available simple models, idx <-> str mapping is based on this list
        self.model_names = model_names

    def __getitem__(self, item: Union[int, str]) -> SimpModBook:
        if type(item) == int:
            return self.lib[self.model_names[item]]
        elif type(item) == str:
            return self.lib[item]
        else:
            raise NotImplementedError

    def model_name(self, idx: int) -> str:
        return self.model_names[idx]

    def model_idx(self, name: str) -> int:
        return self.model_names.index(name)

    @property
    def nmodels(self):
        return self._nmodels

    @nmodels.setter
    def nmodels(self, nmodels: int):
        self._nmodels = nmodels

    def predict(self, action):
        """
        Run predict on the books of all models
        :param action:
        :return:
        """
        for model in self.model_names:
            # Send action to models for letting them predict using their resp. dynamics
            self.lib[model].predict(action)

    def update(self, obs):
        """
        Run observation_update on the books of all simple models
        :param obs:
        :return:
        """
        for model in self.model_names:
            # Send observation to all models to let them update using their resp. perceptions
            self.lib[model].observation_update(obs)

    def reset_episode(self, obs: np.ndarray):
        """
        For resetting local history after a single episode
        :return:
        """
        for model in self.model_names:
            # Send initial obs to simple models for initializing their approximate state estimates
            self.lib[model].reset_episode(obs)

    def reset_trial(self):
        """
        Reset the simple model library by purging everything that is learned online
        :return:
        """
        for model in self.model_names:
            self.lib[model].hard_reset()
