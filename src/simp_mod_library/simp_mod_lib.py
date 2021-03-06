import numpy as np
from src.simp_mod_library.simp_mod_book import SimpModBook
import torch
from typing import List, Union


class SimpModLib:
    """
    Abbreviated SML
    Class to encapsulate everything needed for the interaction between the agent and the library of simple model
    dynamics priors. Lib is a tool used by BaseAgent.
    Can be thought of as a sophisticated wrapper around a list of SimpModBooks (SMB)
    """
    def __init__(self, model_names: List[str], rob_mass: float, dir_manager, device: str):
        """
        :param model_names: List of names of simple models in library
        :param rob_mass: Common robot mass parameter
        :param dir_manager: Dir manager object passed down from agent
        :param device: cpu/gpu
        """
        self.device = device

        self.dir_manager = dir_manager

        # Sort the model names lexicographically for deterministic indexing of books in lib.
        model_names.sort()
        # Have used setter ...
        self.nmodels = len(model_names)

        # The key data element of this Library
        #  A lib is a dictionary ... irony ?
        self.lib = {}

        # Create a separate Simple Model Book object for every model
        #  Infer and verify that all models are compatible for a library
        self.rob_dim: int = None    # Number of states in common robot state
        self.action_dim: int = None  # Number of dims in actions should be common
        for mod_name in model_names:
            smodel_book = SimpModBook(simp_mod=mod_name, dir_manager=self.dir_manager, device=device)
            # Consistency check for newly created SMB
            if self.rob_dim is not None:
                if smodel_book.cfg.rob_dim != self.rob_dim:
                    raise NotImplementedError("Model library with different robot states is not implemented")
            # Else value not set, set it
            else:
                self.rob_dim = smodel_book.cfg.rob_dim
            # Consistency check for newly created SMB
            if self.action_dim is not None:
                if smodel_book.cfg.action_dim != self.action_dim:
                    raise NotImplementedError("Model library with different action spaces is not implemented")
            # Else value not set, set it
            else:
                self.action_dim = smodel_book.cfg.action_dim
            self.lib[mod_name] = smodel_book

        # List of available simple models, idx <-> str mapping is based on this list
        self.model_names = model_names

        # - - - - - - - - - - - - - - - - -
        # Robot related params common to the books of all the models in the library
        self.rob_state = torch.zeros(1, self.rob_dim, device=self.device)
        self.rob_mass_float = rob_mass
        self.rob_mass = torch.tensor(self.rob_mass_float, device=self.device)   # Learnable version
        # - - - - - - - - - - - - - - - - -

    def __str__(self):
        return "Library with simple models {0}".format(self.model_names)

    def __repr__(self):
        return self.__str__()

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

    def predict(self, action, rob_state=None):
        """
        Run predict on the books of all models
        :param action:
        :param rob_state: GT robot state passed to the methods, if none, then last set rob_state of lib used
        :return:
        """
        # Infer batch size from action shape
        B, _ = action.size()
        # Use last set rob_state for lib if None
        if rob_state is None:
            rob_state = self.rob_state.repeat(1, B).view(-1, self.rob_dim)
        for model in self.model_names:
            # Send action and rob_state to models for letting them predict using their resp. dynamics
            self.lib[model].predict(action, rob_state)

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
        # Clear params that are global to library
        self.rob_state = torch.zeros(1, self.rob_dim, device=self.device)
        self.rob_mass = torch.tensor(self.rob_mass_float, device=self.device)
        for model in self.model_names:
            self.lib[model].reset_book()

    def save_episode_data(self):
        """
        Save all the data/state built up and iterated over for the entire library
        :return:
        """
        for model in self.model_names:
            self.lib[model].save_episode_data()

