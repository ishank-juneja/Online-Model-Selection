from src.simp_mod_library.simp_mod_datastruct import SimpModStruct
from src.transition_distributions import HeuristicUnscentedKalman, GPUnscentedKalman
from typing import List, Type, Union


class SimpModLib:
    def __init__(self, model_names: List[str], online_gp: bool, goal):
        """
        :param model_names: List of names of simple models in library
        :param online_gp: Whether to use online GP as transition model or something else
        :param goal: Some notion of goal which is passed to the simple model specific cost functions
         that are utilized by the goal
        TODO: Make goal into a fixed task-agnostic interface, maybe a Tuple
        """
        # Sort the model names lexicographically
        model_names.sort()

        # Infer number of models in lib
        self.nmodels = len(model_names)

        # A lib is a dictionary ... irony ?
        self.lib = {}

        self.online_gp = online_gp

        if self.online_gp:
            transition_dist = GPUnscentedKalman
        else:
            transition_dist = HeuristicUnscentedKalman

        # Load in perceptions
        for mod_name in model_names:
            smodel_struct = SimpModStruct(simp_mod=mod_name, transition_dist=transition_dist, goal=goal)
            self.lib[mod_name] = smodel_struct

        # List of available simple models, idx <-> str mapping is based on this list
        self.smodels = list(self.lib.keys())

    def __getitem__(self, item: Union[int, str]) -> SimpModStruct:
        if type(item) == int:
            return self.lib[self.smodels[item]]
        elif type(item) == str:
            return self.lib[item]
        else:
            raise NotImplementedError

    def model_name(self, idx: int) -> str:
        return self.smodels[idx]

    def model_idx(self, name: str) -> int:
        return self.smodels.index(name)

    @property
    def nmodels(self):
        return self._nmodels

    @nmodels.setter
    def nmodels(self, nmodels: int):
        self._nmodels = nmodels

    def reset(self):
        """
        Reset everything that is learned online
        :return:
        """
        for model in self.smodels:
            model_struct = self.lib[model]
            # Reset the online learned sys-id parameters of dynamics function
            model_struct.trans_dist.transition.reset_params()
            # Reset the number of iterations for which cost_fn has been invoked
            #  used to vary the weight of costs with time-step
            model_struct.cost_fn.iter = 0
            model_struct.cost_fn.uncertainty_cost = 0.0
            # If learning a GP online
            if self.online_gp:
                model_struct.trans_dist.reset_model()

