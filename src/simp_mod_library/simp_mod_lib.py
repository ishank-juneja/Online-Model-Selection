from src.networks import SimpModPerception
from typing import List


class SimpModLib:
    def __init__(self, perception_names: List[dict]):
        # Infer number of models in lib
        self.nmodels = len(perception_names)

        # A lib is a dictionary ... irony ?
        self.lib = {}

        # Load in perceptions
        for idx in range(self.nmodels):
            smodel_per = SimpModPerception(seg_model_name=perception_names[idx]["segmenter"],
                                           encoder_model_name=perception_names[idx]["encoder"])
            # Infer the name of this simple model
            smodel_name = smodel_per.simp_model
            self.lib[smodel_name] = smodel_per

        # List of available simple models
        self.avail_smodels = list(self.lib.keys())

    def __getitem__(self, item: str) -> SimpModPerception:
        return self.lib[item]

    @property
    def nmodels(self):
        return self._nmodels

    @nmodels.setter
    def nmodels(self, nmodels: int):
        self._nmodels = nmodels
