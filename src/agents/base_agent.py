from abc import ABCMeta
from src.simp_mod_library.simp_mod_lib import SimpModLib
from src.transition_models import UnscentedKalmanTransitions


class BaseAgent(metaclass=ABCMeta):
    """
    Base class for constructing agents that control the complex object using passed Simple Model Library
    """
    def __init__(self, smodel_lib: SimpModLib):
        # # Dummy env var over-riden by child classes
        # self.env = None

        self.smodel_lib = smodel_lib

        # Set type of trnasition model being used
        self.trans_model = UnscentedKalmanTransitions

        # Create dict of transition models for all simple models
        self.transition_models = {}

        for simp_model in self.smodel_lib.lib.keys():
            # Retrieve the config object for this simp_model
            smodel_cfg = self.smodel_lib.lib[simp_model].cfg()
            # Construct transition model using config
            smodel_trans = self.trans_model(smodel_cfg)
            self.transition_models[simp_model] = smodel_trans

    @classmethod
    def __new__(cls, *args, **kwargs):
        """
        Make abstract base class non-instaiable
        :param args:
        :param kwargs:
        """
        if cls is BaseAgent:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return object.__new__(cls)
