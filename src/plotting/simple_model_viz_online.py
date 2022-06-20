import matplotlib.pyplot as plt
import numpy as np
import os
from src.plotting.simple_model_viz import SimpleModViz
from typing import Callable, List
STDDEV_SCALE = 1.0


class SMVOnline(SimpleModViz):
    """
    Simple Model Viz functionality with methods for doing offline perception tests
    """
    def __init__(self, simp_model: str, vel_as_color: bool = False):
        super(SMVOnline, self).__init__(simp_model, vel_as_color)

        # TODO: Add methods for uncertainty function visualization
        #  Take the 2D state and add the uncertainties in all the dimensions
        #  to get an uncertainty Heat map

    def
