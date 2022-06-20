import matplotlib.pyplot as plt
import numpy as np
import os
from src.plotting.simple_model_viz import SimpleModViz
from typing import Callable, List
STDDEV_SCALE = 1.0


class SMZOnline(SimpleModViz):
    """
    Simple Model Viz functionality with methods for doing offline perception tests
    """
    def __init__(self, simp_model: str, vel_as_color: bool = False):
        super(SMZOnline, self).__init__(simp_model, vel_as_color)


