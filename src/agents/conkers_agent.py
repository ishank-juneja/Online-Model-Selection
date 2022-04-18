import gym
import gym_cenvs
from src.agents.base_agent import BaseAgent
from src.simp_mod_library.simp_mod_lib import SimpModLib


class ConkersAgent(BaseAgent):
    def __init__(self, smodel_lib: SimpModLib):
        super(ConkersAgent, self).__init__(smodel_lib=smodel_lib)

        # Create and seed Conkers env object
        self.env_name = "Conkers-v0"

        # Env needed for finding dt even if env not stepped through
        #  Whether env will be stepped through or not for test depends on
        #  param self.env.dt
        self.env = gym.make(self.env_name)
        self.env.seed(0)
        self.env.action_space.seed(0)
