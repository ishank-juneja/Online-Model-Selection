import gym
import gym_cenvs
from src.agents.base_agent import BaseAgent
from typing import List


class ConkersAgent(BaseAgent):
    def __init__(self, smodel_list: List[str]):
        super(ConkersAgent, self).__init__(smodel_list=smodel_list)

        # Create and seed Conkers env object
        self.env_name = "Conkers-v0"

        self.action_dimension = 1

        # Planning/Control horizon for Conkers task
        self.episode_T = 100

        # Actions per loop iteration / nrpeeats for action
        self.actions_per_loop = 1

        # Env needed for finding dt even if env not stepped through
        #  Whether env will be stepped through or not for test depends on
        #  param self.env.dt
        self.env = gym.make(self.env_name)
        self.env.seed(0)
        self.env.action_space.seed(0)

        # Set cost functions of lib based on task
        goal = self.env.get_goal()
        self.model_lib.set_cost_fn(goal)

        # Controller related config for task
        self.make_planner()

