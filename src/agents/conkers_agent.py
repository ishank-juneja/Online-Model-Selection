from src.agents.base_agent import BaseAgent
from typing import List


class ConkersAgent(BaseAgent):
    def __init__(self, smodel_list: List[str], device: str = 'cuda:0'):
        super(ConkersAgent, self).__init__(device=device)

        # Set task specific parameters here
        self.env_name: str = 'Conkers-v0'

        # Actions per loop iteration / nrepeats for action
        self.actions_per_loop = 1

        # GT State dimension of system for task
        self.gt_state_dim = 33
        # Dimensions of actions for task
        self.action_dimension = 1
        # Set max episode duration for task
        self.episode_T = 100
        # Set planning horizon for task
        self.planner_H = 20
        # Number of trajectories simulated by planner
        self.planner_N = 1000

        # Indices of the gt state that are observable from the env
        # Ex: For passing down the gt values of actuator related quantities
        self.rob_gt_idx = [0, 11]

        # Params provided by agent to library being created
        self.rob_mass: float = 1.0

        # Actually make the agent based on the task specific params set in this class definition
        self.make_agent_for_task(smodel_list=smodel_list)
