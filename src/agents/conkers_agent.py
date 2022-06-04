from src.agents.base_agent import BaseAgent
from typing import List


class ConkersAgent(BaseAgent):
    def __init__(self, smodel_list: List[str], device: str = 'cuda:0'):
        super(ConkersAgent, self).__init__(smodel_list=smodel_list, device=device)

        # Set task specific parameters here
        self.env_name: str = 'Conkers-v0'

        # Actually make the agent based on the task specific params set in this class definition
        self.make_agent_for_task()

        # Actions per loop iteration / nrepeats for action
        self.actions_per_loop = 1


