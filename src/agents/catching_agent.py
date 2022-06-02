from src.agents.base_agent import BaseAgent
from typing import List


class CatchingAgent(BaseAgent):
    def __init__(self, smodel_list: List[str]):
        super(CatchingAgent, self).__init__(smodel_list=smodel_list, env_name="Catching-v0")

        # TODO: Have environment name be set in the task definition
        #  but be created in the base class
