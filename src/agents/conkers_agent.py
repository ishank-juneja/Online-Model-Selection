from src.agents.base_agent import BaseAgent
from typing import List


class ConkersAgent(BaseAgent):
    def __init__(self, smodel_list: List[str]):
        super(ConkersAgent, self).__init__(smodel_list=smodel_list, env_name="Conkers-v0")




