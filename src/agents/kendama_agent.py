from src.agents.base_agent import BaseAgent
from typing import List


class KendamaAgent(BaseAgent):
    def __init__(self, smodel_list: List[str]):
        super(KendamaAgent, self).__init__(smodel_list=smodel_list, env_name="Kendama-v0")
