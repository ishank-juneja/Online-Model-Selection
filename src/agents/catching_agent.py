from src.agents.base_agent import BaseAgent
from typing import List


# TODO: Bring in sync with functionality in ConkersAgent
class CatchingAgent(BaseAgent):
    def __init__(self, smodel_list: List[str], device: str = 'cuda:0'):
        super(CatchingAgent, self).__init__(device=device)

        # Set task specific parameters here
        self.env_name: str = 'Catching-v0'

        # Planning/Control horizon for task
        self.episode_T = 100

        # Learn uni-modal transition models between the states of a single model
        #  Identical to what the zero-mean GP learns in LVSPC
        # If below is false we blindly rely on nominal dynamics with
        self.learn_unimodal_trans = False
        # Learn inter-model transitions
        self.learn_inter_model = False

        # Actually make the agent based on the task specific params set in this class definition
        self.make_agent_for_task(smodel_list=smodel_list)

        # Indices of the gt state that are observable from the env
        # Ex: For passing down the gt values of actuator related quantities
        self.rob_gt_idx = [0, 8]


