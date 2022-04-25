from src.simp_mod_library import SimpModLib
import torch
from torch import nn


class MMT(nn.Module):
    """
    MMT = Multiple Model Transition Distribution
    Takes in the transition distributions contained within simple model lib for every simple model
    and constructs a Hybrid object out of them that selects the model to be used for the next sample transition of
    every trajectory based on an uncertainty criteria
    """
    def __init__(self, simp_mod_lib: SimpModLib, device: str = 'cuda:0'):
        """
        Initializes a hybrid transition model by combining the transition models of simple
        model priors contained in passed simp model library
        :param simp_mod_lib:
        """
        super(MMT, self).__init__()

        # TODO: Infer this from some common config later ...
        # Number of trajectories simulated by the planner at one go
        self.K: int = 1000

        # Device to put tensors on
        self.device = device

        self.trans_models = []

        # The dimension of hybrid dynamics is the largest among the dimensions of all the priors
        #  the actual dynamics vector has an extra dimension for holding the model idx used
        #  to get to the current state to visualize "hybrid" plans
        self.nx: int = -1
        # For state, we just ensure consistency between the dimensionality of all action spaces
        self.nu: int = None

        # Build dynamics list and infer attributes
        for idx in range(simp_mod_lib.nmodels):
            model_trans = simp_mod_lib[idx].trans_dist
            nx_cur = model_trans.config.state_dimension
            nu_cur = model_trans.config.action_dimension
            if self.nu is None:
                self.nu = nu_cur
            elif self.nu != nu_cur:
                raise ValueError("Inconsistent number of controls {0} and {1} found in models".format(self.nu, nu_cur))
            # State is max among all states
            if nx_cur > self.nx:
                self.nx = nx_cur
            self.dyn_funcs.append(model_trans)

        # Extra dimension for holding model index used to make transition
        self.nx += 1

        # Variable to hold the dynamics mode used to get to the current state
        # Shape: self.K x 1 for concatenation with simple model state
        self.mode = torch.zeros((self.K, 1), dtype=torch.int8, device=self.device)

        # Number of modes = number of models in lib
        self.nmodes = len(self.dyn_funcs)

        # Container for holding the next state
        self.next_state = torch.tensor((self.K, self.nx), dtype=torch.float32, device=self.device)

    def reset_dynamics_mode(self):
        """
        Reset the dynamics mode to model with index 0 after every independent trial
        :return:
        """
        self.mode[:] = 0

    def set_dynamics_mode(self, model_idx: torch.Tensor):
        """
        Method to set the mode of the hybrid dynamics prior
        To be used by the transition model object's sample_dynamics method, which in turn acts as
        the dynamics passed to the MPC method
        :param model_idx: self.K x 1 tensor of ints to set dynamics modes for all trajectories
        :return:
        """
        self.mode[:] = model_idx

    def truncate_state(self, padded_state: torch.Tensor, cur_mode: int) -> torch.Tensor:
        """
        Method looks up the currently active mode of dynamics and truncates the full state vector
        being used to length compatible with model dynamics to be used
        :param padded_state: K x self.nx tensor
        :param cur_mode: int representing currently active mode
        :return:
        """
        # Retrieve the number of states in the mode of dynamics to be invoked in this iteration
        mode_nx = self.dyn_funcs[cur_mode].nx
        # Return the first mode nx states as the state to be massed to ode dynamics
        return padded_state[:, :mode_nx]

    def pad_state(self, truncated_next_state: torch.Tensor) -> torch.Tensor:
        """
        Takes in the truncated next_state and pads it with zeros to bring it up to self.nx length
        :param truncated_next_state: K x self.dyn_funcs[cur_mode].nx
        :return: padded_nstate: K x self.nx
        """
        _, mode_nx = truncated_next_state.shape
        # preallocate tensor of desired size
        padded_nstate = torch.zeros((self.K, self.nx), dtype=torch.float32, device='cuda:0')
        padded_nstate[:, :mode_nx] = truncated_next_state
        # Assign the last column of state reserved for mode_indices uses to obtain the states
        padded_nstate[:, -1] = self.mode.detach().clone()
        return padded_nstate

    def forward(self, state_uncertainty, action):
        """
        Plays the role of the agent.trans_dist.sample_dynamics in original LVSPC code
        Propagate state and action via the hybrid nominal dynamics prior
        :param state_uncertainty: K x (2*self.nx) dimensional state vector from planner,
        2*nx because state + uncertainty over every state,
        Semantics: [simple_model_state | mode idx | uncertainty | model-switch (hand-over) uncertainty]
        :param action: K x self.nu from the planner, actions on the actuation location common/shared across the priors
        :return: K x (2*self.nx + 1) tensor with same semantics as state
        """
        state = state_uncertainty[:self.nx]
        # Init next state with current state since different dynamics func modes are applied iteratively
        self.next_state[:] = state

        # Iterate over every mode of the dynamics function and
        for idx in range(self.nmodes):
            # Get a mask on the rows among the K rows corresponding to model idx
            mask = self.mode == idx  # self.K x 1 Boolean tensor
            # Truncate state to the elements needed by model idx
            state_trunc = self.truncate_state(padded_state=state, cur_mode=idx)
            # Assume dynamics with idx is being applied to all rows of state
            next_state_trunc = self.dyn_funcs[idx](state_trunc, action)
            # Pad state to bring back to self.nx elements per row
            next_state_padded = self.pad_state(next_state_trunc)
            # Only update self.next_state for rows where mask is True
            self.next_state = torch.where(mask, next_state_padded, self.next_state)
        # TODO: Create meaningful uncertainty trend here based on current state
        std = torch.zeros_like(self.next_state)
        # Return copy of next state to avoid mutability related bugs
        return torch.cat((self.next_state, std), dim=1)
