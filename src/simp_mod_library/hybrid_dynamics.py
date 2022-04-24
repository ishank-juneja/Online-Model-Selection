from src.simp_mod_library.simp_mod_lib import SimpModLib
import torch
from torch import nn


class HybridDynamics(nn.Module):
    """
    Hybrid dynamics object to invoke selected nominal dynamics mode
    Expected to be contained within a hybrid transition model that also handles uncertainty ...
    """
    def __init__(self, simp_mod_lib: SimpModLib, device: str = 'cuda:0'):
        """
        Initializes a hybrid dynamics object by combining the nominal dynamics of simple
        model priors contained in passed simp model library
        :param simp_mod_lib:
        """
        super(HybridDynamics, self).__init__()

        # TODO: Infer this from some common config later ...
        # Number of trajectories simulated by the planner at one go
        self.K: int = 1000

        # Device to put tensors on
        self.device = device

        self.dyn_funcs = []

        # The dimension of hybrid dynamics is the largest among the dimensions of all the priors
        #  the actual dynamics vector has an extra dimension for holding the model idx used
        #  to get to the current state to visualize "hybrid" plans
        self.nx: int = -1
        # For state, we just ensure consistency between the dimensionality of all action spaces
        self.nu: int = None

        # Build dynamics list and infer attributes
        for idx in range(simp_mod_lib.nmodels):
            model_dyn = simp_mod_lib[idx].dynamics_fn
            nx_cur = model_dyn.nx
            nu_cur = model_dyn.nu
            if self.nu is None:
                self.nu = nu_cur
            elif self.nu != nu_cur:
                raise ValueError("Inconsistent number of controls {0} and {1} found in models".format(self.nu, nu_cur))
            # State is max among all states
            if nx_cur > self.nx:
                self.nx = nx_cur
            self.dyn_funcs.append(model_dyn)

        # Extra dimension for holding model index used to make transition
        self.nx += 1

        # Variable to hold the dynamics mode used to get to the current state
        # Shape: self.K x 1 for concatenation with simple model state
        self.mode = torch.zeros((self.K, 1), dtype=torch.int8, device=self.device)

        # Number of modes
        self.nmodes = len(self.dyn_funcs)

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
        return padded_state[:, mode_nx]

    def pad_state(self, truncated_next_state: torch.Tensor) -> torch.Tensor:
        """
        Takes in the truncated next_state and pads it with zeros to bring it up to self.nx length
        :param truncated_next_state: K x self.dyn_funcs[cur_mode].nx
        :return: padded_nstate: K x self.nx
        """
        K, _ = truncated_next_state.shape
        # preallocate tensor of desired size
        padded_nstate = torch.zeros((K, self.nx), dtype=torch.float32, device='cuda:0')
        padded_nstate
        return padded_nstate

    def pad_state_with_mode(self, zero_padded_next_state: torch.Tensor) -> torch.Tensor:
        # Get a copy of self.mode for concatenation with state
        mode_cp = self.mode.detach().clone()
        next_state_ret = torch.concat((next_state, mode_cp))

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Propagate state and action via the hybrid nominal dynamics prior
        :param state: K x self.nx dimensional state vector from planner
        :param action: K x self.nu from the planner, actions on the actuation location common/shared across the priors
        :return:
        """
        # Iterate over every mode of the dynamics function
        for idx in range(self.nmodes):

        # Get truncated version of state compatible with current dynamics mode
        cur_state = self.truncate_state(padded_state=state, cur_mode=cur_mode)
        # Invoke the dynamics of the currently active simple model mode
        next_state = self.dyn_funcs[cur_mode](cur_state, action)
        # pad next_state with zeros and mode to bring to K x self.nx dimensions
        next_state_padded = self.pad_state(truncated_next_state=next_state)
        return next_state_padded
