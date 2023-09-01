from typing import Dict, Any
import numpy as np


class StateEncoder:
    """The state produced by DST  is often represented by a dict variable.
        However, most existing policy models (e.g., DQN and Policy Gradient)
        takes as input a vector. Therefore, you have to convert the state to 
        a vector representation before passing it to the policy module.
    """

    def encode(self, state: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def get_out_dim(self) -> int:
        raise NotImplementedError