from typing import List, Union, Dict, Any, Optional
from convlab.model.state_encoder.state_encoder import StateEncoder
from tianshou.policy import BasePolicy
from tianshou.data import Batch
import numpy as np


class Policy(BasePolicy):
    """Policy module interface."""

    def __init__(self, state_encoder: StateEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_encoder = state_encoder

    def predict(
        self, state: Dict[str, Any]
    ) -> Union[Dict[str, List[List[str]]], str]:
        """Predict the next agent action given dialog state.

        Args:
            state (dict):
        Returns:
            action: 
                when the policy outputs dialogue act, the type is 'd-i': [[s,v]] 
                else when the policy outputs utterance directly, the type is str.
        """
        return {}

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        return

