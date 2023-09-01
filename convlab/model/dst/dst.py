"""Dialog State Tracker Interface"""
from typing import Union, List, Dict, Any


class DST:
    """DST module interface."""

    def update(self, action: Union[str, List[str]]) -> Dict[str, Any]:
        """ Update the internal dialog state variable.

        Args:
            action (str or list of list):
                The type is str when DST is word-level (such as NBT), and list of list when it is DA-level.
        Returns:
            new_state (dict):
                Updated dialog state, with the same form of previous state.
        """
        pass

    def update_turn(self, sys_utt: str, user_utt: str) -> Dict[str, Any]:
        """ Update the internal dialog state variable with .

        Args:
            sys_utt (str):
                system utterance of current turn, set to `None` for the first turn
            user_utt (str):
                user utterance of current turn
        Returns:
            new_state (dict):
                Updated dialog state, with the same form of previous state.
        """
        pass

