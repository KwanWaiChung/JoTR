from typing import Dict, Any, List


class ActionEncoder:
    def decode(
        self, action_idx: int, state: Dict[str, Any]
    ) -> Dict[str, List[List[str]]]:
        """
        Args:
            action_idx (int): Index of action.
            state (Dict[str, Any]): The state.

        Returns:
            Dict[str, List[List[str,str]]]: Map 'd-i' to [[s,v]].

        """
        raise NotImplementedError

    def encode(self, action: Dict[str, List[List[str]]]) -> int:
        """_summary_

        Args:
            action (Dict[str, List[List[str]]]): 'd-i': [[s,v]]


        Returns:
            int: action_idx

        """
        raise NotImplementedError

    def get_out_dim(self) -> int:
        raise NotImplementedError
