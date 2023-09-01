from typing import Dict, List

"""Natural Language Generation Interface"""


class NLG:
    """NLG module interface."""

    def generate(self, action: Dict[str, List[List[str]]]):
        """Generate a natural language utterance conditioned on the dialog act.
        
        Args:
            action:
                The dialog action produced by dialog policy module.
                It's a mapping from 'd-i' to [[s-v]]
        Returns:
            utterance (str):
                A natural langauge utterance.
        """
        return ""
