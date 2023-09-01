from typing import List


class NLU:
    """NLU module interface."""

    def predict(self, utterance: str, context: List[str] = list()):
        """Predict the dialog act of a natural language utterance.
        
        Args:
            utterance (str):
                A natural language utterance.
            context (list of str):
                Previous utterances.

        Returns:
            action (list of list):
                The dialog act of utterance.
        """
        return []
