from convlab.model.dst.dst import DST
from convlab.data.sgd.info import init_state
from util import get_diaact_list
from typing import List, Dict, Any
import json


class RuleDST(DST):
    def __init__(self, lower_case=False):
        self.lower_case = lower_case
        self.init_session()

    def update(self, user_act: List[List[str]] = None) -> Dict[str, Any]:
        if isinstance(user_act, dict):
            user_act = get_diaact_list(
                diaacts=user_act, intent_first=True, add_value=True
            )

        for intent, domain, slot, value in user_act:
            if self.lower_case:
                intent, domain, slot, value = (
                    intent.lower(),
                    domain.lower(),
                    slot.lower(),
                    value.lower(),
                )
            if domain not in self.state["belief_state"]:
                continue
            if intent.lower() == "inform":
                if slot in self.state["belief_state"][domain]:
                    self.state["belief_state"][domain][slot] = value
            elif intent.lower() == "request":
                if domain not in self.state["request_state"]:
                    self.state["request_state"][domain] = []
                if slot not in self.state["request_state"][domain]:
                    self.state["request_state"][domain].append(slot.lower())
        return self.state

    def init_session(self):
        self.state = init_state()
        if self.lower_case:
            self.state = json.loads(json.dumps(self.state).lower())
