import json
import os
import copy

from typing import Dict, List, Any
from convlab.data.multiwoz.util import normalize_value
from convlab.data.multiwoz.info import REF_SYS_DA, init_state
from convlab.model.dst.dst import DST


class RuleDST(DST):
    """Rule based DST which trivially updates new values from NLU result to states."""

    def __init__(self):
        prefix = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.value_dict = json.load(
            open(os.path.join(prefix, "data/multiwoz/value_dict.json"))
        )
        self.init_session()

    def update(
        self, user_act: Dict[str, List[List[str]]] = None
    ) -> Dict[str, Any]:
        # print('------------------{}'.format(user_act))
        if not isinstance(user_act, dict):
            raise Exception(
                "Expect user_act to be <class 'dict'> type but get {}.".format(
                    type(user_act)
                )
            )
        previous_state = self.state
        new_belief_state = copy.deepcopy(previous_state["belief_state"])
        new_request_state = copy.deepcopy(previous_state["request_state"])
        for domain_type in user_act.keys():
            domain, tpe = domain_type.lower().split("-")
            if domain in ["unk", "general", "booking"]:
                continue
            if tpe == "inform":
                for k, v in user_act[domain_type]:
                    k = REF_SYS_DA[domain.capitalize()].get(k, k)
                    if k is None:
                        continue
                    if domain not in new_belief_state:
                        raise Exception(
                            "Error: domain <{}> not in new belief state".format(
                                domain
                            )
                        )
                    domain_dic = new_belief_state[domain]
                    assert "semi" in domain_dic
                    assert "book" in domain_dic

                    if k in domain_dic["semi"]:
                        nvalue = normalize_value(self.value_dict, domain, k, v)
                        new_belief_state[domain]["semi"][k] = nvalue
                    elif k in domain_dic["book"]:
                        new_belief_state[domain]["book"][k] = v
                    elif k.lower() in domain_dic["book"]:
                        new_belief_state[domain]["book"][k.lower()] = v
                    elif k == "trainID" and domain == "train":
                        new_belief_state[domain]["book"][k] = normalize_value(
                            self.value_dict, domain, k, v
                        )
                    else:
                        # raise Exception('unknown slot name <{}> of domain <{}>'.format(k, domain))
                        with open("unknown_slot.log", "a+") as f:
                            f.write(
                                "unknown slot name <{}> of domain <{}>\n".format(
                                    k, domain
                                )
                            )
            elif tpe == "request":
                for k, v in user_act[domain_type]:
                    k = REF_SYS_DA[domain.capitalize()].get(k, k)
                    if domain not in new_request_state:
                        new_request_state[domain] = {}
                    if k not in new_request_state[domain]:
                        new_request_state[domain][k] = 0

        new_state = copy.deepcopy(previous_state)
        new_state["belief_state"] = new_belief_state
        new_state["request_state"] = new_request_state
        new_state["user_action"] = user_act

        self.state = new_state

        return self.state

    def init_session(self):
        self.state = init_state()
