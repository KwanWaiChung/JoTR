from convlab.model.policy.usr_sgd_rule import DEF_VAL_UNK
from convlab.env.env import Environment
from typing import Dict, List, Tuple
from copy import deepcopy
import json


class AggressiveRewardEnv(Environment):
    def __init__(
        self,
        inform_intents: List[str],
        request_intents: List[str],
        slot_mapping: Dict[str, Dict[str, str]] = {},
        rew_weight: float = 1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rew_weight = rew_weight
        self.inform_intents = [s.lower() for s in inform_intents]
        self.request_intents = [s.lower() for s in request_intents]
        self.slot_mapping = json.loads(json.dumps(slot_mapping).lower())
        self.prev_usr_goal = deepcopy(self.get_goal())

    def _get_reward(self) -> float:
        main_rew = super()._get_reward()
        extra_rew = 0
        # domain -> 'info'/'reqt' -> slot value pairs
        # d-i -> [[s,v]]
        sys_action: Dict[str, List[List[str]]] = json.loads(
            json.dumps(self.get_state()["system_action"]).lower()
        )
        goal = json.loads(json.dumps(self.prev_usr_goal).lower())
        informed_acts = []
        requested_acts = []
        prev_usr_informed = [
            (d, self.slot_mapping.get(d, {}).get(s, s))
            for d, i, s in self.get_previous_acts(
                role="usr", intents=self.inform_intents
            )
        ]
        prev_sys_requested = [
            (d, self.slot_mapping.get(d, {}).get(s, s))
            for d, i, s in self.get_previous_acts(
                role="sys", intents=self.request_intents
            )
        ]
        for diaact, svs in sys_action.items():
            domain, intent = diaact.split("-")
            if domain not in goal:
                continue
            for slot, value in svs:
                if domain in self.slot_mapping and slot != "none":
                    slot = self.slot_mapping[domain].get(slot, slot)
                # sys informed a slot in user's request list which hasn't been
                # informed before.
                if (
                    intent in self.inform_intents
                    and "reqt" in goal[domain]
                    and slot in goal[domain.lower()]["reqt"]
                    and goal[domain.lower()]["reqt"][slot] == DEF_VAL_UNK
                    and (domain, intent, slot) not in informed_acts
                ):
                    extra_rew += 1
                    informed_acts.append((domain, intent, slot))
                # sys requested a slot in user's info list
                elif (
                    intent in self.request_intents
                    and "info" in goal[domain.lower()]
                    and slot in goal[domain.lower()]["info"]
                    and (domain, slot)
                    not in prev_sys_requested + prev_usr_informed
                    and (domain, intent, slot) not in requested_acts
                ):
                    extra_rew += 1
                    requested_acts.append((domain, intent, slot))
                elif intent in self.inform_intents + self.request_intents:
                    extra_rew -= 1
        self.prev_usr_goal = deepcopy(self.get_goal())
        return main_rew + extra_rew * self.rew_weight


class ConservativeRewardEnv(Environment):
    def __init__(
        self,
        inform_intents: List[str],
        request_intents: List[str],
        rew_weight: float = 1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rew_weight = rew_weight
        self.inform_intents = [s.lower() for s in inform_intents]
        self.request_intents = [s.lower() for s in request_intents]

    def _get_reward(self) -> float:
        """
        1. Informed what have been just requested
        2. Request an unrequested slots
        """
        main_rew = super()._get_reward()
        extra_rew = 0
        prev_usr_action = {}
        history = self.get_state()["history"]
        if len(history) > 1:
            prev_usr_action: Dict[str, List[List[str]]] = json.loads(
                json.dumps(self.get_state()["history"]).lower()
            )[-2][1]
        prev_usr_action_list = [
            (*di.split("-"), s)
            for di, svs in prev_usr_action.items()
            for s, v in svs
        ]
        prev_sys_reqt = self.get_previous_acts(
            role="sys", intents=self.request_intents
        )
        sys_action: Dict[str, List[List[str]]] = json.loads(
            json.dumps(self.get_state()["system_action"]).lower()
        )

        for diaact, svs in sys_action.items():
            domain, intent = diaact.split("-")
            for slot, value in svs:
                if (
                    intent in self.inform_intents
                    and (domain, "request", slot) in prev_usr_action_list
                ):
                    extra_rew += 1
                elif (
                    intent in self.request_intents
                    and (domain, intent, slot) not in prev_sys_reqt
                ):
                    extra_rew += 1
        return main_rew + extra_rew * self.rew_weight


class ActionNumEnvironment(Environment):
    def __init__(self, rew_weight: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rew_weight = rew_weight

    def _get_reward(self) -> float:
        main_rew = super()._get_reward()
        usr_action: Dict[str, List[List[str]]] = json.loads(
            json.dumps(self.get_state()["user_action"]).lower()
        )
        count = 0
        for diaact, svs in usr_action.items():
            count += len(svs)
        return main_rew + count * self.rew_weight
