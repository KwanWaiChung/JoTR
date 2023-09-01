# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import random
import json
import os
import numpy as np

from convlab.data.multiwoz.info import REF_SYS_DA, REF_USR_DA
from convlab.data.multiwoz.dbquery import query
from convlab.data.multiwoz.dbquery2 import Database
from convlab.model.action_encoder.action_encoder import ActionEncoder
from typing import Dict, Any, List, Tuple, Union


DEFAULT_VOCAB_FILE = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "data/multiwoz/da_slot_cnt.json",
)
DEFAULT_DOMAIN_FILE = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "data/multiwoz/domain.json",
)
DEFAULT_INTENT_FILE = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "data/multiwoz/intent.json",
)
DEFAULT_SLOT_FILE = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "data/multiwoz/slot.json",
)


digit = "0123456789"


def generate_phone_num(length, seeder: Dict[str, Any] = {}):
    """Generate a phone num."""
    string = ""
    while len(string) < length:
        string += digit[seeder.get("py", random).randint(0, 999999) % 10]
    return string


def generate_car(seeder: Dict[str, Any] = {}):
    """Generate a car for taxi booking."""
    car_types = [
        "toyota",
        "skoda",
        "bmw",
        "honda",
        "ford",
        "audi",
        "lexus",
        "volvo",
        "volkswagen",
        "tesla",
    ]
    p = seeder.get("py", random).randint(0, 999999) % len(car_types)
    return car_types[p]


class SkipException(Exception):
    def __init__(self):
        pass


class ActionVocab:
    def __init__(self, vocab_path=DEFAULT_VOCAB_FILE, num_actions=500):
        # add general actions
        self.vocab = [
            {"general-welcome": ["none"]},
            {"general-greet": ["none"]},
            {"general-bye": ["none"]},
            {"general-reqmore": ["none"]},
        ]
        # add single slot actions
        for domain in REF_SYS_DA:
            for slot in REF_SYS_DA[domain]:
                self.vocab.append({domain + "-Inform": [slot]})
                self.vocab.append({domain + "-Request": [slot]})
        # add actions from stats
        with open(vocab_path, "r") as f:
            stats = json.load(f)
            for action_string in stats:
                try:
                    act_strings = action_string.split(";];")
                    action_dict = {}
                    for act_string in act_strings:
                        if act_string == "":
                            continue
                        domain_act, slots = act_string.split("[", 1)
                        domain, act_type = domain_act.split("-")
                        if act_type in ["NoOffer", "OfferBook"]:
                            action_dict[domain_act] = ["none"]
                        elif act_type in ["Select"]:
                            if slots.startswith("none"):
                                raise SkipException
                            action_dict[domain_act] = [slots.split(";")[0]]
                        else:
                            action_dict[domain_act] = sorted(slots.split(";"))
                    if action_dict not in self.vocab:
                        self.vocab.append(action_dict)
                    # else:
                    #     print("Duplicate action", str(action_dict))
                except SkipException as e:
                    print(act_strings)
                if len(self.vocab) >= num_actions:
                    break
        print("{} actions are added to vocab".format(len(self.vocab)))
        # pprint(self.vocab)

    def get_action(self, action_index):
        return self.vocab[action_index]


class HrlActionVocab(ActionVocab):
    def __init__(self, domains: List[str] = None, num_actions: int = 500):
        if domains:
            self.domains = domains
        else:
            self.domains = list(json.load(open(DEFAULT_DOMAIN_FILE)).keys())
        self.intents = list(json.load(open(DEFAULT_INTENT_FILE)).keys())
        self.slots = list(json.load(open(DEFAULT_SLOT_FILE)).keys())
        self.intent_slots = [
            f"{intent}-{slot}"
            for intent in self.intents
            for slot in self.slots
        ]
        # add general actions
        self.vocab = [
            {"general-welcome": ["none"]},
            {"general-greet": ["none"]},
            {"general-bye": ["none"]},
            {"general-reqmore": ["none"]},
        ]
        with open(DEFAULT_VOCAB_FILE, "r") as f:
            stats = json.load(f)
            for action_string in stats:
                try:
                    act_strings = action_string.split(";];")
                    action_dict = {}
                    for act_string in act_strings:
                        if act_string == "":
                            continue
                        domain_act, slots = act_string.split("[", 1)
                        domain, act_type = domain_act.split("-")
                        if domain not in self.domains:
                            continue
                        if act_type in ["NoOffer", "OfferBook"]:
                            action_dict[domain_act] = ["none"]
                        elif act_type in ["Select"]:
                            if slots.startswith("none"):
                                raise SkipException
                            action_dict[domain_act] = [slots.split(";")[0]]
                        else:
                            action_dict[domain_act] = sorted(slots.split(";"))
                    if action_dict and action_dict not in self.vocab:
                        self.vocab.append(action_dict)
                    # else:
                    #     print("Duplicate action", str(action_dict))
                except SkipException as e:
                    print(act_strings)
                if len(self.vocab) >= num_actions:
                    break
        print("{} actions are added to vocab".format(len(self.vocab)))

        # add single slot actions
        for domain in self.domains:
            for intent in self.intents:
                for slot in self.slots:
                    action_dict = {f"{domain}-{intent}": [slot]}
                    if action_dict not in self.vocab:
                        self.vocab.append(action_dict)

    def get_action(
        self,
        action_index=None,
        domain_index=None,
        intent_index=None,
        slot_index=None,
        intent_slot_index=None,
    ):
        result = []
        if action_index is not None:
            result.append(self.vocab[action_index])
        if domain_index is not None:
            result.append(self.domains[domain_index])
        if intent_index is not None:
            result.append(self.intents[intent_index])
        if slot_index is not None:
            result.append(self.slots[slot_index])
        if intent_slot_index is not None:
            result.append(self.intent_slots[intent_slot_index])
        return result

    def get_index(
        self,
        action: str = None,
        domain: str = None,
        intent: str = None,
        slot: str = None,
        intent_slot: str = None,
    ) -> List[str]:
        result = []
        if action is not None:
            d, i, s = action.split("-")
            action = {f"{d}-{i}": [s]}
            result.append(self.vocab.index(action))
        if domain is not None:
            result.append(self.domains.index(domain))
        if intent is not None:
            result.append(self.intents.index(intent))
        if slot is not None:
            result.append(self.slots.index(slot))
        if intent_slot is not None:
            result.append(self.intent_slots.index(intent_slot))
        return result


class MultiwozActionEncoder(ActionEncoder):
    def __init__(self, num_actions=300):
        self.action_vocab = ActionVocab(num_actions=num_actions)
        self.current_domain = "Restaurant"

    def get_out_dim(self) -> int:
        return len(self.action_vocab.vocab)

    def _find_best_delex_act(self, action: Dict[str, List[str]]) -> int:
        """It tries to find a target action that satisfy 2 requirements:
            1. The target action doesn't contain extra slots.
            2. The extra slots relative to target action is minimized.
            If it fails to find a target action that satisfy the first
            requirement. It proceeds to find a target action that minimizes
            the number of slot difference between the action and the target
            action.

        Args:
            action (Dict[str, List[str]]): 'd-i': [s]

        Returns:
            action_idx: int.

        """

        def _score(a1: Dict[str, List[str]], a2: Dict[str, List[str]]):
            """_summary_

            Args:
                a1 (Dict[str, List[str]]): _description_
                a2 (Dict[str, List[str]]): _description_

            Returns:
                int: Number of slots in a1 but not in a2.
            """
            score = 0
            for domain_act in a1:
                if domain_act not in a2:
                    score += len(a1[domain_act])
                else:
                    score += len(set(a1[domain_act]) - set(a2[domain_act]))
            return score

        best_p_action_index = -1
        best_p_score = float("inf")
        best_pn_action_index = -1
        best_pn_score = float("inf")
        for i, v_action in enumerate(self.action_vocab.vocab):
            if v_action == action:
                return i
            else:
                p_score = _score(action, v_action)
                n_score = _score(v_action, action)
                if p_score > 0 and n_score == 0 and p_score < best_p_score:
                    best_p_action_index = i
                    best_p_score = p_score
                else:
                    if p_score + n_score < best_pn_score:
                        best_pn_action_index = i
                        best_pn_score = p_score + n_score
        if best_p_action_index >= 0:
            return best_p_action_index
        return best_pn_action_index

    def encode(self, action: Dict[str, List[Tuple[str, str]]]) -> int:
        # action: {'d-i': [[s, v]]}
        # 'd-i': [[s]]
        delex_act = {}
        for domain_act in action:
            domain, act_type = domain_act.split("-", 1)
            if act_type in ["NoOffer", "OfferBook"]:
                delex_act[domain_act] = ["none"]
            elif act_type in ["Select"]:
                for sv in action[domain_act]:
                    if sv[0] != "none":
                        delex_act[domain_act] = [sv[0]]
                        break
            else:
                delex_act[domain_act] = [sv[0] for sv in action[domain_act]]
        return self._find_best_delex_act(delex_act)

    def decode(
        self, action_index: int, state: Dict[str, Any]
    ) -> Dict[str, List[List[str]]]:
        domains = [
            "Attraction",
            "Hospital",
            "Hotel",
            "Restaurant",
            "Taxi",
            "Train",
            "Police",
        ]
        delex_action = self.action_vocab.get_action(action_index)
        action = {}

        for act in delex_action:
            domain, act_type = act.split("-")
            if domain in domains:
                self.current_domain = domain
            if act_type == "Request":
                action[act] = []
                for slot in delex_action[act]:
                    action[act].append([slot, "?"])
            elif act == "Booking-Book":
                constraints = []
                for slot in state["belief_state"][self.current_domain.lower()][
                    "semi"
                ]:
                    if (
                        state["belief_state"][self.current_domain.lower()][
                            "semi"
                        ][slot]
                        != ""
                    ):
                        constraints.append(
                            [
                                slot,
                                state["belief_state"][
                                    self.current_domain.lower()
                                ]["semi"][slot],
                            ]
                        )
                kb_result = query(self.current_domain.lower(), constraints)
                if len(kb_result) == 0:
                    action[act] = [["none", "none"]]
                else:
                    if "Ref" in kb_result[0]:
                        action[act] = [["Ref", kb_result[0]["Ref"]]]
                    else:
                        action[act] = [["Ref", "N/A"]]
            elif domain not in domains:
                action[act] = [["none", "none"]]
            else:
                if act == "Taxi-Inform":
                    for info_slot in ["leaveAt", "arriveBy"]:
                        if (
                            info_slot in state["belief_state"]["taxi"]["semi"]
                            and state["belief_state"]["taxi"]["semi"][
                                info_slot
                            ]
                            != ""
                        ):
                            car = generate_car()
                            phone_num = generate_phone_num(11)
                            action[act] = []
                            action[act].append(["Car", car])
                            action[act].append(["Phone", phone_num])
                            break
                    else:
                        action[act] = [["none", "none"]]
                elif act in [
                    "Train-Inform",
                    "Train-NoOffer",
                    "Train-OfferBook",
                ]:
                    for info_slot in ["departure", "destination"]:
                        if (
                            info_slot
                            not in state["belief_state"]["train"]["semi"]
                            or state["belief_state"]["train"]["semi"][
                                info_slot
                            ]
                            == ""
                        ):
                            action[act] = [["none", "none"]]
                            break
                    else:
                        for info_slot in ["leaveAt", "arriveBy"]:
                            if (
                                info_slot
                                in state["belief_state"]["train"]["semi"]
                                and state["belief_state"]["train"]["semi"][
                                    info_slot
                                ]
                                != ""
                            ):
                                self.domain_fill(
                                    delex_action, state, action, act
                                )
                                break
                        else:
                            action[act] = [["none", "none"]]
                elif domain in domains:
                    self.domain_fill(delex_action, state, action, act)

        return action

    def domain_fill(self, delex_action, state, action, act):
        domain, act_type = act.split("-")
        constraints = []
        for slot in state["belief_state"][domain.lower()]["semi"]:
            if state["belief_state"][domain.lower()]["semi"][slot] != "":
                constraints.append(
                    [slot, state["belief_state"][domain.lower()]["semi"][slot]]
                )
        if act_type in [
            "NoOffer",
            "OfferBook",
        ]:  # NoOffer['none'], OfferBook['none']
            action[act] = []
            for slot in constraints:
                action[act].append(
                    [REF_USR_DA[domain].get(slot[0], slot[0]), slot[1]]
                )
        elif act_type in [
            "Inform",
            "Recommend",
            "OfferBooked",
        ]:  # Inform[Slot,...], Recommend[Slot, ...]
            kb_result = query(domain.lower(), constraints)
            # print("Policy Util")
            # print(constraints)
            # print(len(kb_result))
            if len(kb_result) == 0:
                action[act] = [["none", "none"]]
            else:
                action[act] = []
                for slot in delex_action[act]:
                    if slot == "Choice":
                        action[act].append([slot, len(kb_result)])
                    elif slot == "Ref":
                        if "Ref" in kb_result[0]:
                            action[act].append(["Ref", kb_result[0]["Ref"]])
                        else:
                            action[act].append(["Ref", "N/A"])
                    else:
                        try:
                            action[act].append(
                                [
                                    slot,
                                    kb_result[0][
                                        REF_SYS_DA[domain].get(slot, slot)
                                    ],
                                ]
                            )
                        except:
                            action[act].append([slot, "N/A"])
                if len(action[act]) == 0:
                    action[act] = [["none", "none"]]
        elif act_type in ["Select"]:  # Select[Slot]
            kb_result = query(domain.lower(), constraints)
            if len(kb_result) < 2:
                action[act] = [["none", "none"]]
            else:
                slot = delex_action[act][0]
                action[act] = []
                slot2 = REF_SYS_DA[domain].get(slot, slot)
                if slot2 in kb_result[0]:
                    action[act].append([slot, kb_result[0][slot2]])
                else:
                    action[act].append([slot, "N/A"])
                if slot2 in kb_result[1]:
                    action[act].append([slot, kb_result[1][slot2]])
                else:
                    action[act].append([slot, "N/A"])
        else:
            # print("Cannot decode:", str(delex_action))
            action[act] = [["none", "none"]]


# class HrlActionEncoder(MultiwozActionEncoder):
#     def __init__(
#         self,
#         domain_level: bool = False,
#         intent_level: bool = False,
#         domains: List[str] = None,
#     ):
#         """Can only give one act now.

#         Args:
#             domain (bool, optional): If True, domain is treated as one
#                 hierarchical level.
#             intent (bool, optional): If True, intent is treated as one
#                 hierarchical level.

#         """
#         # super().__init__()
#         self.domain_level = domain_level
#         self.intent_level = intent_level
#         self.current_domain = "Restaurant"
#         self.action_vocab = HrlActionVocab(domains)

#     def encode(self, action: Dict[str, List[str]], separate=False) -> int:
#         action_idx = super().encode(action)
#         if not separate:
#             return action_idx
#         domains, intents = zip(*[a.split("-") for a in action])
#         assert len(domains) == 1
#         assert len(intents) == 1
#         # assume only only action.
#         slots = list(action.values())[0]
#         domains_idx = self.action_vocab.domains.index(domains[0])
#         intents_idx = self.action_vocab.intents.index(intents[0])
#         slots_idx = self.action_vocab.slots.index(slots[0])
#         intent_slots_idx = self.action_vocab.intent_slots.index(
#             f"{intents[0]}-{slots[0]}"
#         )
#         return (
#             action_idx,
#             intent_slots_idx,
#             domains_idx,
#             intents_idx,
#             slots_idx,
#         )

#     def get_out_dim(self) -> int:
#         if self.domain_level and self.intent_level:
#             return len(self.action_vocab.slots)
#         elif self.domain_level:
#             return len(self.action_vocab.intent_slots)
#         else:
#             return len(self.action_vocab.vocab)

#     def decode(
#         self,
#         action_index: int = None,
#         state: Dict[str, Any] = None,
#         domain_index=None,
#         intent_index=None,
#         slot_index=None,
#         intent_slot_index=None,
#     ):
#         if action_index is not None:
#             action = super().decode(action_index, state)
#             domain_intent = list(action.keys())[0]
#             if action[domain_intent] == []:
#                 slot_value = ["none", "none"]
#             else:
#                 slot_value = action[domain_intent][0]
#             action = {domain_intent: [slot_value]}
#             return action
#         if domain_index is not None:
#             return self.action_vocab.domains[domain_index]
#         if intent_slot_index is not None:
#             return self.action_vocab.intent_slots[intent_slot_index]
#         if intent_index is not None:
#             return self.action_vocab.intents[intent_index]
#         if slot_index is not None:
#             return self.action_vocab.slots[slot_index]


class HMActionVocab:
    def __init__(
        self,
        vocab_path_domain=DEFAULT_DOMAIN_FILE,
        vocab_path_intent=DEFAULT_INTENT_FILE,
        vocab_path_slot=DEFAULT_SLOT_FILE,
        vocab_path=DEFAULT_VOCAB_FILE,
    ):

        self.vocab_domain = []
        self.vocab_intent = []
        self.vocab_slot = []

        self.vocab_domain = list(json.load(open(vocab_path_domain)).keys())
        self.vocab_intent = list(json.load(open(vocab_path_intent)).keys())
        self.vocab_slot = list(json.load(open(vocab_path_slot)).keys())

        print(
            "{} domain_action are added to vocab".format(
                len(self.vocab_domain)
            )
        )
        print(
            "{} intent_action are added to vocab".format(
                len(self.vocab_intent)
            )
        )
        print("{} slot_action are added to vocab".format(len(self.vocab_slot)))

        self.vocab = [
            {"general-welcome": ["none"]},
            {"general-greet": ["none"]},
            {"general-bye": ["none"]},
            {"general-reqmore": ["none"]},
        ]
        # add single slot actions
        for domain in REF_SYS_DA:
            for slot in REF_SYS_DA[domain]:
                self.vocab.append({domain + "-Inform": [slot]})
                self.vocab.append({domain + "-Request": [slot]})
        # add actions from stats
        with open(vocab_path, "r") as f:
            stats = json.load(f)
            for action_string in stats:
                try:
                    act_strings = action_string.split(";];")
                    action_dict = {}
                    for act_string in act_strings:
                        if act_string == "":
                            continue
                        domain_act, slots = act_string.split("[", 1)
                        domain, act_type = domain_act.split("-")
                        if act_type in ["NoOffer", "OfferBook"]:
                            action_dict[domain_act] = ["none"]
                        elif act_type in ["Select"]:
                            if slots.startswith("none"):
                                raise SkipException
                            action_dict[domain_act] = [slots.split(";")[0]]
                        else:
                            action_dict[domain_act] = sorted(slots.split(";"))
                    if action_dict not in self.vocab:
                        self.vocab.append(action_dict)
                    # else:
                    #     print("Duplicate action", str(action_dict))
                except SkipException as e:
                    print(act_strings)
                if len(self.vocab) >= 400:
                    break
        print("{} actions are added to vocab".format(len(self.vocab)))

    def get_actions(self, domain_idx, intent_idx, slot_idx):
        return (
            self.vocab_domain[domain_idx],
            self.vocab_intent[intent_idx],
            self.vocab_slot[slot_idx],
        )

    def get_action(self, action_index):
        return self.vocab[action_index]

    def get_subgoal(self, domain):
        return self.vocab_domain.index(domain)

    def get_index(self, domain_idx=None, intent_idx=None, slot_idx=None):
        assert domain_idx is not None
        if intent_idx is None and slot_idx is None:
            return self.vocab_domain.index(domain_idx)
        assert intent_idx is not None
        if slot_idx is None:
            return (
                self.vocab_domain.index(domain_idx),
                self.vocab_intent.index(intent_idx),
            )
        return (
            self.vocab_domain.index(domain_idx),
            self.vocab_intent.index(intent_idx),
            self.vocab_slot.index(slot_idx),
        )


class HrlActionEncoder:
    def __init__(
        self,
        domain_level: bool = False,
        intent_level: bool = False,
        domains: List[str] = None,
        single_action: bool = True,
    ):
        """Can only give one act now.

        Args:
            domain (bool, optional): If True, domain is treated as one
                hierarchical level.
            intent (bool, optional): If True, intent is treated as one
                hierarchical level.

        """
        self.domain_level = domain_level
        self.intent_level = intent_level
        self.current_domain = "Restaurant"
        self.action_vocab = HrlActionVocab(domains)
        self.single_action = single_action
        self.db = Database()

    def get_out_dim(self) -> int:
        if self.domain_level and self.intent_level:
            return len(self.action_vocab.slots)
        elif self.domain_level:
            return len(self.action_vocab.intent_slots)
        else:
            return len(self.action_vocab.vocab)

    def _find_best_delex_act(self, action: Dict[str, List[str]]) -> int:
        """_summary_

        Args:
            action (Dict[str, List[str]]): 'd-i': [s]

        Returns:
            action_idx: int.
        """

        def _score(a1, a2):
            score = 0
            for domain_act in a1:
                if domain_act not in a2:
                    score += len(a1[domain_act])
                else:
                    score += len(set(a1[domain_act]) - set(a2[domain_act]))
            return score

        best_p_action_index = -1
        best_p_score = float("inf")
        best_pn_action_index = -1
        best_pn_score = float("inf")
        for i, v_action in enumerate(self.action_vocab.vocab):
            if v_action == action:
                return i
            else:
                p_score = _score(action, v_action)
                n_score = _score(v_action, action)
                if p_score > 0 and n_score == 0 and p_score < best_p_score:
                    best_p_action_index = i
                    best_p_score = p_score
                else:
                    if p_score + n_score < best_pn_score:
                        best_pn_action_index = i
                        best_pn_score = p_score + n_score
        if best_p_action_index >= 0:
            return best_p_action_index
        return best_pn_action_index

    def encode(self, action: Dict[str, List[Tuple[str, str]]]) -> int:
        # action: {'d-i': [[s, v]]}
        # 'd-i': [[s]]
        delex_act = {}
        for domain_act in action:
            domain, act_type = domain_act.split("-", 1)
            if act_type in ["NoOffer", "OfferBook"]:
                delex_act[domain_act] = ["none"]
            elif act_type in ["Select"]:
                for sv in action[domain_act]:
                    if sv[0] != "none":
                        delex_act[domain_act] = [sv[0]]
                        break
            else:
                delex_act[domain_act] = [sv[0] for sv in action[domain_act]]
        return self._find_best_delex_act(delex_act)

    def decode(
        self,
        action_index: Union[Tuple[int, int, int], int],
        state: Dict[str, Any],
    ) -> Dict[str, List[Tuple[str, str]]]:
        domains = [
            "Attraction",
            "Hospital",
            "Hotel",
            "Restaurant",
            "Taxi",
            "Train",
            "Police",
        ]

        if isinstance(action_index, (list, np.ndarray)):
            if self.domain_level and self.intent_level:
                assert len(action_index) == 3
                a1, a2, a3 = action_index[0], action_index[1], action_index[2]
                domain, intent, slot = self.action_vocab.get_action(
                    domain_index=a1, intent_index=a2, slot_index=a3
                )
                delex_action = {domain + "-" + intent: [slot]}
            elif self.domain_level:
                assert len(action_index) == 2
                a1, a2 = action_index[0], action_index[1]
                domain, intent_slot = self.action_vocab.get_action(
                    domain_index=a1, intent_slot_index=a2
                )
                intent, slot = intent_slot.split("-")
                delex_action = {domain + "-" + intent: [slot]}
            else:
                raise ValueError(
                    "Expect action_index to be int but got "
                    f"{action_index}. domain_level and intent_level are False."
                )

        elif np.issubdtype(action_index, np.integer):
            delex_action = self.action_vocab.get_action(action_index)[0]
        else:
            print(action_index)
            raise ValueError(
                "Type of action_index must be either int or iterable"
            )

        action = {}
        for act in delex_action:
            domain, act_type = act.split("-")
            if domain in domains:
                self.current_domain = domain
            if act_type == "Request":
                action[act] = []
                for slot in delex_action[act]:
                    action[act].append([slot, "?"])
            elif act == "Booking-Book":
                constraints = []
                for slot in state["belief_state"][self.current_domain.lower()][
                    "semi"
                ]:
                    if (
                        state["belief_state"][self.current_domain.lower()][
                            "semi"
                        ][slot]
                        != ""
                    ):
                        constraints.append(
                            [
                                slot,
                                state["belief_state"][
                                    self.current_domain.lower()
                                ]["semi"][slot],
                            ]
                        )
                kb_result = self.db.query(
                    self.current_domain.lower(), constraints
                )
                if len(kb_result) == 0:
                    action[act] = [["none", "none"]]
                else:
                    if "Ref" in kb_result[0]:
                        action[act] = [["Ref", kb_result[0]["Ref"]]]
                    else:
                        action[act] = [["Ref", "N/A"]]
            elif domain not in domains:
                action[act] = [["none", "none"]]
            else:
                if act == "Taxi-Inform":
                    for info_slot in ["leaveAt", "arriveBy"]:
                        if (
                            info_slot in state["belief_state"]["taxi"]["semi"]
                            and state["belief_state"]["taxi"]["semi"][
                                info_slot
                            ]
                            != ""
                        ):
                            car = generate_car()
                            phone_num = generate_phone_num(11)
                            action[act] = []
                            action[act].append(["Car", car])
                            action[act].append(["Phone", phone_num])
                            break
                    else:
                        action[act] = [["none", "none"]]
                elif act in [
                    "Train-Inform",
                    "Train-NoOffer",
                    "Train-OfferBook",
                ]:
                    for info_slot in ["departure", "destination"]:
                        if (
                            info_slot
                            not in state["belief_state"]["train"]["semi"]
                            or state["belief_state"]["train"]["semi"][
                                info_slot
                            ]
                            == ""
                        ):
                            action[act] = [["none", "none"]]
                            break
                    else:
                        for info_slot in ["leaveAt", "arriveBy"]:
                            if (
                                info_slot
                                in state["belief_state"]["train"]["semi"]
                                and state["belief_state"]["train"]["semi"][
                                    info_slot
                                ]
                                != ""
                            ):
                                self.domain_fill(
                                    delex_action, state, action, act
                                )
                                break
                        else:
                            action[act] = [["none", "none"]]
                elif domain in domains:
                    self.domain_fill(delex_action, state, action, act)

        if isinstance(action_index, (list, np.ndarray)):
            return action

        # choose one action
        # if self.single_action:
        if len(action.keys()) > 1:
            key = list(action.keys())[0]
            if len(action[key]) > 1:
                return {key: [action[key][0]]}
            return {key: action[key]}

        key = list(action.keys())[0]
        if len(action[key]) > 1:
            return {key: [action[key][0]]}
        return action

    def domain_fill(self, delex_action, state, action, act):
        domain, act_type = act.split("-")
        constraints = []
        for slot in state["belief_state"][domain.lower()]["semi"]:
            if state["belief_state"][domain.lower()]["semi"][slot] != "":
                constraints.append(
                    [slot, state["belief_state"][domain.lower()]["semi"][slot]]
                )
        if act_type in [
            "NoOffer",
            "OfferBook",
        ]:  # NoOffer['none'], OfferBook['none']
            action[act] = []
            for slot in constraints:
                action[act].append(
                    [REF_USR_DA[domain].get(slot[0], slot[0]), slot[1]]
                )
        elif act_type in [
            "Inform",
            "Recommend",
            "OfferBooked",
        ]:  # Inform[Slot,...], Recommend[Slot, ...]
            kb_result = self.db.query(domain.lower(), constraints)
            # print("Policy Util")
            # print(constraints)
            # print(len(kb_result))
            if len(kb_result) == 0:
                action[act] = [["none", "none"]]
            else:
                action[act] = []
                for slot in delex_action[act]:
                    if slot == "Choice":
                        action[act].append([slot, len(kb_result)])
                    elif slot == "Ref":
                        if "Ref" in kb_result[0]:
                            action[act].append(["Ref", kb_result[0]["Ref"]])
                        else:
                            action[act].append(["Ref", "N/A"])
                    else:
                        try:
                            action[act].append(
                                [
                                    slot,
                                    kb_result[0][
                                        REF_SYS_DA[domain].get(slot, slot)
                                    ],
                                ]
                            )
                        except:
                            action[act].append([slot, "N/A"])
                if len(action[act]) == 0:
                    action[act] = [["none", "none"]]
        elif act_type in ["Select"]:  # Select[Slot]
            kb_result = self.db.query(domain.lower(), constraints)
            if len(kb_result) < 2:
                action[act] = [["none", "none"]]
            else:
                slot = delex_action[act][0]
                action[act] = []
                try:
                    action[act].append(
                        [
                            slot,
                            kb_result[0][REF_SYS_DA[domain].get(slot, slot)],
                        ]
                    )
                except:
                    action[act].append([slot, "N/A"])
                try:
                    action[act].append(
                        [
                            slot,
                            kb_result[1][REF_SYS_DA[domain].get(slot, slot)],
                        ]
                    )
                except:
                    action[act].append([slot, "N/A"])
        else:
            # print('Cannot decode:', str(delex_action))
            action[act] = [["none", "none"]]
        if len(action[act]) == 0:
            action[act] = [["none", "none"]]


# class HMHrlActionEncoderCopy:
#     def __init__(self):
#         self.action_vocab = HMActionVocab()
#         self.current_domain = "Restaurant"
#         self.db = Database()

#     def get_out_dim(self) -> int:
#         return len(self.action_vocab.vocab_slot)

#     def _find_best_delex_act(self, action: Dict[str, List[str]]) -> int:
#         """_summary_

#         Args:
#             action (Dict[str, List[str]]): 'd-i': [s]

#         Returns:
#             action_idx: int.
#         """

#         def _score(a1, a2):
#             score = 0
#             for domain_act in a1:
#                 if domain_act not in a2:
#                     score += len(a1[domain_act])
#                 else:
#                     score += len(set(a1[domain_act]) - set(a2[domain_act]))
#             return score

#         best_p_action_index = -1
#         best_p_score = float("inf")
#         best_pn_action_index = -1
#         best_pn_score = float("inf")
#         for i, v_action in enumerate(self.action_vocab.vocab):
#             if v_action == action:
#                 return i
#             else:
#                 p_score = _score(action, v_action)
#                 n_score = _score(v_action, action)
#                 if p_score > 0 and n_score == 0 and p_score < best_p_score:
#                     best_p_action_index = i
#                     best_p_score = p_score
#                 else:
#                     if p_score + n_score < best_pn_score:
#                         best_pn_action_index = i
#                         best_pn_score = p_score + n_score
#         if best_p_action_index >= 0:
#             return best_p_action_index
#         return best_pn_action_index

#     def encode(self, action: Dict[str, List[Tuple[str, str]]]) -> int:
#         # action: {'d-i': [[s, v]]}
#         # 'd-i': [[s]]
#         delex_act = {}
#         for domain_act in action:
#             domain, act_type = domain_act.split("-", 1)
#             if act_type in ["NoOffer", "OfferBook"]:
#                 delex_act[domain_act] = ["none"]
#             elif act_type in ["Select"]:
#                 for sv in action[domain_act]:
#                     if sv[0] != "none":
#                         delex_act[domain_act] = [sv[0]]
#                         break
#             else:
#                 delex_act[domain_act] = [sv[0] for sv in action[domain_act]]
#         return self._find_best_delex_act(delex_act)

#     def decode(
#         self,
#         action_index: Union[Tuple[int, int, int], int],
#         state: Dict[str, Any],
#     ) -> Dict[str, List[Tuple[str, str]]]:
#         domains = [
#             "Attraction",
#             "Hospital",
#             "Hotel",
#             "Restaurant",
#             "Taxi",
#             "Train",
#             "Police",
#         ]

#         if isinstance(action_index, (list, np.ndarray)):
#             assert len(action_index) == 3
#             a1, a2, a3 = action_index[0], action_index[1], action_index[2]
#             domain, intent, slot = self.action_vocab.get_actions(a1, a2, a3)
#             delex_action = {domain + "-" + intent: [slot]}
#         elif np.issubdtype(action_index, np.integer):
#             delex_action = self.action_vocab.get_action(action_index)
#         else:
#             print(action_index)
#             raise ValueError(
#                 "Type of action_index must be either int or iterable"
#             )

#         action = {}
#         for act in delex_action:
#             domain, act_type = act.split("-")
#             if domain in domains:
#                 self.current_domain = domain
#             if act_type == "Request":
#                 action[act] = []
#                 for slot in delex_action[act]:
#                     action[act].append([slot, "?"])
#             elif act == "Booking-Book":
#                 constraints = []
#                 for slot in state["belief_state"][self.current_domain.lower()][
#                     "semi"
#                 ]:
#                     if (
#                         state["belief_state"][self.current_domain.lower()][
#                             "semi"
#                         ][slot]
#                         != ""
#                     ):
#                         constraints.append(
#                             [
#                                 slot,
#                                 state["belief_state"][
#                                     self.current_domain.lower()
#                                 ]["semi"][slot],
#                             ]
#                         )
#                 kb_result = self.db.query(
#                     self.current_domain.lower(), constraints
#                 )
#                 if len(kb_result) == 0:
#                     action[act] = [["none", "none"]]
#                 else:
#                     if "Ref" in kb_result[0]:
#                         action[act] = [["Ref", kb_result[0]["Ref"]]]
#                     else:
#                         action[act] = [["Ref", "N/A"]]
#             elif domain not in domains:
#                 action[act] = [["none", "none"]]
#             else:
#                 if act == "Taxi-Inform":
#                     for info_slot in ["leaveAt", "arriveBy"]:
#                         if (
#                             info_slot in state["belief_state"]["taxi"]["semi"]
#                             and state["belief_state"]["taxi"]["semi"][
#                                 info_slot
#                             ]
#                             != ""
#                         ):
#                             car = generate_car()
#                             phone_num = generate_phone_num(11)
#                             action[act] = []
#                             action[act].append(["Car", car])
#                             action[act].append(["Phone", phone_num])
#                             break
#                     else:
#                         action[act] = [["none", "none"]]
#                 elif act in [
#                     "Train-Inform",
#                     "Train-NoOffer",
#                     "Train-OfferBook",
#                 ]:
#                     for info_slot in ["departure", "destination"]:
#                         if (
#                             info_slot
#                             not in state["belief_state"]["train"]["semi"]
#                             or state["belief_state"]["train"]["semi"][
#                                 info_slot
#                             ]
#                             == ""
#                         ):
#                             action[act] = [["none", "none"]]
#                             break
#                     else:
#                         for info_slot in ["leaveAt", "arriveBy"]:
#                             if (
#                                 info_slot
#                                 in state["belief_state"]["train"]["semi"]
#                                 and state["belief_state"]["train"]["semi"][
#                                     info_slot
#                                 ]
#                                 != ""
#                             ):
#                                 self.domain_fill(
#                                     delex_action, state, action, act
#                                 )
#                                 break
#                         else:
#                             action[act] = [["none", "none"]]
#                 elif domain in domains:
#                     self.domain_fill(delex_action, state, action, act)

#         if isinstance(action_index, list):
#             return action

#         # choose one action
#         if len(action.keys()) > 1:
#             key = list(action.keys())[0]
#             if len(action[key]) > 1:
#                 return {key: [action[key][0]]}
#             return {key: action[key]}

#         key = list(action.keys())[0]
#         if len(action[key]) > 1:
#             return {key: [action[key][0]]}

#         return action

#     def domain_fill(self, delex_action, state, action, act):
#         domain, act_type = act.split("-")
#         constraints = []
#         for slot in state["belief_state"][domain.lower()]["semi"]:
#             if state["belief_state"][domain.lower()]["semi"][slot] != "":
#                 constraints.append(
#                     [slot, state["belief_state"][domain.lower()]["semi"][slot]]
#                 )
#         if act_type in [
#             "NoOffer",
#             "OfferBook",
#         ]:  # NoOffer['none'], OfferBook['none']
#             action[act] = []
#             for slot in constraints:
#                 action[act].append(
#                     [REF_USR_DA[domain].get(slot[0], slot[0]), slot[1]]
#                 )
#         elif act_type in [
#             "Inform",
#             "Recommend",
#             "OfferBooked",
#         ]:  # Inform[Slot,...], Recommend[Slot, ...]
#             kb_result = self.db.query(domain.lower(), constraints)
#             # print("Policy Util")
#             # print(constraints)
#             # print(len(kb_result))
#             if len(kb_result) == 0:
#                 action[act] = [["none", "none"]]
#             else:
#                 action[act] = []
#                 for slot in delex_action[act]:
#                     if slot == "Choice":
#                         action[act].append([slot, len(kb_result)])
#                     elif slot == "Ref":
#                         if "Ref" in kb_result[0]:
#                             action[act].append(["Ref", kb_result[0]["Ref"]])
#                         else:
#                             action[act].append(["Ref", "N/A"])
#                     else:
#                         try:
#                             action[act].append(
#                                 [
#                                     slot,
#                                     kb_result[0][
#                                         REF_SYS_DA[domain].get(slot, slot)
#                                     ],
#                                 ]
#                             )
#                         except:
#                             action[act].append([slot, "N/A"])
#                 if len(action[act]) == 0:
#                     action[act] = [["none", "none"]]
#         elif act_type in ["Select"]:  # Select[Slot]
#             kb_result = self.db.query(domain.lower(), constraints)
#             if len(kb_result) < 2:
#                 action[act] = [["none", "none"]]
#             else:
#                 slot = delex_action[act][0]
#                 action[act] = []
#                 try:
#                     action[act].append(
#                         [
#                             slot,
#                             kb_result[0][REF_SYS_DA[domain].get(slot, slot)],
#                         ]
#                     )
#                 except:
#                     action[act].append([slot, "N/A"])
#                 try:
#                     action[act].append(
#                         [
#                             slot,
#                             kb_result[1][REF_SYS_DA[domain].get(slot, slot)],
#                         ]
#                     )
#                 except:
#                     action[act].append([slot, "N/A"])
#         else:
#             # print('Cannot decode:', str(delex_action))
#             action[act] = [["none", "none"]]
#         if len(action[act]) == 0:
#             action[act] = [["none", "none"]]
