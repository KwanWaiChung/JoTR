from convlab.model.action_encoder.multiwoz_action_encoder import (
    HrlActionEncoder,
    generate_car,
    generate_phone_num,
)
from typing import Iterable, List, Dict, Any, Tuple
from util import str_to_diaact, get_diaact_dict
from data.dataset import DialogActSessionDataset, DialogActTurnDataset
from data.sgd_dataset import SGDDialogActTurnDataset
from convlab.data.multiwoz.dbquery2 import Database
from convlab.data.sgd.dbquery import Database as SGDDatabase
from convlab.data.sgd.info import NUL_VALUE
import numpy as np
import json

domains = [
    "Attraction",
    "Hospital",
    "Hotel",
    "Restaurant",
    "Taxi",
    "Train",
    "Police",
]


class DAGPTActionEncoder2(HrlActionEncoder):
    def __init__(
        self, tokenizer, has_prefix: bool, seeder: Dict[str, Any] = {}
    ):
        self.tokenizer = tokenizer
        self.db = Database(seeder=seeder)
        self.current_domain = "Restaurant"
        self.has_prefix = has_prefix
        self.seeder = seeder

    def get_out_dim(self) -> int:
        # not important
        return 1024

    def _decode_act(self, delex_action, state):
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
                            car = generate_car(seeder=self.seeder)
                            phone_num = generate_phone_num(
                                11, seeder=self.seeder
                            )
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

    def decode(
        self, response_ids: Iterable[int], state: Dict[str, Any]
    ) -> Dict[str, List[List[str]]]:
        """
        Args:
            response_ids (Iterable[int]): Array of response input ids.
            state (Dict[str, Any]): The state.

        Returns:
            Dict[str, List[List[str,str]]]: Map 'd-i' to [[s,v]].

        """
        response_str: str = self.tokenizer.decode(response_ids)
        _delex_action: List[
            Tuple[str, str, str]
        ] = DialogActSessionDataset.str_to_diaact(
            response_str, has_prefix=self.has_prefix
        )
        delex_action: Dict[str, List[str]] = {}
        for d, i, s in _delex_action:
            delex_action.setdefault(f"{d}-{i}", []).append(s)

        action = self._decode_act(delex_action, state)
        return action


class SGDActionEncoder:
    def __init__(
        self,
        tokenizer,
        output_act_prefix: str,
        decoder_prefix: str,
        end_token: str,
        lower_case: bool = False,
    ):
        self.tokenizer = tokenizer
        self.db = SGDDatabase()
        self.current_domain = "restaurant"
        self.start_token = output_act_prefix or decoder_prefix
        self.end_token = end_token
        self.lower_case = lower_case

    def get_out_dim(self) -> int:
        # not important
        return 1024

    def decode(
        self, response_ids: Iterable[int], state: Dict[str, Any]
    ) -> Dict[str, List[List[str]]]:
        response_str: str = self.tokenizer.decode(response_ids)
        _delex_actions: List[
            Tuple[str, str, str]
        ] = SGDDialogActTurnDataset.str_to_diaact(
            response_str,
            start_token=self.start_token,
            end_token=self.end_token,
        )
        delex_actions = []
        for domain, intent, slot in _delex_actions:
            if domain in ["", "none"]:
                belief_state = {}
                results = []
            else:
                belief_state = state["belief_state"][domain.lower()]
                results = self.db.query(
                    domain=domain.lower(),
                    constraints=belief_state,
                )
            # choose the one with the most slots
            result = {}
            max_k = 0
            for r in results:
                if len(r) > max_k:
                    max_k = len(r)
                    result = r

            if intent.lower() == "request":
                delex_actions.append([intent, domain, slot, "?"])
            elif intent.lower() == "informcount":
                delex_actions.append([intent, domain, slot, f"{len(results)}"])
            elif intent.lower() in ["confirm", "inform", "offer"]:
                if (
                    slot.lower() in belief_state
                    and belief_state[slot.lower()] not in NUL_VALUE
                ):
                    delex_actions.append(
                        [intent, domain, slot, belief_state[slot.lower()]]
                    )
                elif len(results) > 0 and slot.lower() in result:
                    delex_actions.append(
                        [intent, domain, slot, f"{result[slot.lower()]}"]
                    )
                elif len(results) > 0:  # slot not in db result
                    delex_actions.append(
                        [intent, domain, slot, "not available"]
                    )
                else:
                    delex_actions.append([intent, domain, slot, "none"])
            else:
                delex_actions.append([intent, domain, slot, "none"])

        act_dict = get_diaact_dict(delex_actions)
        if self.lower_case:
            act_dict = json.loads(json.dumps(act_dict).lower())
        return act_dict


class DAGPTActionEncoder3(DAGPTActionEncoder2):
    def __init__(
        self,
        tokenizer,
        output_act_prefix: str,
        decoder_prefix: str,
        end_token: str,
        seeder: Dict[str, Any] = {},
    ):
        self.tokenizer = tokenizer
        self.db = Database(seeder=seeder)
        self.current_domain = "Restaurant"
        self.start_token = output_act_prefix or decoder_prefix
        self.end_token = end_token
        self.seeder = seeder

    def get_out_dim(self) -> int:
        # not important
        return 1024

    def decode(
        self, response_ids: Iterable[int], state: Dict[str, Any]
    ) -> Dict[str, List[List[str]]]:
        """
        Args:
            response_ids (Iterable[int]): Array of response input ids.
            state (Dict[str, Any]): The state.

        Returns:
            Dict[str, List[List[str,str]]]: Map 'd-i' to [[s,v]].

        """
        input_str: str = self.tokenizer.decode(response_ids)
        _delex_action: List[
            Tuple[str, str, str]
        ] = DialogActTurnDataset.str_to_diaact(
            input_str, start_token=self.start_token, end_token=self.end_token
        )
        delex_action: Dict[str, List[str]] = {}
        for d, i, s in _delex_action:
            da = f"{d}-{i}"
            if da not in delex_action:
                delex_action[da] = [s]
            elif s not in delex_action[da]:
                delex_action[da].append(s)
            else:
                pass

        action = self._decode_act(delex_action, state)
        return action
