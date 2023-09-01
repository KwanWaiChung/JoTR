from convlab.model.state_encoder.multiwoz_state_encoder import (
    MultiwozStateEncoder,
)
from convlab.data.multiwoz.dbquery2 import Database
from convlab.data.sgd.dbquery import Database as SGDDatabase
from data.dataset import DialogActSessionDataset, DialogActTurnDataset
from data.sgd_dataset import SGDDialogActTurnDataset
from typing import List, Dict, Any, Tuple
from util import diaact_to_str, get_diaact_list
from collections import defaultdict
import json
import torch

categories = ["domain", "intent", "slot", "db"]
symbols = ["[d]", "[i]", "[s]", "[db]"]
domains = list(json.load(open("data/multiwoz/domain.json", "r")).keys())
intents = list(json.load(open("data/multiwoz/intent.json", "r")).keys())
slots = list(json.load(open("data/multiwoz/slot.json", "r")).keys())


class SGDStateEncoder:
    def __init__(
        self,
        tokenizer,
        usr_diaact_prefix: str,
        sys_diaact_prefix: str,
        belief_prefix: str,
        db_prefix: str,
        remove_belief_value: bool = False,
        add_repeat_act_num: bool = False,
    ):
        self.usr_diaact_prefix = usr_diaact_prefix
        self.sys_diaact_prefix: str = sys_diaact_prefix
        self.belief_prefix: str = belief_prefix
        self.db_prefix: str = db_prefix
        self.db = SGDDatabase()
        self.tokenizer = tokenizer
        self.remove_belief_value = remove_belief_value
        self.add_repeat_act_num = add_repeat_act_num

    def encode(self, state) -> List[torch.Tensor]:
        """

        Args:
            state (_type_): _description_

        Returns:
            List[torch.Tensor]: 5 Tensors corresponding to
                last_usr_diaact_ids of shape (T1)
                last_sys_diaact_ids of shape (T2)
                belief_ids of shape (T3)
                db_ids of shape (T4)
                turn of shape () (i.e. scalar)
        """
        sys_act: List[List[str]] = state["system_action"]
        usr_act: List[List[str]] = state["user_action"]
        if isinstance(sys_act, dict):
            sys_act = get_diaact_list(
                sys_act, intent_first=False, add_value=False
            )
        if isinstance(usr_act, dict):
            usr_act = get_diaact_list(
                usr_act, intent_first=False, add_value=False
            )

        belief_state = state["belief_state"]
        domain = usr_act[0][0]
        db_results = {}
        if domain in belief_state:
            db_results[domain] = self.db.query(domain, belief_state[domain])
        (
            sys_act_str,
            belief_str,
            db_str,
        ) = SGDDialogActTurnDataset.get_turn_str(
            sys_act,
            belief_state,
            db_results,
            remove_belief_value=self.remove_belief_value,
            repeat_count=self.sys_da_count
            if self.add_repeat_act_num
            else None,
        )
        (usr_act_str, *_) = SGDDialogActTurnDataset.get_turn_str(
            usr_act,
            {},
            {},
            remove_belief_value=self.remove_belief_value,
            repeat_count=self.usr_da_count
            if self.add_repeat_act_num
            else None,
        )
        last_usr_diaact_ids = torch.tensor(
            self.tokenizer.encode(
                f"{self.usr_diaact_prefix} {usr_act_str}".strip()
            ),
            dtype=torch.long,
        )
        last_sys_diaact_ids = torch.tensor(
            self.tokenizer.encode(
                f"{self.sys_diaact_prefix} {sys_act_str}".strip()
            ),
            dtype=torch.long,
        )
        belief_ids = torch.tensor(
            self.tokenizer.encode(
                f"{self.belief_prefix} {belief_str}".strip()
            ),
            dtype=torch.long,
        )
        db_ids = torch.tensor(
            self.tokenizer.encode(f"{self.db_prefix} {db_str}".strip()),
            dtype=torch.long,
        )
        self.turn += 1
        for da_tuple in usr_act:
            self.usr_da_count[tuple(da_tuple)] += 1
        for da_tuple in sys_act:
            self.sys_da_count[tuple(da_tuple)] += 1
        return (
            last_usr_diaact_ids,
            last_sys_diaact_ids,
            belief_ids,
            db_ids,
            torch.tensor(self.turn, dtype=torch.long),
        )

    def get_out_dim(self) -> int:
        # not important
        return 1024

    def init_session(self):
        self.turn = -1
        self.usr_da_count = defaultdict(int)
        self.sys_da_count = defaultdict(int)


class DAGPTStateEncoder:
    def __init__(
        self,
        tokenizer,
        usr_diaact_prefix: str,
        sys_diaact_prefix: str,
        belief_prefix: str,
        db_prefix: str,
        seeder: Dict[str, Any] = {},
        remove_belief_value: bool = False,
        add_repeat_act_num: bool = False,
    ):
        self.usr_diaact_prefix = usr_diaact_prefix
        self.sys_diaact_prefix: str = sys_diaact_prefix
        self.belief_prefix: str = belief_prefix
        self.db_prefix: str = db_prefix
        self.db = Database(seeder=seeder)
        self.tokenizer = tokenizer
        self.remove_belief_value = remove_belief_value
        self.add_repeat_act_num = add_repeat_act_num

    def encode(self, state) -> List[torch.Tensor]:
        """

        Args:
            state (_type_): _description_

        Returns:
            List[torch.Tensor]: 5 Tensors corresponding to
                last_usr_diaact_ids of shape (T1)
                last_sys_diaact_ids of shape (T2)
                belief_ids of shape (T3)
                db_ids of shape (T4)
                turn of shape () (i.e. scalar)
        """
        # d-i: [[s,v]]
        sys_act: Dict[str, List[List[str]]] = state["system_action"]
        usr_act: Dict[str, List[List[str]]] = state["user_action"]
        belief_state: Dict[str, List[List[str]]] = state["belief_state"]
        (
            sys_act_str,
            belief_str,
            db_str,
            _,
            _,
        ) = DialogActTurnDataset.diaact_to_str(
            sys_act,
            belief_state,
            remove_belief_value=self.remove_belief_value,
            repeat_count=self.sys_da_count
            if self.add_repeat_act_num
            else None,
        )
        usr_act_str, *_ = DialogActTurnDataset.diaact_to_str(
            usr_act,
            {},
            repeat_count=self.usr_da_count
            if self.add_repeat_act_num
            else None,
        )
        last_usr_diaact_ids = torch.tensor(
            self.tokenizer.encode(
                f"{self.usr_diaact_prefix} {usr_act_str}".strip()
            ),
            dtype=torch.long,
        )
        last_sys_diaact_ids = torch.tensor(
            self.tokenizer.encode(
                f"{self.sys_diaact_prefix} {sys_act_str}".strip()
            ),
            dtype=torch.long,
        )
        belief_ids = torch.tensor(
            self.tokenizer.encode(
                f"{self.belief_prefix} {belief_str}".strip()
            ),
            dtype=torch.long,
        )
        db_ids = torch.tensor(
            self.tokenizer.encode(f"{self.db_prefix} {db_str}".strip()),
            dtype=torch.long,
        )
        self.turn += 1
        da_tuples: List[Tuple[str, str, str]] = get_diaact_list(usr_act)
        for da_tuple in da_tuples:
            self.usr_da_count[da_tuple] += 1
        da_tuples: List[Tuple[str, str, str]] = get_diaact_list(sys_act)
        for da_tuple in da_tuples:
            self.sys_da_count[da_tuple] += 1
        return (
            last_usr_diaact_ids,
            last_sys_diaact_ids,
            belief_ids,
            db_ids,
            torch.tensor(self.turn, dtype=torch.long),
        )

    def init_session(self):
        self.turn = -1
        self.usr_da_count = defaultdict(int)
        self.sys_da_count = defaultdict(int)

    def get_out_dim(self) -> int:
        # not important
        return 1024


class DAGPTStateEncoder:
    def __init__(
        self,
        tokenizer,
        seeder: Dict[str, Any] = {},
    ):
        self.db = Database(seeder=seeder)
        self.tokenizer = tokenizer

    def encode(self, state) -> List[int]:
        # d-i: [[s,v]]
        system_action: Dict[str, List[List[str]]] = state["system_action"]
        user_action: Dict[str, List[List[str]]] = state["user_action"]

        if self.turn > 0:
            self._encode_da(system_action, state["belief_state"])
        self._encode_da(user_action)
        self.turn += 1
        return self._encode_flatten_diaact()

    def _encode_da(self, da: Dict[str, List[List[str]]], belief_state={}):
        """

        Args:
            da (Dict[str, List[List[str]]]):  d-i -> [[s,v]]
            belief_state (dict, optional): _description_. Defaults to {}.

        """
        turn_str, domains = diaact_to_str(da)
        if belief_state != {}:
            for domain in domains:
                if domain.lower() not in ["booking", "general"]:
                    entities = self.db.query(
                        domain.lower(),
                        tuple(belief_state[domain.lower()]["semi"].items()),
                    )
                    turn_str += f"{len(entities)} matches of {domain}; "
        turn_str += "[te]"
        self.flatten_diaact.append(turn_str)

    def _encode_flatten_diaact(self) -> List[int]:
        flatten_diaact_str = " ".join(self.flatten_diaact).strip()
        prev_symb = -1
        input_ids = []
        for token in flatten_diaact_str.split():
            if token in symbols:
                input_ids.append(self.tokenizer.convert_tokens_to_ids(token))
                prev_symb = symbols.index(token)
            elif prev_symb == 0:  # domain
                input_ids += self.tokenizer(token, add_prefix_space=True)[
                    "input_ids"
                ]
            elif prev_symb == 1:  # intent
                input_ids += self.tokenizer(token, add_prefix_space=True)[
                    "input_ids"
                ]
            elif prev_symb == 2:  # slot
                input_ids += self.tokenizer(token, add_prefix_space=True)[
                    "input_ids"
                ]
            elif prev_symb == 3:  # db
                input_ids += self.tokenizer(token, add_prefix_space=True)[
                    "input_ids"
                ]
            else:
                raise ValueError("Invalid condition.")
        return input_ids

    def init_session(self):
        self.turn = 0
        self.flatten_diaact = []

    def get_out_dim(self) -> int:
        # not important
        return 1024
