from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from util import get_logger, get_diaact_list
from typing import List, Tuple, Dict, Any
from copy import deepcopy
from convlab.data.multiwoz.dbquery2 import Database
from transformers import PreTrainedTokenizer
from data.tokenizer import FixedVocabTokenizer
from time import time
from collections import defaultdict
import numpy as np
import os
import pickle
import torch
import json
import random


logger = get_logger()


class DialogActTurnDataset(Dataset):
    db = Database()
    domains = list(json.load(open("data/multiwoz/domain.json", "r")).keys())
    intents = list(json.load(open("data/multiwoz/intent.json", "r")).keys())
    slots = list(json.load(open("data/multiwoz/slot.json", "r")).keys())

    def __init__(
        self,
        mode: str,
        context_tokenizer,
        response_tokenizer=None,
        n_samples: int = -1,
        usr_diaact_prefix: str = "",
        sys_diaact_prefix: str = "",
        belief_prefix: str = "",
        db_prefix: str = "",
        decoder_prefix: str = "",
        output_belief=False,
        output_belief_prefix: str = "",
        output_num_act=False,
        output_num_act_prefix: str = "",
        output_act_prefix: str = "",
        end_token: str = "[end]",
        max_resp_len: int = 256,
        load_path: str = None,
        save_path: str = None,
        overwrite_cache: bool = False,
        character: str = "all",
        remove_belief_value: bool = False,
        add_repeat_act_num: bool = False,
    ):
        """
        Args:
            mode (str): _description_
            n_samples (int): _description_
            max_seq_len (int, optional): _description_. Defaults to 1024.
            load_path (str, optional): _description_. Defaults to None.
            save_path (str, optional): _description_. Defaults to None.
            overwrite_cache (bool, optional): _description_. Defaults to False.
            character (str, optional): Either `all`, `sys` or `usr`.
            remove_belief_value (bool): If True, exclude the slot value.
            add_repeat_act_num (bool): If True, add the number of repeats of
                user action in the str.
        """
        self.n_samples = n_samples
        self.context_tokenizer = context_tokenizer
        self.response_tokenizer = response_tokenizer
        self.max_resp_len = max_resp_len
        self.character = character
        self.usr_diaact_prefix = usr_diaact_prefix
        self.sys_diaact_prefix = sys_diaact_prefix
        self.belief_prefix = belief_prefix
        self.db_prefix = db_prefix
        if decoder_prefix is None:
            raise ValueError("decoder_prefix can't be empty.")
        self.decoder_prefix = decoder_prefix
        self.output_belief = output_belief
        self.output_belief_prefix = output_belief_prefix
        self.output_num_act = output_num_act
        self.output_num_act_prefix = output_num_act_prefix
        self.output_act_prefix = output_act_prefix
        self.end_token = end_token
        self.remove_belief_value = remove_belief_value
        self.add_repeat_act_num = add_repeat_act_num
        if (
            load_path is None
            or overwrite_cache
            or not os.path.isfile(load_path + f"_{mode}.pkl")
        ):
            raw_data = json.load(open(f"data/multiwoz/{mode}.json", "r"))
            self.data = []
            for session_id, session in tqdm(raw_data.items()):
                turn_dict = {
                    "session_id": session_id,
                    "diaact": [],
                    "belief_state": [],
                    "change_belief_state": [],
                    # "db_result": [],
                    "diaact_str": [],
                    "belief_str": [],
                    "change_belief_str": [],
                    "db_str": [],
                }
                d1 = []
                d2 = []
                usr_da_count = defaultdict(int)
                sys_da_count = defaultdict(int)
                for turn_id, turn in enumerate(session["log"]):
                    da = turn["dialog_act"]
                    belief_state = turn["metadata"]
                    old_belief_state = (
                        session["log"][turn_id - 2]["metadata"]
                        if turn_id > 1
                        else {}
                    )
                    change_belief_state = {}
                    da_tuples: List[Tuple[str, str, str]] = get_diaact_list(da)
                    for domain in belief_state:
                        if domain not in old_belief_state:
                            change_belief_state[domain] = belief_state[domain]
                        else:
                            diff_dict = dict(
                                set(belief_state[domain]["semi"].items())
                                - set(old_belief_state[domain]["semi"].items())
                            )
                            if len(diff_dict) > 0:
                                change_belief_state[domain] = {}
                                change_belief_state[domain]["semi"] = diff_dict
                    start_time = time()
                    (
                        turn_str,
                        belief_str,
                        db_str,
                        domains,
                        db_result,
                    ) = self.diaact_to_str(
                        diaacts=da,
                        belief_state=belief_state,
                        remove_belief_value=remove_belief_value,
                        repeat_count=(
                            usr_da_count if turn_id % 2 == 0 else sys_da_count
                        )
                        if self.add_repeat_act_num
                        else None,
                    )
                    change_belief_str = self.diaact_to_str(
                        diaacts=da,
                        belief_state=change_belief_state,
                        remove_belief_value=self.remove_belief_value,
                    )[1]
                    d1.append(time() - start_time)
                    start_time = time()
                    turn_dict["diaact"].append(da)
                    turn_dict["belief_state"].append(belief_state)
                    turn_dict["change_belief_state"].append(
                        change_belief_state
                    )
                    # turn_dict["db_result"].append(db_result)
                    turn_dict["diaact_str"].append(turn_str)
                    turn_dict["belief_str"].append(belief_str)
                    turn_dict["change_belief_str"].append(change_belief_str)
                    turn_dict["db_str"].append(db_str)
                    turn_dict["turn_id"] = turn_id
                    if turn_id % 2 == 0:
                        for da_tuple in da_tuples:
                            usr_da_count[da_tuple] += 1
                        if character in ["usr", "all"]:
                            self.data.append(self._copy_dict(turn_dict))
                    elif turn_id % 2 == 1:
                        for da_tuple in da_tuples:
                            sys_da_count[da_tuple] += 1
                        if character in ["sys", "all"]:
                            self.data.append(self._copy_dict(turn_dict))
                    d2.append(time() - start_time)
                # print(sum(d1), np.mean(d1))
                # print(sum(d2), np.mean(d2))
            if save_path is not None:
                save_path = save_path + f"_{mode}.pkl"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                logger.info(f"Saving dataset cache to {save_path}")
                pickle.dump(
                    {
                        "data": self.data,
                    },
                    open(save_path, "wb"),
                )
        else:
            load_path = load_path + f"_{mode}.pkl"
            logger.info(f"Loading dataset cache from {load_path}")
            pkl_dict = pickle.load(open(load_path, "rb"))
            self.data = pkl_dict["data"]
        self.all_data = self.data
        if n_samples > 0:
            self.data = random.sample(self.data, k=n_samples)
            logger.info(f"Sampled {n_samples} samples.")

    def _get_vocab(self, is_response: bool = True) -> List[str]:
        if is_response:
            sequences = [
                " ".join(turn["diaact_str"]) for turn in self.all_data
            ]
            if self.output_belief:
                sequences += [
                    " ".join(turn["belief_str"]) for turn in self.all_data
                ]
            vocab = self.build_vocab(sequences)
            vocab += [
                self.decoder_prefix,
                self.end_token,
                self.output_belief_prefix,
                self.output_num_act_prefix,
                self.output_act_prefix,
            ]
            if self.output_num_act:
                vocab += [str(i) for i in range(1, 10)]
        else:
            sequences = [
                " ".join(
                    turn["diaact_str"] + turn["belief_str"] + turn["db_str"]
                )
                for turn in self.all_data
            ]

            vocab = self.build_vocab(sequences)
            vocab += [
                self.usr_diaact_prefix,
                self.sys_diaact_prefix,
                self.belief_prefix,
                self.db_prefix,
            ]
        return vocab

    def build_response_tokenizer(self):
        vocab = self._get_vocab(is_response=True)
        tokenizer = FixedVocabTokenizer(
            vocab=vocab,
            model_max_length=self.max_resp_len,
            eos_token=None,
            pad_token="[PAD]",
            unk_token="[UNK]",
            add_token_type=False,
        )
        return tokenizer

    def build_context_tokenizer(self):
        vocab = self._get_vocab(is_response=False)
        tokenizer = FixedVocabTokenizer(
            vocab=vocab,
            model_max_length=self.max_resp_len,
            eos_token=None,
            pad_token="[PAD]",
            unk_token="[UNK]",
            add_token_type=False,
            bos_token="[CLS]",
        )
        return tokenizer

    def __getitem__(self, index):
        """
        Returns:
            session_id: str.
            last_usr_diaact_ids: Tensor of shape (T1).
            last_usr_diaact_attn_mask: Tensor of shape (T1).
            last_sys_diaact_ids: Tensor of shape (T2).
            last_sys_diaact_attn_mask: Tensor of shape (T2).
            belief_ids: Tensor of shape (T3).
            belief_attn_mask: Tensor of shape (T3).
            db_ids: Tensor of shape (T4).
            db_attn_mask: Tensor of shape (T4).
            output_ids: Tensor of shape (T5).
            output_mask: Tensor of shape (T5).
            turns: Tensor of shape ().
            diaact_str: str.
            belief_str: str.
            change_belief_str: str.
            db_str: str.
        """
        data = self.data[index]
        # it's system turn
        if len(data["diaact_str"]) % 2 == 0:
            last_usr_diaact_ids = self.context_tokenizer.encode(
                f"{self.usr_diaact_prefix} {data['diaact_str'][-2] if len(data['diaact_str']) > 1 else ''}"
                .strip()
            )
            last_sys_diaact_ids = self.context_tokenizer.encode(
                f"{self.sys_diaact_prefix} {data['diaact_str'][-3] if len(data['diaact_str']) > 2 else ''}"
                .strip()
            )
        else:  # user turn
            last_usr_diaact_ids = self.context_tokenizer.encode(
                f"{self.usr_diaact_prefix} {data['diaact_str'][-3] if len(data['diaact_str']) > 2 else ''}"
                .strip()
            )
            last_sys_diaact_ids = self.context_tokenizer.encode(
                f"{self.sys_diaact_prefix} {data['diaact_str'][-2] if len(data['diaact_str']) > 1 else ''}"
                .strip()
            )
        belief_ids = self.context_tokenizer.encode(
            f"{self.belief_prefix} {data['belief_str'][-1]}".strip()
        )
        db_ids = self.context_tokenizer.encode(
            f"{self.db_prefix} {data['db_str'][-1]}".strip()
        )
        output_str = f"{self.decoder_prefix} "
        if self.output_belief:
            output_str += (
                f"{self.output_belief_prefix} {data['change_belief_str'][-1]} "
            )
        if self.output_num_act:
            num_act = 0
            da = data["diaact"][-1]
            for sv in da.values():
                num_act += len(sv)
            output_str += f"{self.output_num_act_prefix} {num_act} "

        output_str += (
            f"{self.output_act_prefix} {data['diaact_str'][-1]} {self.end_token}"
        )
        output_ids = self.response_tokenizer.encode(output_str.strip())

        last_usr_diaact_ids = torch.tensor(
            last_usr_diaact_ids, dtype=torch.long
        )
        last_usr_diaact_attn_mask = torch.ones_like(last_usr_diaact_ids)
        last_sys_diaact_ids = torch.tensor(
            last_sys_diaact_ids, dtype=torch.long
        )
        last_sys_diaact_attn_mask = torch.ones_like(last_sys_diaact_ids)
        belief_ids = torch.tensor(belief_ids, dtype=torch.long)
        belief_attn_mask = torch.ones_like(belief_ids)
        db_ids = torch.tensor(db_ids, dtype=torch.long)
        db_attn_mask = torch.ones_like(db_ids)
        output_ids = torch.tensor(output_ids, dtype=torch.long)
        output_mask = torch.ones_like(output_ids)
        turns = torch.tensor(int(data["turn_id"] / 2), dtype=torch.long)
        return (
            data["session_id"],
            last_usr_diaact_ids,
            last_usr_diaact_attn_mask,
            last_sys_diaact_ids,
            last_sys_diaact_attn_mask,
            belief_ids,
            belief_attn_mask,
            db_ids,
            db_attn_mask,
            output_ids,
            output_mask,
            turns,
            data["diaact"],
            data["diaact_str"],
            data["belief_str"],
            data["change_belief_str"],
            data["db_str"],
        )

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        """
        Returns:
            session_ids: List[str].
            last_usr_diaact_ids: Tensor of shape (B, T1).
            last_usr_diaact_attn_mask: Tensor of shape (B, T1).
            last_sys_diaact_ids: Tensor of shape (B, T2).
            last_sys_diaact_attn_mask: Tensor of shape (B, T2).
            belief_ids: Tensor of shape (B, T3).
            belief_attn_mask: Tensor of shape (B, T3).
            db_ids: Tensor of shape (B, T4).
            db_attn_mask: Tensor of shape (B, T4).
            output_ids: Tensor of shape (B, T5).
            output_mask: Tensor of shape (B, T5).
            output_type_ids: Tensor of shape (B, T5).
            turns: Tensor of shape (B).
            diaact: List[Dict]
            diaact_strs: List[str].
            belief_strs: List[str].
            change_belief_strs: List[str].
            db_strs: List[str].
        """
        (
            session_ids,
            last_usr_diaact_ids,
            last_usr_diaact_attn_mask,
            last_sys_diaact_ids,
            last_sys_diaact_attn_mask,
            belief_ids,
            belief_attn_mask,
            db_ids,
            db_attn_mask,
            output_ids,
            output_mask,
            turns,
            diaacts,
            diaact_strs,
            belief_strs,
            change_belief_strs,
            db_strs,
        ) = zip(*batch)
        last_usr_diaact_ids = pad_sequence(
            last_usr_diaact_ids,
            batch_first=True,
            padding_value=self.context_tokenizer.pad_token_id,
        )
        last_usr_diaact_attn_mask = pad_sequence(
            last_usr_diaact_attn_mask, batch_first=True, padding_value=0
        )
        last_sys_diaact_ids = pad_sequence(
            last_sys_diaact_ids,
            batch_first=True,
            padding_value=self.context_tokenizer.pad_token_id,
        )
        last_sys_diaact_attn_mask = pad_sequence(
            last_sys_diaact_attn_mask, batch_first=True, padding_value=0
        )
        belief_ids = pad_sequence(
            belief_ids,
            batch_first=True,
            padding_value=self.context_tokenizer.pad_token_id,
        )
        belief_attn_mask = pad_sequence(
            belief_attn_mask, batch_first=True, padding_value=0
        )
        db_ids = pad_sequence(
            db_ids,
            batch_first=True,
            padding_value=self.context_tokenizer.pad_token_id,
        )
        db_attn_mask = pad_sequence(
            db_attn_mask, batch_first=True, padding_value=0
        )
        output_ids = pad_sequence(
            output_ids,
            batch_first=True,
            padding_value=self.response_tokenizer.pad_token_id,
        )
        output_mask = pad_sequence(
            output_mask, batch_first=True, padding_value=0
        )
        turns = torch.stack(turns, dim=0)
        return (
            session_ids,
            last_usr_diaact_ids,
            last_usr_diaact_attn_mask,
            last_sys_diaact_ids,
            last_sys_diaact_attn_mask,
            belief_ids,
            belief_attn_mask,
            db_ids,
            db_attn_mask,
            output_ids,
            output_mask,
            turns,
            diaacts,
            diaact_strs,
            belief_strs,
            change_belief_strs,
            db_strs,
        )

    @staticmethod
    def _copy_dict(d: Dict[str, List[Any]]):
        """This is a customize method to copy a dict that is much faster than
        deep copy. I assume that I will only append things on the List but
        won't modify existing elements.
        """
        new_d = {}
        for k, v in d.items():
            if isinstance(v, (int, str)):
                new_d[k] = v
            else:
                new_d[k] = [*v]
        return new_d

    @staticmethod
    def build_vocab(utterances: List[str]) -> List[str]:
        vocabs = set()
        for utt in utterances:
            tokens = utt.split()
            for token in tokens:
                vocabs.add(token)
        return sorted(list(vocabs))

    @staticmethod
    def str_to_diaact(
        input_str: str, start_token: str, end_token: str
    ) -> List[Tuple[str, str, str]]:
        input_seq: List[str] = input_str.split()
        if start_token not in input_seq:
            return []
        i = input_seq.index(start_token)
        prev_domain = ""
        prev_intent = ""
        diaacts = []
        for token in input_seq[i + 1 :]:
            if token == end_token:
                break
            elif prev_domain == "":
                if token in DialogActTurnDataset.domains:
                    prev_domain = token
            elif prev_intent == "":
                if token in DialogActTurnDataset.intents:
                    prev_intent = token
                else:
                    prev_domain = ""
            elif token in DialogActTurnDataset.slots:
                diaacts.append((prev_domain, prev_intent, token))
                prev_domain = ""
                prev_intent = ""
            else:
                prev_domain = ""
                prev_intent = ""
        return diaacts

    @classmethod
    def diaact_to_str(
        cls,
        diaacts: Dict[str, List[List[str]]],
        belief_state: Dict = {},
        remove_belief_value: bool = False,
        repeat_count: Dict[Tuple[str, str, str], int] = None,
    ) -> str:
        """

        Args:
            diaacts (Dict[str, List[List[str]]]): Dict of d-i -> [[s,v]]

        Returns:
            turn_str: The diaact sequence.
            belief_str: The belief sequence.
            db_str: The db sequence.
            domains List[str]: The domains involved.
            db_results List[Dict[str, Any]]: If belief states are provided, it will
                return the number of matched entities in each domain.

        """
        turn_str = ""
        belief_str = ""
        db_str = ""
        domains = set()
        db_results = {}
        for di in diaacts.keys():
            domain, intent = di.split("-")
            domains.add(domain)
            for sv in diaacts[di]:
                turn_str += f"{domain} {intent} {sv[0]} "
                if repeat_count is not None:
                    count = repeat_count.get((domain, intent, sv[0]), 0)
                    turn_str += f"{count} "
        if belief_state != {}:
            # for domain in domains:
            #     if domain.lower() not in belief_state:
            #         continue
            for domain in belief_state.keys():
                if domain.lower() not in ["booking", "general"]:
                    sv = tuple(belief_state[domain.lower()]["semi"].items())
                    add_domain_name = False
                    for s, v in sv:
                        if v != "" and v != "not mentioned":
                            if remove_belief_value:
                                belief_str += f"{domain} {s} "
                            else:
                                if not add_domain_name:
                                    belief_str += f"{domain} "
                                    add_domain_name = True
                                belief_str += f"{s} {v} "
                    if add_domain_name:
                        belief_str += "; "
                    if domain in [
                        "restaurant",
                        "hotel",
                        "attraction",
                        "train",
                    ]:
                        entities = cls.db.query(
                            domain.lower(),
                            sv,
                        )
                        db_str += f"{domain} {len(entities)} "
                        db_results[domain] = entities
        return (
            turn_str.strip(),
            belief_str.strip(),
            db_str.strip(),
            domains,
            db_results,
        )

