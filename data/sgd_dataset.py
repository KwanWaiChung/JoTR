from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from util import get_logger
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Union
import random
import pickle
import torch
import json
import os


logger = get_logger()


class SGDDialogActTurnDataset(Dataset):
    domain_mapping = {
        k.lower(): v.lower()
        for k, v in json.load(
            open("data/sgd/mapping_domain.json", "r")
        ).items()
    }
    intent_mapping = {
        k.lower(): v.lower()
        for k, v in json.load(
            open("data/sgd/mapping_intent.json", "r")
        ).items()
    }
    slot_mapping = {
        k.lower(): v.lower() if isinstance(v, str) else v
        for k, v in json.load(open("data/sgd/mapping_slot.json", "r")).items()
    }
    domains = []
    for domain in json.load(open("data/sgd/domain.json", "r")):
        if "_" in domain:
            domain = domain.split("_")[0]
        domain = domain_mapping.get(domain.lower(), domain)
        domains.append(domain)

    intents = []
    for intent in json.load(open("data/sgd/intent.json", "r")):
        intents.append(intent_mapping.get(intent.lower(), intent))

    slots = []
    for slot in json.load(open("data/sgd/slot.json", "r")):
        normed_slot = slot_mapping.get(slot.lower(), slot)
        if normed_slot == []:
            normed_slot = slot
        slots.append(normed_slot)

    def __init__(
        self,
        mode: str,
        context_tokenizer,
        response_tokenizer=None,
        decoder_prefix: str = "",
        output_act_prefix: str = "",
        usr_diaact_prefix: str = "",
        sys_diaact_prefix: str = "",
        belief_prefix: str = "",
        db_prefix: str = "",
        end_token: str = "[end]",
        max_resp_len: int = 256,
        load_path: str = None,
        save_path: str = None,
        overwrite_cache: bool = False,
        character: str = "all",
        remove_belief_value: bool = False,
        add_repeat_act_num: bool = False,
        n_samples: int = -1,
    ):
        if mode not in ["train", "validation", "test"]:
            raise ValueError(
                "`mode` can either be train, validation or test. Received:"
                f" {mode}"
            )
        self.context_tokenizer = context_tokenizer
        self.response_tokenizer = response_tokenizer
        self.decoder_prefix = decoder_prefix
        self.output_act_prefix = output_act_prefix
        self.usr_diaact_prefix = usr_diaact_prefix
        self.sys_diaact_prefix = sys_diaact_prefix
        self.belief_prefix = belief_prefix
        self.db_prefix = db_prefix
        self.end_token = end_token
        self.max_resp_len = max_resp_len

        if (
            load_path is None
            or overwrite_cache
            or not os.path.isfile(load_path + f"_{mode}.pkl")
        ):
            raw_data = json.load(open(f"data/sgd/dialogues.json", "r"))
            self.data = []
            for session in tqdm(raw_data, desc="Processing data"):
                if session["data_split"] != mode:
                    continue
                usr_da_count = defaultdict(int)
                sys_da_count = defaultdict(int)
                turn_dict = {
                    "diaact": [],
                    "diaact_str": [],
                    "belief_state": [],
                    "belief_str": [],
                    "db_str": [],
                    "dialogue_id": session["dialogue_id"],
                }
                last_belief = {}
                for i, turn in enumerate(session["turns"]):
                    da: List[Tuple[str, str, str]] = self.transform_data_act(
                        turn["dialogue_acts"]
                    )
                    da_str, belief_str, db_str = self.get_turn_str(
                        da,
                        last_belief,
                        turn.get("db_results", {}),
                        remove_belief_value=remove_belief_value,
                        repeat_count=(
                            usr_da_count if i % 2 == 0 else sys_da_count
                        )
                        if add_repeat_act_num
                        else None,
                    )
                    turn_dict["diaact"].append(da)
                    turn_dict["diaact_str"].append(da_str)
                    turn_dict["belief_state"].append(last_belief)
                    turn_dict["belief_str"].append(belief_str)
                    turn_dict["db_str"].append(db_str)
                    turn_dict["turn"] = i
                    turn_dict["speaker"] = turn["speaker"]
                    if turn["speaker"] == "system":
                        for a in da:
                            sys_da_count[a] += 1
                            if character in ["sys", "all"]:
                                self.data.append(self._copy_dict(turn_dict))
                    else:
                        for a in da:
                            usr_da_count[a] += 1
                            if character in ["usr", "all"]:
                                self.data.append(self._copy_dict(turn_dict))
                    last_belief = turn.get("state", {})
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

    @staticmethod
    def _copy_dict(d: Dict[str, List[Any]]):
        """This is a customize method to copy a dict that is much faster than
        deep copy. I assume that I will only append things on the List and
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
    def transform_data_act(
        data_action,
        add_value: bool = False,
        lower_case: bool = False,
        domain_first=True,
    ) -> Union[List[Tuple[str, str, str]], List[Tuple[str, str, str, str]]]:
        action_list = []
        for _, dialog_act in data_action.items():
            for act in dialog_act:
                value = act.get("value", "none")
                if not value:
                    if "request" in act["intent"]:
                        value = "?"
                    else:
                        value = "none"
                domain = SGDDialogActTurnDataset.norm_domain(act["domain"])
                if not domain:
                    domain = "none"
                intent = SGDDialogActTurnDataset.intent_mapping.get(
                    act["intent"].lower(), act["intent"]
                )
                if not intent:
                    intent = "none"
                slot = SGDDialogActTurnDataset.norm_slot(act["slot"])
                if not slot:
                    slot = "none"
                if lower_case:
                    domain, intent, slot, value = (
                        domain.lower(),
                        intent.lower(),
                        slot.lower(),
                        value.lower(),
                    )
                if not domain_first:
                    domain, intent = intent, domain
                if add_value:
                    action_list.append((domain, intent, slot, value))
                else:
                    action_list.append((domain, intent, slot))
        return sorted(action_list)

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
            domain_token = SGDDialogActTurnDataset.norm_domain(token)
            intent_token = SGDDialogActTurnDataset.intent_mapping.get(
                token.lower(), token
            )
            slot_token = SGDDialogActTurnDataset.norm_slot(token)
            if token == end_token:
                break
            elif token.lower() in [
                "bye",
                "reqmore",
                "affirm",
                "negate",
                "thankyou",
            ]:
                diaacts.append(("", token, ""))
                prev_domain = ""
                prev_intent = ""
            elif prev_domain == "":
                if domain_token.lower() in SGDDialogActTurnDataset.domains + [
                    "none"
                ]:
                    prev_domain = token
            elif prev_intent == "":
                if intent_token.lower() in SGDDialogActTurnDataset.intents + [
                    "none"
                ]:
                    prev_intent = token
                else:
                    prev_domain = ""
            elif slot_token.lower() in SGDDialogActTurnDataset.slots + [
                "none"
            ]:
                diaacts.append((prev_domain, prev_intent, token))
                prev_domain = ""
                prev_intent = ""
            else:
                prev_domain = ""
                prev_intent = ""
        return diaacts

    @staticmethod
    def get_turn_str(
        diaacts: List[Tuple[str, str, str]],
        belief_state: Dict[str, Dict[str, str]] = {},
        db_results: Dict[str, List[Dict[str, str]]] = {},
        remove_belief_value: bool = False,
        repeat_count: Dict[Tuple[str, str, str], int] = None,
    ) -> Tuple[str, str, str]:
        diaact_str = ""
        belief_str = ""
        db_str = ""
        for da in diaacts:
            diaact_str += f"{da[0]} {da[1]} {da[2]} "
            if repeat_count is not None:
                count = repeat_count.get(da, 0)
                diaact_str += f"{count} "
        if belief_state:
            for domain, svs in belief_state.items():
                domain = SGDDialogActTurnDataset.norm_domain(domain)
                for s, v in svs.items():
                    if v != "":
                        s = SGDDialogActTurnDataset.norm_slot(s)
                        if remove_belief_value:
                            belief_str += f"{domain} {s} "
                        else:
                            belief_str += f"{domain} {s} {v} "
        if db_results:
            for domain, results in db_results.items():
                domain = SGDDialogActTurnDataset.norm_domain(domain)
                db_str += f"{domain} {len(results)} "
        return diaact_str.strip(), belief_str.strip(), db_str.strip()

    @staticmethod
    def norm_domain(domain: str) -> str:
        if "_" in domain:
            domain = domain.split("_")[0]
        domain = SGDDialogActTurnDataset.domain_mapping.get(
            domain.lower(), domain
        )
        return domain

    @staticmethod
    def norm_intent(intent: str) -> str:
        intent = SGDDialogActTurnDataset.intent_mapping.get(
            intent.lower(), intent
        )
        return intent

    @staticmethod
    def norm_slot(slot: str) -> str:
        normed_slot = SGDDialogActTurnDataset.slot_mapping.get(
            slot.lower(), slot
        )
        if normed_slot == []:
            normed_slot = slot
        return normed_slot

    def _get_vocab(self, is_response: bool = True) -> List[str]:
        sequences = [" ".join(turn["diaact_str"]) for turn in self.all_data]
        if not is_response:
            sequences += [
                " ".join(turn["belief_str"] + turn["db_str"])
                for turn in self.all_data
            ]
        vocabs = set()
        for utt in sequences:
            tokens = utt.split()
            for token in tokens:
                vocabs.add(token)
        vocabs.update(
            set(
                [
                    self.decoder_prefix,
                    self.end_token,
                    self.output_act_prefix,
                ]
            )
        )
        return sorted(list(vocabs))

    def __getitem__(self, index: Any):
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
        else:  # usr turn
            last_usr_diaact_ids = self.context_tokenizer.encode(
                f"{self.usr_diaact_prefix} {data['diaact_str'][-3] if len(data['diaact_str']) > 2 else ''}"
                .strip()
            )
            last_sys_diaact_ids = self.context_tokenizer.encode(
                f"{self.sys_diaact_prefix} {data['diaact_str'][-2] if len(data['diaact_str']) > 1 else ''}"
                .strip()
            )
        # belief state is only availabe in usr turn.
        belief_ids = self.context_tokenizer.encode(
            f"{self.belief_prefix} {data['belief_str'][-1].strip()}"
        )
        db_ids = self.context_tokenizer.encode(
            f"{self.db_prefix} {data['db_str'][-1]}".strip()
        )
        output_str = f"{self.decoder_prefix} "
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
        turns = torch.tensor(int(data["turn"] / 2), dtype=torch.long)
        return (
            data["dialogue_id"],
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
            data["db_str"],
        )

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        (
            dialogue_ids,
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
            dialogue_ids,
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
            db_strs,
        )
