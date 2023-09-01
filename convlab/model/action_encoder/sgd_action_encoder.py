import json
from convlab.data.sgd.info import INIT_BELIEF_STATE, NUL_VALUE
from convlab.model.action_encoder.action_encoder import ActionEncoder
from convlab.data.sgd.dbquery import Database
from typing import Dict, List, Any
from util import get_diaact_dict

DEFAULT_VOCAB_FILE = "convlab/data/sgd/da_slot_cnt.json"


class ActionVocab:
    def __init__(self, vocab_path=DEFAULT_VOCAB_FILE, num_actions=500):
        # add general actions
        self.vocab = []
        # add single slot actions
        for domain in INIT_BELIEF_STATE:
            for slot in INIT_BELIEF_STATE[domain]:
                self.vocab.append({domain + "-inform": [slot]})
                self.vocab.append({domain + "-request": [slot]})
        # add actions from stats
        with open(vocab_path, "r") as f:
            stats = json.load(f)
            for act_str in stats:
                acts = act_str.split(";")
                act_dict = {}
                for act in acts:
                    intent, domain, slot = act.split(",")
                    di = f"{domain}-{intent}"
                    if di not in act_dict:
                        act_dict[di] = []
                    act_dict[di].append(slot)
                if act_dict not in self.vocab:
                    self.vocab.append(act_dict)
                if len(self.vocab) >= num_actions:
                    break
        print("{} actions are added to vocab".format(len(self.vocab)))
        # pprint(self.vocab)

    def get_action(self, action_index) -> Dict[str, List[str]]:
        return self.vocab[action_index]

    def get_size(self) -> int:
        return len(self.vocab)


class SGDActionEncoder(ActionEncoder):
    db = Database()

    def __init__(self, num_actions=500) -> None:
        self.action_vocab = ActionVocab(num_actions=num_actions)
        self.current_domain = ""

    def get_out_dim(self) -> int:
        return self.action_vocab.get_size()

    def encode(self, action: Dict[str, List[List[str]]]) -> int:
        # action: {'d-i': [[s, v]]}
        # I want: 'd-i': [s]
        delex_act = {}
        for di, svs in action.items():
            delex_act[di] = [sv[0] for sv in svs]
        return self._find_best_delex_act(delex_act)

    def decode(
        self, action_index: int, state: Dict[str, Any]
    ) -> Dict[str, List[List[str]]]:
        action_dict: Dict[str, List[str]] = self.get_action(action_index)
        action_list = [
            [*di.split("-"), s]
            for di, slots in action_dict.items()
            for s in slots
        ]
        action_dict: Dict[str, List[List[str]]] = self.add_act_value(
            action_list, state["belief_state"]
        )
        return action_dict

    def get_action(self, i: int) -> Dict[str, List[str]]:
        return self.action_vocab.get_action(i)

    @staticmethod
    def add_act_value(
        acts: List[List[str]], state: Dict[str, Any]
    ) -> Dict[str, List[List[str]]]:
        """Add appropriate action value by querying db.

        Args:
            acts (List[List[str]]): [domain, intent, slot].
            state (Dict[str, Any]): belief slot value pairs

        Returns:
            Dict[str, List[List[str]]]: 'd-i': [[s,v]]

        """
        delex_actions = []
        for domain, intent, slot in acts:
            if domain == "":
                belief_state = {}
                results = []
            else:
                belief_state = state[domain.lower()]
                results = SGDActionEncoder.db.query(
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

        act_dict = get_diaact_dict(delex_actions, intent_first=True)
        act_dict = json.loads(json.dumps(act_dict).lower())
        return act_dict

    @staticmethod
    def _score(a1, a2):
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
        best_p_action_index = -1
        best_p_score = float("inf")
        best_pn_action_index = -1
        best_pn_score = float("inf")
        for i, v_action in enumerate(self.action_vocab.vocab):
            if v_action == action:
                return i
            else:
                p_score = self._score(action, v_action)
                n_score = self._score(v_action, action)
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
