import numpy as np
from convlab.data.sgd.info import (
    init_state,
    NOT_MENTIONED,
    DONT_CARE,
    INIT_BELIEF_STATE,
    REF_USR_DA,
)
from convlab.model.state_encoder.state_encoder import StateEncoder
from convlab.data.sgd.dbquery import Database
from typing import Dict, List, Tuple


class SGDStateEncoder(StateEncoder):
    """The class is the multi-hot basic state encoder."""

    def __init__(self):
        self.domains = list(INIT_BELIEF_STATE)
        self.db = Database()

    def encode(self, state):
        db_vector = self.get_db_state(state["belief_state"])
        info_vector = self.get_info_state(state["belief_state"])
        request_vector = self.get_request_state(state["request_state"])
        user_act_vector = self.get_user_act_state(state["user_action"])
        history_vector = self.get_history_state(state["history"])

        return np.concatenate(
            (
                db_vector,
                info_vector,
                request_vector,
                user_act_vector,
                history_vector,
            )
        )

    def init_session(self):
        pass

    def get_out_dim(self) -> int:
        return self.encode(init_state()).shape[0]

    def get_db_state(self, belief_state) -> np.ndarray:
        vector = np.zeros(6 * len(self.domains))
        if belief_state == {}:
            return vector

        for idx, domain in enumerate(self.domains):
            entities = self.db.query(domain, belief_state[domain].items())
            num = len(entities)
            if num == 0:
                vector[idx * 6 : idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
            elif num == 1:
                vector[idx * 6 : idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
            elif num == 2:
                vector[idx * 6 : idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
            elif num == 3:
                vector[idx * 6 : idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
            elif num == 4:
                vector[idx * 6 : idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
            elif num >= 5:
                vector[idx * 6 : idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
        return vector

    def get_info_state(
        self, belief_state: Dict[str, Dict[str, str]]
    ) -> np.ndarray:
        """

        Args:
            belief_state (Dict[str, Dict[str, str]]): Dict of domain
                to Dict of slot values pairs.

        Returns:
            np.ndarray: size of number of slot value pairs in all domains.

        """
        vector = []
        for domain in self.domains:
            domain_active = False
            for s, v in sorted(belief_state[domain].items()):
                slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
                if v in NOT_MENTIONED:
                    slot_enc[0] = 1
                elif v in DONT_CARE:
                    slot_enc[1] = 1
                    domain_active = True
                else:
                    slot_enc[2] = 1
                    domain_active = True
                vector += slot_enc
            vector += [1] if domain_active else [0]
        return np.array(vector)

    def _decode_info_state(
        self, info_vec: np.ndarray
    ) -> Tuple[
        Dict[str, List[str]],
        Dict[str, List[str]],
        Dict[str, List[str]],
        List[str],
    ]:
        """_summary_

        Args:
            info_vec (np.ndarray): _description_

        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: Not mentioned
                slots, don't care slots, informed slots and active domains.

        """
        belief_state = INIT_BELIEF_STATE.copy()
        idx = 0
        nm_slots, dc_slots, i_slots = {}, {}, {}
        active_domains = []
        for domain in self.domains:
            nm_slots[domain] = []
            dc_slots[domain] = []
            i_slots[domain] = []
            for s in sorted(belief_state[domain]):
                vec = info_vec[idx : idx + 3]
                if vec[0] == 1:
                    nm_slots[domain].append(s)
                elif vec[1] == 1:
                    dc_slots[domain].append(s)
                else:
                    i_slots[domain].append(s)
                idx += 3
            if info_vec[idx] == 1:
                active_domains.append(domain)
            idx += 1
        assert idx == len(info_vec)
        return nm_slots, dc_slots, i_slots, active_domains

    def get_request_state(self, request_state) -> np.ndarray:
        vector = []
        for domain in self.domains:
            slots: List[str] = sorted(REF_USR_DA[domain])
            domain_vector = [0] * len(slots)
            if domain in request_state:
                for slot in request_state[domain]:
                    domain_vector[slots.index(slot)] = 1
            vector += domain_vector
        return np.array(vector)

    def _decode_request_state(
        self, request_vec: np.ndarray
    ) -> Dict[str, List[str]]:
        """_summary_

        Args:
            request_vec (np.ndarray): _description_

        Returns:
            Dict[str, List[str]]: Dict of domain to list of slots.

        """
        request_state = {}
        idx = 0
        for domain in self.domains:
            slots: List[str] = sorted(REF_USR_DA[domain])
            request_state[domain] = []
            for s in slots:
                if request_vec[idx] == 1:
                    request_state[domain].append(s)
                idx += 1
        assert idx == len(request_vec)
        return request_state

    def get_user_act_state(
        self, user_act: Dict[str, List[List[str]]]
    ) -> np.ndarray:
        user_act_vector = []

        for domain in self.domains:
            for slot in REF_USR_DA[domain]:
                for act_type in ["inform", "request"]:
                    domain_act = domain + "-" + act_type
                    if domain_act in user_act and slot in [
                        sv[0] for sv in user_act[domain_act]
                    ]:
                        user_act_vector.append(1)
                    else:
                        user_act_vector.append(0)
        return np.array(user_act_vector)

    def _decode_user_act_state(
        self, vec: np.ndarray
    ) -> List[Tuple[str, str, str]]:
        """_summary_

        Args:
            vec (np.ndarray): _description_

        Returns:
            List[Tuple[str, str, str]]: d, i, s.

        """
        user_act = []
        idx = 0
        for domain in self.domains:
            for slot in REF_USR_DA[domain]:
                for act_type in ["inform", "request"]:
                    if vec[idx] == 1:
                        user_act.append((domain, act_type, slot))
                    idx += 1
        assert len(vec) == idx
        return user_act

    def get_history_state(self, history):
        history_vector = []

        user_act = None
        repeat_count = 0
        user_act_repeat_vector = [0] * 5
        for turn in reversed(history):
            if user_act is None:
                user_act = turn[1]
            elif user_act == turn[1]:
                repeat_count += 1
            else:
                break
        user_act_repeat_vector[min(4, repeat_count)] = 1
        history_vector += user_act_repeat_vector

        return history_vector
