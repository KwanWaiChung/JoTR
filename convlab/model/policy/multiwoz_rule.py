# -*- coding: utf-8 -*-
from tianshou.data import Batch
from typing import Optional, Union, Any, Dict, List
import numpy as np
import torch
import random
from convlab.model.policy.policy import Policy
from convlab.model.action_encoder.multiwoz_action_encoder import (
    MultiwozActionEncoder,
)
from convlab.model.state_encoder.multiwoz_state_encoder import (
    MultiwozStateEncoder,
)

from convlab.model.policy.sys_multiwoz_rule import RuleBasedMultiwozBot

# from convlab.model.policy.usr_multiwoz_rule import UserPolicyAgendaMultiWoz
from convlab.model.policy.usr_multiwoz_rule2 import UserPolicyAgendaMultiWoz

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RulePolicy(Policy):
    def __init__(
        self,
        character="sys",
        domains=None,
        max_turn: int = 20,
        seeder: Dict[str, Any] = {},
        single_action=False,
        state_encoder=None,
        action_encoder=None,
    ):
        super().__init__(state_encoder=state_encoder)
        self.character = character
        self.action_encoder = action_encoder
        self.single_action = single_action
        self.seeder = seeder

        if character == "sys":
            self.policy = RuleBasedMultiwozBot(seeder)
        elif character == "usr":
            self.policy = UserPolicyAgendaMultiWoz(
                max_turn, domains, seeder=seeder
            )
        else:
            raise NotImplementedError("unknown character {}".format(character))

    def predict(self, state: Dict[str, Any]) -> Dict[str, List[List[str]]]:
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
            For usr_rule_policy, it will be system action dict. Because of
            the absence of DST in pipeline_agent.

        Returns:
            action: Mapping of 'd-i' to [[s,v]]
        """
        action = self.policy.predict(state)
        if self.single_action:
            # domain_intent = list(action.keys())[0]
            # # may be empty list
            # if action[domain_intent] == []:
            #     slot_value = ["none", "none"]
            # else:
            #     slot_value = action[domain_intent][0]

            domain_intent = self.seeder.get("py", random).choice(
                list(action.keys())
            )
            slot_value = self.seeder.get("py", random).choice(
                action[domain_intent]
                or [
                    [
                        "none",
                        "none",
                    ]
                ]
            )
            action = {domain_intent: [slot_value]}
        return action

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        # shape: (num_envs)
        actions: np.ndarray = np.stack(
            [self.action_encoder.encode(self.predict(s)) for s in state],
            axis=0,
        )
        batch.act = actions
        return batch

    def init_session(self, **kwargs):
        """
        Restore after one session
        """
        self.policy.init_session(**kwargs)

    def is_terminated(self):
        if self.character == "sys":
            return None
        return self.policy.is_terminated()

    def get_reward(self):
        if self.character == "sys":
            return None
        return self.policy.get_reward()

    def get_in_reward(self, domain):
        if self.character == "sys":
            return None
        return self.policy.get_in_reward(domain)

    def get_goal(self):
        if hasattr(self.policy, "get_goal"):
            return self.policy.get_goal()
        return None

    def update(self, sample_size, buffer):
        return {}


class SingleRulePolicy(RulePolicy):
    def predict(self, state):
        actions = self.policy.predict(state)
        return [self.seeder.get("py", random).choice(actions)]
