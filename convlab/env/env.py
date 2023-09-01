from typing import List, Union, Dict, Any, Tuple
from convlab.model.nlg.nlg import NLG
from convlab.model.dst.dst import DST
from convlab.model.nlu.nlu import NLU
from convlab.model.action_encoder.action_encoder import ActionEncoder
from convlab.model.state_encoder.state_encoder import StateEncoder
from convlab.dialog_agent.agent import PipelineAgent
from convlab.env.multiwoz_evaluator import Evaluator
from convlab.utils.misc import set_seed
from gym import spaces
import numpy as np
import gym
import json


class Environment(gym.Env):
    def __init__(
        self,
        sys_nlg: NLG,
        usr: PipelineAgent,
        sys_nlu: NLU,
        sys_dst: DST,
        state_encoder: StateEncoder,
        action_encoder: ActionEncoder = None,
        evaluator: Evaluator = None,
        encode_state: bool = True,
        return_history: bool = False,
    ):
        super().__init__()
        self.sys_nlg = sys_nlg
        self.usr = usr
        self.sys_nlu = sys_nlu
        self.sys_dst = sys_dst
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder

        self.evaluator = evaluator
        self.encode_state = encode_state
        self.return_history = return_history

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(state_encoder.get_out_dim(),), dtype=np.int32
        )
        self.action_space = spaces.Discrete(self.action_encoder.get_out_dim())
        self.turn = 0
        self.last_domain: str = ""

    def seed(self, seed_num: int):
        set_seed(seed_num)

    def reset(self):
        self.turn = 0
        self.last_domain: str = ""
        self.usr.init_session()
        self.sys_dst.init_session()
        self.state_encoder.init_session()
        if self.evaluator:
            self.evaluator.add_goal(self.usr.policy.get_goal())
        s, r, t, i = self.step({})
        return s

    def step(self, action: Union[int, Dict[str, List[List[str]]]]):
        self.turn += 1
        if not isinstance(action, dict):
            action: Dict[str, List[List[str]]] = self.action_encoder.decode(
                action, self.sys_dst.state
            )

        self.sys_dst.state["system_action"] = action
        if action:
            self.last_domain = list(action.keys())[0].split("-")[0]
        model_response = (
            self.sys_nlg.generate(action) if self.sys_nlg else action
        )
        observation = self.usr.response(model_response)
        # self.sys_dst.state["history"].append(["sys", model_response])
        # self.sys_dst.state["history"].append(["usr", observation])
        self.sys_dst.state["history"].append([model_response, observation])

        if self.evaluator:
            self.evaluator.add_sys_da(self.usr.get_in_da())
            self.evaluator.add_usr_da(self.usr.get_out_da())
        dialog_act = (
            self.sys_nlu.predict(observation) if self.sys_nlu else observation
        )
        self.sys_dst.state["user_action"] = dialog_act
        state = self.sys_dst.update(dialog_act)
        s_vec = self.state_encoder.encode(state)
        if self.encode_state:
            state = s_vec

        reward = self._get_reward()
        # reward = self.usr.policy.policy.get_reward()
        terminated = self.usr.is_terminated()
        # domain_success defined by my function
        i_r, domain_success = None, None
        i_info = self.usr.get_in_reward(self.last_domain)
        if i_info is not None:
            i_r, domain_success = i_info
        info = {
            "turn": self.turn,
            "domain_success": domain_success,
            "i_r": i_r,
        }
        if terminated:
            info["success"] = self._get_success()
        if self.return_history:
            info["history"] = json.dumps(self.sys_dst.state["history"])
        return state, reward, terminated, info

    def _get_success(self):
        if self.evaluator:
            return self.evaluator.task_success()
        else:
            p = self.usr.policy
            if hasattr(p, "policy"):
                p = p.policy
            return p.is_success()

    def _get_reward(self):
        # Note usr.policy.turn stands for one party turn
        # Note that there are some weird case where the task succeeded
        # but it's not terminated.
        p = self.usr.policy
        if hasattr(p, "policy"):
            p = p.policy
        if self.usr.is_terminated() and self._get_success():
            return 2 * p.max_turn
        elif self.usr.is_terminated():
            return -1 * p.max_turn
        else:
            return -1

    def get_state(self) -> Dict[str, Any]:
        return self.sys_dst.state

    def get_goal(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Returns:
            domain -> 'info'/'reqt' -> slot value pairs
        """
        return self.usr.policy.get_goal()

    def get_previous_acts(
        self, role="usr", intents=[]
    ) -> List[Tuple[str, str, str]]:
        histories = json.loads(json.dumps(self.get_state()["history"]).lower())
        if len(histories) <= 1:
            return []
        elif role == "usr":
            actions = [acts[1] for acts in histories[:-1]]
        else:
            actions = [acts[0] for acts in histories[:-1]]
        filtered_actions = []
        for act in actions:
            for diaact, svs in act.items():
                domain, intent = diaact.split("-")
                if len(intents) == 0:
                    for s, v in svs:
                        filtered_actions.append((domain, intent, s))
                elif intent in intents:
                    for s, v in svs:
                        filtered_actions.append((domain, intent, s))
        return filtered_actions
