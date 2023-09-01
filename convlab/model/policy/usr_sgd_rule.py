#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""


import copy
import json
import random
import re

from typing import List, Tuple, Dict, Union, Any
from convlab.data.sgd.goal_generator import GoalGenerator

DEF_VAL_UNK = "?"  # Unknown
DEF_VAL_DNC = "don't care"  # Do not care
DEF_VAL_NUL = "none"  # for none
DEF_VAL_BOOKED = "yes"  # for booked
DEF_VAL_NOBOOK = "no"  # for booked
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, DEF_VAL_NOBOOK]
BOOK_SLOT = []


class UserPolicyAgendaSGD:
    def __init__(
        self,
        max_turn=20,
        domains=None,
        seeder: Dict[str, Any] = {},
        lower_case: bool = False,
    ):
        # OK
        """
        Constructor for User_Policy_Agenda class.
        """
        self.max_turn = max_turn * 2
        self.max_initiative = 4

        self.goal_generator = GoalGenerator(domains=domains, seeder=seeder)

        self.__turn = 0
        self.goal = None
        self.agenda = None
        self.seeder = seeder
        self.lower_case = lower_case

    def get_domains(self) -> List[str]:
        return self.goal_generator.get_domains()

    def reset_turn(self):
        self.__turn = 0

    def init_session(self, ini_goal=None):
        """Build new Goal and Agenda for next session"""
        self.reset_turn()
        if not ini_goal:
            self.goal = Goal(self.goal_generator)
        else:
            self.goal = ini_goal
        self.domain_goals = self.goal.domain_goals
        self.agenda = Agenda(self.goal, seeder=self.seeder)
        self.complete_domain = []
        self.booking = []
        self.complete = False
        self.last_stack = copy.deepcopy(self.agenda.get_stack())

    def get_in_reward(self, domain):
        new = -1.0
        self.complete = False
        # if len(self.booking) > 0 and domain not in self.booking:

        if (
            domain.lower() in self.domain_goals
            and domain not in self.complete_domain
        ):
            if "reqt" in self.domain_goals[domain.lower()]:
                reqt_vals = self.domain_goals[domain.lower()]["reqt"].values()
                xflg = True
                for val in reqt_vals:
                    if val in NOT_SURE_VALS:
                        # self.complete = False
                        xflg = False
                        break
                if xflg:
                    self.complete = True
                    new = 40.0
                    self.complete_domain.append(domain)
                    if "booked" in self.domain_goals[domain.lower()]:
                        self.booking.append(domain)
            else:
                assert "book" in self.domain_goals[domain.lower()]
                cur_stack = self.agenda.get_stack()
                # if len(self.last_stack) != len(cur_stack):
                #  self.complete = False

                if len(self.last_stack) == len(cur_stack):
                    # assert self.complete is True
                    self.complete = True
                    new = 40.0
                    self.complete_domain.append(domain)
                    if (
                        "booked" in self.domain_goals[domain.lower()]
                        and self.domain_goals[domain.lower()]["booked"]
                        in NOT_SURE_VALS
                    ):  # yes, yes todo
                        self.booking.append(domain)

        elif domain.lower() in ["booking"] and len(self.booking) > 0:
            for dm in self.booking:
                assert "booked" in self.domain_goals[dm.lower()]
                assert dm in self.complete_domain

                if (
                    self.domain_goals[dm.lower()]["booked"]
                    not in NOT_SURE_VALS
                ):  # 2: ? yes 3: ? ?
                    new = 40.0
                    self.complete = True
                    self.booking.remove(dm)

        self.last_stack = copy.deepcopy(self.agenda.get_stack())
        return new, self.complete

    def predict(
        self,
        sys_dialog_act: Union[
            List[Tuple[str, str, str, str]], Dict[str, List[Tuple[str, str]]]
        ],
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Predict an user act based on state and preorder system action.
        Args:
            sys_dialog_act (list): [[i, d, s, v], ...]. OR
                {d-i: [[s,v], ...]}.


        Returns:
            usr_dialog_act: {d-i: [[s,v], ...]}.

        """
        self.__turn += 2

        # compatable with Convlab1
        if isinstance(sys_dialog_act, dict):
            tuples = []
            for domain_intent, svs in sys_dialog_act.items():
                for slot, value in svs:
                    domain, intent = domain_intent.split("-")
                    tuples.append([intent, domain, str(slot), str(value)])
            sys_dialog_act = tuples
        assert isinstance(sys_dialog_act, list), sys_dialog_act

        sys_action = {}
        for intent, domain, slot, value in sys_dialog_act:
            intent, domain, slot, value = (
                intent.lower(),
                domain.lower(),
                slot.lower(),
                value.lower(),
            )
            if slot == "choice" and value.strip().lower() in ["0", "zero"]:
                nooffer_key = "-".join([domain, "nooffer"])
                sys_action.setdefault(nooffer_key, [])
                sys_action[nooffer_key].append(["none", "none"])
            else:
                k = "-".join([domain, intent])
                sys_action.setdefault(k, [])
                sys_action[k].append([slot, value])

        if self.__turn > self.max_turn:
            self.agenda.close_session()
        else:
            sys_action = self._transform_sysact_in(sys_action)
            self.agenda.update(sys_action, self.goal)
            if self.goal.task_complete():
                self.agenda.close_session()

        # A -> A' + user_action
        # action = self.agenda.get_action(self.seeder.get('py', random).randint(2, self.max_initiative))
        action = self.agenda.get_action(self.max_initiative)

        # transform to DA
        action = self._transform_usract_out(action)
        if self.lower_case:
            action = json.loads(json.dumps(action).lower())
        return action

    def is_terminated(self):
        # Is there any action to say?
        return self.agenda.is_empty()

    def is_success(self):
        if not self.goal.task_complete():
            return False
        for domain in self.domain_goals:
            domain_goal = self.domain_goals[domain]
            constraints = dict(
                domain_goal["info"], **domain_goal["reqt"]
            ).items()
            constraints = [
                (slot, value)
                for slot, value in constraints
                if value != "not available"
            ]
            if (
                len(
                    self.goal_generator.db.query(
                        domain=domain, constraints=constraints
                    )
                )
                == 0
            ):
                return False
        return True

    def get_goal(self):
        return self.domain_goals

    def get_reward(self):
        return self._reward()

    def _reward(self):
        """
        Calculate reward based on task completion
        Returns:
            reward (float): Reward given by user.
        """
        if self.goal.task_complete():
            reward = 2.0 * self.max_turn
        elif self.agenda.is_empty():
            reward = -1.0 * self.max_turn
        else:
            reward = -1.0
        return reward

    @classmethod
    def _transform_usract_out(cls, action):
        # print('before transform', action)
        new_action = {}
        for act in action.keys():
            if "-" in act:
                if "general" not in act:
                    (dom, intent) = act.split("-")
                    new_act = dom.capitalize() + "-" + intent.capitalize()
                    new_action[new_act] = []
                    for pairs in action[act]:
                        # slot = REF_USR_DA_M[dom.capitalize()].get(
                        #     pairs[0], None
                        # )
                        slot = pairs[0]
                        if pairs[0] == "none" and pairs[1] == "none":
                            new_action[new_act].append(["none", "none"])
                        elif pairs[0] == "choice" and pairs[1] == "any":
                            new_action[new_act].append(["Choice", "any"])
                        elif pairs[0] == "NotBook" and pairs[1] == "none":
                            new_action[new_act].append(["NotBook", "none"])
                        elif slot is not None:
                            new_action[new_act].append([slot, pairs[1]])
                    # new_action[new_act] = [[REF_USR_DA_M[dom.capitalize()].get(pairs[0], pairs[0]), pairs[1]] for pairs in action[act]]
                else:
                    new_action[act] = action[act]
            else:
                pass
        # print('after transform', new_action)
        return new_action

    @classmethod
    def _transform_sysact_in(cls, action):
        # print("sys in", action)
        new_action = {}
        if not isinstance(action, dict):
            # logging.warning('illegal da: {}'.format(action))
            return new_action

        for act in action.keys():
            if not isinstance(act, str) or "-" not in act:
                # logging.warning('illegal act: {}'.format(act))
                continue

            if "general" not in act:
                (dom, intent) = act.lower().split("-")
                new_list = []
                for pairs in action[act]:
                    if (
                        (
                            not isinstance(pairs, list)
                            and not isinstance(pairs, tuple)
                        )
                        or (len(pairs) < 2)
                        or (
                            not isinstance(pairs[0], str)
                            or (
                                not isinstance(pairs[1], str)
                                and not isinstance(pairs[1], int)
                            )
                        )
                    ):
                        # logging.warning('illegal pairs: {}'.format(pairs))
                        continue

                    new_list.append(
                        [
                            pairs[0],
                            cls._normalize_value(
                                dom,
                                intent,
                                pairs[0],
                                pairs[1],
                            ),
                        ]
                    )

                    if len(new_list) > 0:
                        new_action[act.lower()] = new_list
            else:
                new_action[act.lower()] = action[act]
        # print("sys in transformed", new_action)
        return new_action

    @classmethod
    def _normalize_value(cls, domain, intent, slot, value):
        if intent == "request":
            return DEF_VAL_UNK
        # ignore the rest for now
        return value

        if domain not in cls.stand_value_dict.keys():
            return value

        if slot not in cls.stand_value_dict[domain]:
            return value

        if slot in ["parking", "internet"] and value == "none":
            return "yes"

        value_list = cls.stand_value_dict[domain][slot]
        low_value_list = [item.lower() for item in value_list]
        value_list = sorted(list(set(value_list) | set(low_value_list)))
        if value not in value_list:
            normalized_v = simple_fuzzy_match(value_list, value)
            if normalized_v is not None:
                return normalized_v
            # try some transformations
            cand_values = transform_value(value)
            for cv in cand_values:
                _nv = simple_fuzzy_match(value_list, cv)
                if _nv is not None:
                    return _nv
            if check_if_time(value):
                return value

            # logging.debug('Value not found in standard value set: [%s] (slot: %s domain: %s)' % (value, slot, domain))
        return value


def transform_value(value):
    cand_list = []
    # a 's -> a's
    if " 's" in value:
        cand_list.append(value.replace(" 's", "'s"))
    # a - b -> a-b
    if " - " in value:
        cand_list.append(value.replace(" - ", "-"))
    return cand_list


def simple_fuzzy_match(value_list, value):
    # check contain relation
    v0 = " ".join(value.split())
    v0N = "".join(value.split())
    for val in value_list:
        v1 = " ".join(val.split())
        if v0 in v1 or v1 in v0 or v0N in v1 or v1 in v0N:
            return v1
    value = value.lower()
    v0 = " ".join(value.split())
    v0N = "".join(value.split())
    for val in value_list:
        v1 = " ".join(val.split())
        if v0 in v1 or v1 in v0 or v0N in v1 or v1 in v0N:
            return v1
    return None


def check_if_time(value):
    value = value.strip()
    match = re.search(r"(\d{1,2}:\d{1,2})", value)
    if match is None:
        return False
    groups = match.groups()
    if len(groups) <= 0:
        return False
    return True


def check_constraint(slot, val_usr, val_sys):
    try:
        if slot == "arriveBy":
            val1 = int(val_usr.split(":")[0]) * 100 + int(
                val_usr.split(":")[1]
            )
            val2 = int(val_sys.split(":")[0]) * 100 + int(
                val_sys.split(":")[1]
            )
            if val1 < val2:
                return True
        elif slot == "leaveAt":
            val1 = int(val_usr.split(":")[0]) * 100 + int(
                val_usr.split(":")[1]
            )
            val2 = int(val_sys.split(":")[0]) * 100 + int(
                val_sys.split(":")[1]
            )
            if val1 > val2:
                return True
        else:
            if val_usr != val_sys:
                return True
        return False
    except:
        return False


class Goal(object):
    """User Goal Model Class."""

    def __init__(self, goal_generator: GoalGenerator):
        """
        create new Goal by random
        Args:
            goal_generator (GoalGenerator): Goal Generator.
        """
        self.domain_goals = goal_generator.get_user_goal()

        self.domains = list(self.domain_goals["domain_ordering"])
        del self.domain_goals["domain_ordering"]

        for domain in self.domains:
            if "reqt" in self.domain_goals[domain].keys():
                self.domain_goals[domain]["reqt"] = {
                    slot: DEF_VAL_UNK
                    for slot in self.domain_goals[domain]["reqt"]
                }

            if "book" in self.domain_goals[domain].keys():
                self.domain_goals[domain]["booked"] = DEF_VAL_UNK

    def set_user_goal(self, user_goal):
        """
        set new Goal given user goal generated by goal_generator.get_user_goal()
        Args:
            user_goal : user goal generated by GoalGenerator.
        """
        self.domain_goals = user_goal

        self.domains = list(self.domain_goals["domain_ordering"])
        del self.domain_goals["domain_ordering"]

        for domain in self.domains:
            if "reqt" in self.domain_goals[domain].keys():
                self.domain_goals[domain]["reqt"] = {
                    slot: DEF_VAL_UNK
                    for slot in self.domain_goals[domain]["reqt"]
                }

            if "book" in self.domain_goals[domain].keys():
                self.domain_goals[domain]["booked"] = DEF_VAL_UNK

    def task_complete(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        for domain in self.domains:
            if "reqt" in self.domain_goals[domain]:
                reqt_vals = self.domain_goals[domain]["reqt"].values()
                for val in reqt_vals:
                    if val in NOT_SURE_VALS:
                        return False

            if "booked" in self.domain_goals[domain]:
                if self.domain_goals[domain]["booked"] in NOT_SURE_VALS:
                    return False
        return True

    def next_domain_incomplete(self):
        # request
        for domain in self.domains:
            # reqt
            if "reqt" in self.domain_goals[domain]:
                requests = self.domain_goals[domain]["reqt"]
                unknow_reqts = [
                    key
                    for (key, val) in requests.items()
                    if val in NOT_SURE_VALS
                ]
                if len(unknow_reqts) > 0:
                    return (
                        domain,
                        "reqt",
                        ["name"] if "name" in unknow_reqts else unknow_reqts,
                    )

            # book
            if "booked" in self.domain_goals[domain]:
                if self.domain_goals[domain]["booked"] in NOT_SURE_VALS:
                    return (
                        domain,
                        "book",
                        self.domain_goals[domain]["fail_book"]
                        if "fail_book" in self.domain_goals[domain].keys()
                        else self.domain_goals[domain]["book"],
                    )

        return None, None, None

    def __str__(self):
        return (
            "-----Goal-----\n"
            + json.dumps(self.domain_goals, indent=4)
            + "\n-----Goal-----"
        )


class Agenda(object):
    def __init__(self, goal: Goal, seeder={}):
        """
        Build a new agenda from goal
        Args:
            goal (Goal): User goal.
        """

        def random_sample(data, minimum=0, maximum=1000, seeder={}):
            return seeder.get("py", random).sample(
                data,
                seeder.get("py", random).randint(
                    min(len(data), minimum), min(len(data), maximum)
                ),
            )

        self.seeder = seeder
        self.CLOSE_ACT = "general-bye"
        self.HELLO_ACT = "general-greet"
        self.__cur_push_num = 0

        self.__stack = []

        # there is a 'bye' action at the bottom of the stack
        self.__push(self.CLOSE_ACT)

        for idx in range(len(goal.domains) - 1, -1, -1):
            domain = goal.domains[idx]

            # inform
            # first ask fail_info which return no result then ask info
            if "fail_info" in goal.domain_goals[domain]:
                for slot in random_sample(
                    goal.domain_goals[domain]["fail_info"].keys(),
                    len(goal.domain_goals[domain]["fail_info"]),
                    seeder=self.seeder,
                ):
                    self.__push(
                        domain + "-inform",
                        slot,
                        goal.domain_goals[domain]["fail_info"][slot],
                    )
            elif "info" in goal.domain_goals[domain]:
                for slot in random_sample(
                    goal.domain_goals[domain]["info"].keys(),
                    len(goal.domain_goals[domain]["info"]),
                    seeder=self.seeder,
                ):
                    self.__push(
                        domain + "-inform",
                        slot,
                        goal.domain_goals[domain]["info"][slot],
                    )

            self.__push(domain + "-inform", "none", "none")

        self.cur_domain = None

    def get_stack(self):
        return self.__stack

    def update(self, sys_action, goal: Goal):
        """
        update Goal by current agent action and current goal. { A' + G" + sys_action -> A" }
        Args:
            sys_action (dict): Preorder system action.s
            goal (Goal): User Goal
        """
        self.__cur_push_num = 0
        self._update_current_domain(sys_action, goal)

        for diaact in sys_action.keys():
            slot_vals = sys_action[diaact]
            if "nooffer" in diaact:
                if self.update_domain(diaact, slot_vals, goal):
                    return
            elif "nobook" in diaact:
                if self.update_booking(diaact, slot_vals, goal):
                    return

        for diaact in sys_action.keys():
            if "nooffer" in diaact or "nobook" in diaact:
                continue

            slot_vals = sys_action[diaact]
            if "booking" in diaact:
                if self.update_booking(diaact, slot_vals, goal):
                    return
            elif "general" in diaact:
                if self.update_general(diaact, slot_vals, goal):
                    return
            else:
                if self.update_domain(diaact, slot_vals, goal):
                    return

        for diaact in sys_action.keys():
            if "inform" in diaact or "recommend" in diaact:
                for slot, val in sys_action[diaact]:
                    if slot == "name":
                        self._remove_item(
                            diaact.split("-")[0] + "-inform", "choice"
                        )
            if "booking" in diaact and self.cur_domain:
                g_book = self._get_goal_infos(self.cur_domain, goal)[-2]
                if len(g_book) == 0:
                    self._push_item(
                        self.cur_domain + "-inform", "NotBook", "none"
                    )
            if "OfferBook" in diaact:
                domain = diaact.split("-")[0]
                g_book = self._get_goal_infos(domain, goal)[-2]
                if len(g_book) == 0:
                    self._push_item(domain + "-inform", "NotBook", "none")

        self.post_process(goal)

    def post_process(self, goal: Goal):
        unk_dom, unk_type, data = goal.next_domain_incomplete()
        if unk_dom is not None:
            if (
                unk_type == "reqt"
                and not self._check_reqt_info(unk_dom)
                and not self._check_reqt(unk_dom)
            ):
                for slot in data:
                    self._push_item(unk_dom + "-request", slot, DEF_VAL_UNK)
            elif (
                unk_type == "book"
                and not self._check_reqt_info(unk_dom)
                and not self._check_book_info(unk_dom)
            ):
                for (slot, val) in data.items():
                    self._push_item(unk_dom + "-inform", slot, val)

    def update_booking(self, diaact, slot_vals, goal: Goal):
        """
        Handel Book-XXX
        :param diaact:      Dial-Act
        :param slot_vals:   slot value pairs
        :param goal:        Goal
        :return:            True:user want to close the session. False:session is continue
        """
        _, intent = diaact.split("-")
        domain = self.cur_domain

        isover = False
        if domain not in goal.domains:
            isover = False

        elif intent in ["book", "inform"]:
            isover = self._handle_inform(domain, intent, slot_vals, goal)

        elif intent in ["nobook"]:
            isover = self._handle_nobook(domain, intent, slot_vals, goal)

        elif intent in ["request"]:
            isover = self._handle_request(domain, intent, slot_vals, goal)

        return isover

    def update_domain(self, diaact, slot_vals, goal: Goal):
        """
        Handel Domain-XXX
        :param diaact:      Dial-Act
        :param slot_vals:   slot value pairs
        :param goal:        Goal
        :return:            True:user want to close the session. False:session is continue
        """
        domain, intent = diaact.split("-")

        isover = False
        if domain not in goal.domains:
            isover = False

        elif intent.lower() in [
            "inform",
            "recommend",
            "offerbook",
            "offerbooked",
            "offer",
        ]:
            isover = self._handle_inform(domain, intent, slot_vals, goal)

        elif intent.lower() in ["request"]:
            isover = self._handle_request(domain, intent, slot_vals, goal)

        elif intent.lower() in ["nooffer", 'notifyfailure']:
            isover = self._handle_nooffer(domain, intent, slot_vals, goal)

        elif intent.lower() in ["select"]:
            isover = self._handle_select(domain, intent, slot_vals, goal)

        return isover

    def update_general(self, diaact, slot_vals, goal: Goal):
        domain, intent = diaact.split("-")

        if intent == "bye":
            # self.close_session()
            # return True
            pass
        elif intent == "greet":
            pass
        elif intent == "reqmore":
            pass
        elif intent == "welcome":
            pass

        return False

    def close_session(self):
        """Clear up all actions"""
        self.__stack = []
        self.__cur_push_num = 0
        self.__push(self.CLOSE_ACT)

    def get_action(self, initiative=1):
        """
        get multiple acts based on initiative
        Args:
            initiative (int): number of slots , just for 'inform'
        Returns:
            action (dict): user diaact
        """
        # print(self)
        diaacts, slots, values = self.__pop(initiative)
        action = {}
        for (diaact, slot, value) in zip(diaacts, slots, values):
            if diaact not in action.keys():
                action[diaact] = []
            action[diaact].append([slot, value])

        self._setdefault_current_domain_by_usraction(action)

        return action

    def is_empty(self):
        """
        Is the agenda already empty
        Returns:
            (boolean): True for empty, False for not.
        """
        return len(self.__stack) <= 0

    @staticmethod
    def _my_value(value):
        new_value = value
        if value in NOT_SURE_VALS:
            new_value = '"' + value + '"'
        return new_value

    def _get_goal_infos(self, domain, goal: Goal):
        g_reqt = goal.domain_goals[domain].get("reqt", dict({}))
        g_info = goal.domain_goals[domain].get("info", dict({}))
        g_fail_info = goal.domain_goals[domain].get("fail_info", dict({}))
        g_book = goal.domain_goals[domain].get("book", dict({}))
        g_fail_book = goal.domain_goals[domain].get("fail_book", dict({}))
        return g_reqt, g_info, g_fail_info, g_book, g_fail_book

    def _handle_inform(self, domain, intent, slot_vals, goal: Goal):
        (
            g_reqt,
            g_info,
            g_fail_info,
            g_book,
            g_fail_book,
        ) = self._get_goal_infos(domain, goal)

        info_right = True
        for [slot, value] in slot_vals:
            # For multiple choices, add new intent to select one:
            if (
                intent.lower() == "informcount"
                and value.strip().lower()
                not in [
                    "0",
                    "zero",
                ]
            ):
                self._push_item(domain + "-inform", "choice", "any")

            if slot in g_reqt:
                if not self._check_reqt_info(domain):
                    self._remove_item(domain + "-request", slot)
                    # g_reqt[slot] = self._my_value(value)
                    g_reqt[slot] = value

            elif slot in g_fail_info and value != g_fail_info[slot]:
                self._push_item(domain + "-inform", slot, g_fail_info[slot])
                info_right = False

            elif (
                not g_fail_info
                and slot in g_info
                and check_constraint(slot, g_info[slot], value)
            ):
                self._push_item(domain + "-inform", slot, g_info[slot])
                info_right = False

            elif slot in g_fail_book and value != g_fail_book[slot]:
                self._push_item(domain + "-inform", slot, g_fail_book[slot])
                info_right = False

            elif not g_fail_book and slot in g_book and value != g_book[slot]:
                self._push_item(domain + "-inform", slot, g_book[slot])
                info_right = False

        if intent in ["book", "offerbooked"] and info_right:
            # booked ok
            if "booked" in goal.domain_goals[domain]:
                goal.domain_goals[domain]["booked"] = DEF_VAL_BOOKED
            # self._push_item('general-thank')

        return False

    def _handle_request(self, domain, intent, slot_vals, goal: Goal):
        (
            g_reqt,
            g_info,
            g_fail_info,
            g_book,
            g_fail_book,
        ) = self._get_goal_infos(domain, goal)
        for [slot, _] in slot_vals:
            if slot == "time":
                if domain in ["train", "restaurant"]:
                    slot = "duration" if domain == "train" else "time"
                else:
                    # logging.warning('illegal booking slot: %s, slot: %s domain' % (slot, domain))
                    continue

            if slot in g_reqt:
                pass
            elif slot in g_fail_info:
                self._push_item(domain + "-inform", slot, g_fail_info[slot])
            elif not g_fail_info and slot in g_info:
                self._push_item(domain + "-inform", slot, g_info[slot])

            elif slot in g_fail_book:
                self._push_item(domain + "-inform", slot, g_fail_book[slot])
            elif not g_fail_book and slot in g_book:
                self._push_item(domain + "-inform", slot, g_book[slot])

            else:

                if domain == "taxi" and (
                    slot == "destination" or slot == "departure"
                ):
                    places = [
                        dom
                        for dom in goal.domains[: goal.domains.index("taxi")]
                        if dom
                        in [
                            "attraction",
                            "hotel",
                            "restaurant",
                            "police",
                            "hospital",
                        ]
                    ]  # name will not appear in reqt
                    if len(places) >= 1 and slot == "destination":
                        place_idx = -1
                    elif len(places) >= 2 and slot == "departure":
                        place_idx = -2
                    else:
                        place_idx = None
                    if place_idx:
                        if (
                            goal.domain_goals[places[place_idx]]["info"].get(
                                "name", DEF_VAL_NUL
                            )
                            not in NOT_SURE_VALS
                        ):
                            place = goal.domain_goals[places[place_idx]][
                                "info"
                            ]["name"]
                        # elif goal.domain_goals[places[place_idx]]['reqt'].get('address', DEF_VAL_NUL) not in NOT_SURE_VALS:
                        #     place = goal.domain_goals[places[place_idx]]['reqt']['address']
                        else:
                            place = "the " + places[place_idx]
                        self._push_item(domain + "-inform", slot, place)
                    else:
                        self._push_item(domain + "-inform", slot, DEF_VAL_DNC)

                else:
                    # for those sys requests that are not in user goal
                    self._push_item(domain + "-inform", slot, DEF_VAL_DNC)

        return False

    def _handle_nooffer(self, domain, intent, slot_vals, goal: Goal):
        (
            g_reqt,
            g_info,
            g_fail_info,
            g_book,
            g_fail_book,
        ) = self._get_goal_infos(domain, goal)
        if g_fail_info:
            # update info data to the stack
            for slot in g_info.keys():
                if (slot not in g_fail_info) or (
                    slot in g_fail_info and g_fail_info[slot] != g_info[slot]
                ):
                    self._push_item(domain + "-inform", slot, g_info[slot])

            # change fail_info name
            goal.domain_goals[domain]["fail_info_fail"] = goal.domain_goals[
                domain
            ].pop("fail_info")
        elif g_reqt:
            self.close_session()
            return True
        return False

    def _handle_nobook(self, domain, intent, slot_vals, goal: Goal):
        (
            g_reqt,
            g_info,
            g_fail_info,
            g_book,
            g_fail_book,
        ) = self._get_goal_infos(domain, goal)
        if g_fail_book:
            # Discard fail_book data and update the book data to the stack
            for slot in g_book.keys():
                if (slot not in g_fail_book) or (
                    slot in g_fail_book and g_fail_book[slot] != g_book[slot]
                ):
                    self._push_item(domain + "-inform", slot, g_book[slot])

            # change fail_info name
            goal.domain_goals[domain]["fail_book_fail"] = goal.domain_goals[
                domain
            ].pop("fail_book")
        elif "booked" in goal.domain_goals[domain].keys():
            self.close_session()
            return True
        return False

    def _handle_select(self, domain, intent, slot_vals, goal: Goal):
        (
            g_reqt,
            g_info,
            g_fail_info,
            g_book,
            g_fail_book,
        ) = self._get_goal_infos(domain, goal)
        # delete Choice
        for slot, val in slot_vals:
            if slot == "choice":
                self._push_item(domain + "-inform", "choice", "any")
        slot_vals = [
            [slot, val] for [slot, val] in slot_vals if slot != "choice"
        ]

        if slot_vals:
            slot = slot_vals[0][0]

            if slot in g_fail_info:
                self._push_item(domain + "-inform", slot, g_fail_info[slot])
            elif not g_fail_info and slot in g_info:
                self._push_item(domain + "-inform", slot, g_info[slot])

            elif slot in g_fail_book:
                self._push_item(domain + "-inform", slot, g_fail_book[slot])
            elif not g_fail_book and slot in g_book:
                self._push_item(domain + "-inform", slot, g_book[slot])

            else:
                if not self._check_reqt_info(domain):
                    [slot, value] = self.seeder.get("py", random).choice(
                        slot_vals
                    )
                    self._push_item(domain + "-inform", slot, value)

                    if slot in g_reqt:
                        self._remove_item(domain + "-request", slot)
                        g_reqt[slot] = value
        return False

    def _update_current_domain(self, sys_action, goal: Goal):
        for diaact in sys_action.keys():
            domain, _ = diaact.split("-")
            if domain in goal.domains:
                self.cur_domain = domain

    def _setdefault_current_domain_by_usraction(self, usr_action):
        if self.cur_domain is None:
            for diaact in usr_action.keys():
                domain, _ = diaact.split("-")
                if domain in [
                    "attraction",
                    "hotel",
                    "restaurant",
                    "taxi",
                    "train",
                ]:
                    self.cur_domain = domain

    def _remove_item(self, diaact, slot=DEF_VAL_UNK):
        for idx in range(len(self.__stack)):
            if "general" in diaact:
                if self.__stack[idx]["diaact"] == diaact:
                    self.__stack.remove(self.__stack[idx])
                    break
            else:
                if (
                    self.__stack[idx]["diaact"] == diaact
                    and self.__stack[idx]["slot"] == slot
                ):
                    self.__stack.remove(self.__stack[idx])
                    break

    def _push_item(self, diaact, slot=DEF_VAL_NUL, value=DEF_VAL_NUL):
        self._remove_item(diaact, slot)
        self.__push(diaact, slot, value)
        self.__cur_push_num += 1

    def _check_item(self, diaact, slot=None):
        for idx in range(len(self.__stack)):
            if slot is None:
                if self.__stack[idx]["diaact"] == diaact:
                    return True
            else:
                if (
                    self.__stack[idx]["diaact"] == diaact
                    and self.__stack[idx]["slot"] == slot
                ):
                    return True
        return False

    def _check_reqt(self, domain):
        for idx in range(len(self.__stack)):
            if self.__stack[idx]["diaact"] == domain + "-request":
                return True
        return False

    def _check_reqt_info(self, domain):
        for idx in range(len(self.__stack)):
            if (
                self.__stack[idx]["diaact"] == domain + "-inform"
                and self.__stack[idx]["slot"] not in BOOK_SLOT
            ):
                return True
        return False

    def _check_book_info(self, domain):
        for idx in range(len(self.__stack)):
            if (
                self.__stack[idx]["diaact"] == domain + "-inform"
                and self.__stack[idx]["slot"] in BOOK_SLOT
            ):
                return True
        return False

    def __check_next_diaact_slot(self):
        if len(self.__stack) > 0:
            return self.__stack[-1]["diaact"], self.__stack[-1]["slot"]
        return None, None

    def __check_next_diaact(self):
        if len(self.__stack) > 0:
            return self.__stack[-1]["diaact"]
        return None

    def __push(self, diaact, slot=DEF_VAL_NUL, value=DEF_VAL_NUL):
        if slot in ["people", "day", "area", "pricerange"]:
            for item in self.__stack:
                if (
                    item["slot"] == slot
                    and item["value"] == value
                    and self.seeder.get("py", random).random() < 0.3
                ):
                    if slot == "people":
                        item["value"] = "the same"
                    elif slot == "day":
                        item["value"] = "the same day"
                    elif slot == "pricerange":
                        item[
                            "value"
                        ] = "in the same price range as the {}".format(
                            diaact.split("-")[0]
                        )
                    elif slot == "area":
                        item["value"] = "same area as the {}".format(
                            diaact.split("-")[0]
                        )
        self.__stack.append({"diaact": diaact, "slot": slot, "value": value})

    def __pop(self, initiative=1):
        diaacts = []
        slots = []
        values = []

        p_diaact, p_slot = self.__check_next_diaact_slot()
        if p_diaact.split("-")[1] == "inform" and p_slot in BOOK_SLOT:
            for _ in range(
                10 if self.__cur_push_num == 0 else self.__cur_push_num
            ):
                try:
                    item = self.__stack.pop(-1)
                    diaacts.append(item["diaact"])
                    slots.append(item["slot"])
                    values.append(item["value"])

                    cur_diaact = item["diaact"]

                    next_diaact, next_slot = self.__check_next_diaact_slot()
                    if (
                        next_diaact is None
                        or next_diaact != cur_diaact
                        or next_diaact.split("-")[1] != "inform"
                        or next_slot not in BOOK_SLOT
                    ):
                        break
                except Exception as e:
                    break
        else:
            if self.__cur_push_num == 0 or (
                all(
                    [
                        self.__stack[-i - 1]["value"] == DEF_VAL_DNC
                        for i in range(
                            0, min(len(self.__stack), self.__cur_push_num)
                        )
                    ]
                )
            ):
                # pop more when only dontcare
                num2pop = initiative
            else:
                num2pop = self.__cur_push_num
            for _ in range(num2pop):
                try:
                    item = self.__stack.pop(-1)
                    diaacts.append(item["diaact"])
                    slots.append(item["slot"])
                    values.append(item["value"])

                    cur_diaact = item["diaact"]

                    next_diaact = self.__check_next_diaact()
                    if (
                        next_diaact is None
                        or next_diaact != cur_diaact
                        or (
                            cur_diaact.split("-")[1] == "request"
                            and item["slot"] == "name"
                        )
                    ):
                        break
                except Exception as e:
                    break

        return diaacts, slots, values

    def __str__(self):
        text = "\n-----agenda-----\n"
        text += "<stack top>\n"
        for item in reversed(self.__stack):
            text += str(item) + "\n"
        text += "<stack btm>\n"
        text += "-----agenda-----\n"
        return text
