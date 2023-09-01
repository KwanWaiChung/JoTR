"""
"""

import json, re, copy
import os
import pickle
import random
from collections import Counter
from copy import deepcopy
from pprint import pprint
from typing import Dict, Any, List
from convlab.data.sgd.dbquery import Database
from convlab.data.sgd.info import NUL_VALUE
import numpy as np


domains = {
    "attraction",
    "hotel",
    "restaurant",
    "train",
    "taxi",
    "hospital",
    "police",
}
days = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
domain_keywords = {
    "restaurant": "place to dine",
    "train": "train",
    "hotel": "place to stay",
    "attraction": "places to go",
    "police": "help",
    "taxi": "taxi",
    "hospital": "hospital",
}
request_slot_string_map = {
    "phone": "phone number",
    "pricerange": "price range",
    "duration": "travel time",
    "arriveBy": "arrival time",
    "leaveAt": "departure time",
    "trainID": "train ID",
}
templates = {
    "intro": "You are looking for information in Cambridge.",
    "restaurant": {
        "intro": "You are looking forward to trying local restaurants.",
        "request": "Once you find a restaurant, make sure you get {}.",
        "area": "The restaurant should be in the {}.",
        "food": "The restaurant should serve {} food.",
        "name": (
            "You are looking for a particular restaurant. Its name is"
            " called {}."
        ),
        "pricerange": "The restaurant should be in the {} price range.",
        "book": "Once you find the restaurant you want to book a table {}.",
        "fail_info food": (
            "If there is no such restaurant, how about one that serves {}"
            " food."
        ),
        "fail_info area": (
            "If there is no such restaurant, how about one in the {} area."
        ),
        "fail_info pricerange": (
            "If there is no such restaurant, how about one in the {} price"
            " range."
        ),
        "fail_book time": "If the booking fails how about {}.",
        "fail_book day": "If the booking fails how about {}.",
    },
    "hotel": {
        "intro": "You are looking for a place to stay.",
        "request": "Once you find a hotel, make sure you get {}.",
        "stars": "The hotel should have a star of {}.",
        "area": "The hotel should be in the {}.",
        "type": "The hotel should be in the type of {}.",
        "pricerange": "The hotel should be in the {} price range.",
        "name": (
            "You are looking for a particular hotel. Its name is called {}."
        ),
        "internet yes": "The hotel should include free wifi.",
        "internet no": "The hotel does not need to include free wifi.",
        "parking yes": "The hotel should include free parking.",
        "parking no": "The hotel does not need to include free parking.",
        "book": "Once you find the hotel you want to book it {}.",
        "fail_info type": (
            "If there is no such hotel, how about one that is in the type"
            " of {}."
        ),
        "fail_info area": (
            "If there is no such hotel, how about one that is in the {} area."
        ),
        "fail_info stars": (
            "If there is no such hotel, how about one that has a star of {}."
        ),
        "fail_info pricerange": (
            "If there is no such hotel, how about one that is in the {} price"
            " range."
        ),
        "fail_info parking yes": (
            "If there is no such hotel, how about one that has free parking."
        ),
        "fail_info parking no": (
            "If there is no such hotel, how about one that does not has free"
            " parking."
        ),
        "fail_info internet yes": (
            "If there is no such hotel, how about one that has free wifi."
        ),
        "fail_info internet no": (
            "If there is no such hotel, how about one that does not has free"
            " wifi."
        ),
        "fail_book stay": "If the booking fails how about {} nights.",
        "fail_book day": "If the booking fails how about {}.",
    },
    "attraction": {
        "intro": "You are excited about seeing local tourist attractions.",
        "request": "Once you find an attraction, make sure you get {}.",
        "area": "The attraction should be in the {}.",
        "type": "The attraction should be in the type of {}.",
        "name": (
            "You are looking for a particular attraction. Its name is"
            " called {}."
        ),
        "fail_info type": (
            "If there is no such attraction, how about one that is in the type"
            " of {}."
        ),
        "fail_info area": (
            "If there is no such attraction, how about one in the {} area."
        ),
    },
    "taxi": {
        "intro": "You are also looking for a taxi.",
        "commute": (
            "You also want to book a taxi to commute between the two places."
        ),
        "restaurant": (
            "You want to make sure it arrives the restaurant by the booked"
            " time."
        ),
        "request": "Once you find a taxi, make sure you get {}.",
        "departure": "The taxi should depart from {}.",
        "destination": "The taxi should go to {}.",
        "leaveAt": "The taxi should leave after {}.",
        "arriveBy": "The taxi should arrive by {}.",
    },
    "train": {
        "intro": "You are also looking for a train.",
        "request": "Once you find a train, make sure you get {}.",
        "departure": "The train should depart from {}.",
        "destination": "The train should go to {}.",
        "day": "The train should leave on {}.",
        "leaveAt": "The train should leave after {}.",
        "arriveBy": "The train should arrive by {}.",
        "book": "Once you find the train you want to make a booking {}.",
    },
    "police": {
        "intro": "You were robbed and are looking for help.",
        "request": "Make sure you get {}.",
    },
    "hospital": {
        "intro": "You got injured and are looking for a hospital nearby",
        "request": "Make sure you get {}.",
        "department": "The hospital should have the {} department.",
    },
}

pro_correction = {
    # "info": 0.2,
    "info": 0.0,
    # "reqt": 0.2,
    "reqt": 0.0,
    # "book": 0.2
    "book": 0.0,
}


def null_boldify(content):
    return content


def do_boldify(content):
    return "<b>" + content + "</b>"


def nomial_sample(counter: Counter, seeder: Dict[str, Any] = {}):
    return list(counter.keys())[
        np.argmax(
            seeder.get("np", np.random).multinomial(1, list(counter.values()))
        )
    ]


class GoalGenerator:
    """User goal generator."""

    def __init__(
        self,
        goal_model_path="convlab/data/sgd/goal/goal_model.pkl",
        corpus_path=None,
        boldify=False,
        sample_info_from_trainset=True,
        sample_reqt_from_trainset=False,
        domains=None,
        seeder: Dict[str, Any] = {},
        verbose=True,
    ):
        """
        Args:
            goal_model_path: path to a goal model
            corpus_path: path to a dialog corpus to build a goal model
            boldify: highlight some information in the goal message
            sample_info_from_trainset: if True, sample info slots combination from train set, else sample each slot independently
            sample_reqt_from_trainset: if True, sample reqt slots combination from train set, else sample each slot independently
        """
        self.db = Database()
        self.goal_model_path = goal_model_path
        self.corpus_path = corpus_path
        self.seeder = seeder
        self.boldify = do_boldify if boldify else null_boldify
        self.sample_info_from_trainset = sample_info_from_trainset
        self.sample_reqt_from_trainset = sample_reqt_from_trainset

        if verbose:
            print("=" * 100)
            print("=" * 100)
            print(self.goal_model_path)
            print("=" * 100)
            print("=" * 100)
        # self._build_goal_model()
        if os.path.exists(self.goal_model_path):
            (
                self.ind_slot_dist,
                self.ind_slot_value_dist,
                self.domain_ordering_dist,
                self.book_dist,
                self.slots_num_dist,
                self.slots_combination_dist,
            ) = pickle.load(open(self.goal_model_path, "rb"))
            if verbose:
                print("Loading goal model is done")
        else:
            _ = self._build_goal_model()
            if verbose:
                print("Building goal model is done")
        # remove some slot
        self.dbs_slot = self.get_dbs_slot(Database.db)

        if domains is not None:
            domains = [domain.lower() for domain in domains]
            if domains is not None:
                print("limiting domains to:", domains)
                self.domain_ordering_dist = {
                    k: v
                    for k, v in self.domain_ordering_dist.items()
                    if all([domain in domains for domain in k])
                }

    def get_domains(self) -> List[str]:
        return sorted(
            set(
                [
                    domain
                    for domains in self.domain_ordering_dist
                    for domain in domains
                ]
            )
        )

    def _build_goal_model(self):
        dialogs = json.load(open(self.corpus_path))
        # dialogs = json.loads(json.dumps(dialogs).lower())
        domains = []
        domain_keywords = {}
        for d in dialogs:
            for domain in dialogs[d]["goal"]:
                if domain == "message":
                    continue
                if domain not in domains:
                    domains.append(domain)
                    domain_keywords[domain] = domain
        domains = set(domains)
        # domain ordering
        def _get_dialog_domains(dialog):
            return list(
                filter(
                    lambda x: x in domains and len(dialog["goal"][x]) > 0,
                    dialog["goal"],
                )
            )

        domain_orderings = []
        for d in dialogs:
            d_domains = _get_dialog_domains(dialogs[d])
            first_index = []
            for domain in d_domains:
                message = (
                    [dialogs[d]["goal"]["message"]]
                    if type(dialogs[d]["goal"]["message"]) == str
                    else dialogs[d]["goal"]["message"]
                )
                for i, m in enumerate(message):
                    if (
                        domain_keywords[domain].lower() in m.lower()
                        or domain.lower() in m.lower()
                    ):
                        first_index.append(i)
                        break
            domain_orderings.append(
                tuple(
                    [
                        d
                        for d in map(
                            lambda x: x[1],
                            sorted(
                                zip(first_index, d_domains), key=lambda x: x[0]
                            ),
                        )
                    ]
                )
            )
        domain_ordering_cnt = Counter(domain_orderings)
        self.domain_ordering_dist = deepcopy(domain_ordering_cnt)
        for order in domain_ordering_cnt.keys():
            self.domain_ordering_dist[order] = domain_ordering_cnt[
                order
            ] / sum(domain_ordering_cnt.values())

        # independent goal slot distribution
        ind_slot_value_cnt = dict([(domain, {}) for domain in domains])
        domain_cnt = Counter()
        book_cnt = Counter()
        self.slots_combination_dist = {domain: {} for domain in domains}
        self.slots_num_dist = {domain: {} for domain in domains}

        for d in dialogs:
            iter_set = dialogs[d]["goal"]
            for domain in iter_set:
                if dialogs[d]["goal"][domain] != {}:
                    domain_cnt[domain] += 1
                if "info" in dialogs[d]["goal"][domain]:
                    if "info" not in self.slots_combination_dist[domain]:
                        self.slots_combination_dist[domain]["info"] = {}
                        self.slots_num_dist[domain]["info"] = {}

                    slots = [
                        s
                        for s in sorted(
                            list(dialogs[d]["goal"][domain]["info"].keys())
                        )
                    ]
                    self.slots_combination_dist[domain]["info"].setdefault(
                        tuple(slots), 0
                    )
                    self.slots_combination_dist[domain]["info"][
                        tuple(slots)
                    ] += 1
                    self.slots_num_dist[domain]["info"].setdefault(
                        len(slots), 0
                    )
                    self.slots_num_dist[domain]["info"][len(slots)] += 1

                    for slot in dialogs[d]["goal"][domain]["info"]:
                        if "invalid" in slot:
                            continue
                        if "info" not in ind_slot_value_cnt[domain]:
                            ind_slot_value_cnt[domain]["info"] = {}
                        if slot not in ind_slot_value_cnt[domain]["info"]:
                            ind_slot_value_cnt[domain]["info"][
                                slot
                            ] = Counter()
                        if "care" in dialogs[d]["goal"][domain]["info"][slot]:
                            continue
                        value = dialogs[d]["goal"][domain]["info"][
                            slot
                        ].lower()
                        ind_slot_value_cnt[domain]["info"][slot][value] += 1
                if "reqt" in dialogs[d]["goal"][domain]:
                    if "reqt" not in self.slots_combination_dist[domain]:
                        self.slots_combination_dist[domain]["reqt"] = {}
                        self.slots_num_dist[domain]["reqt"] = {}
                    slots = sorted(dialogs[d]["goal"][domain]["reqt"])
                    if (
                        domain in ["police", "hospital"]
                        and "postcode" in slots
                    ):
                        slots.remove("postcode")
                    else:
                        assert len(slots) > 0, print(
                            sorted(dialogs[d]["goal"][domain]["reqt"]), [slots]
                        )
                    if len(slots) > 0:
                        self.slots_combination_dist[domain]["reqt"].setdefault(
                            tuple(slots), 0
                        )
                        self.slots_combination_dist[domain]["reqt"][
                            tuple(slots)
                        ] += 1
                        self.slots_num_dist[domain]["reqt"].setdefault(
                            len(slots), 0
                        )
                        self.slots_num_dist[domain]["reqt"][len(slots)] += 1

                    for slot in dialogs[d]["goal"][domain]["reqt"]:
                        if "reqt" not in ind_slot_value_cnt[domain]:
                            ind_slot_value_cnt[domain]["reqt"] = Counter()
                        ind_slot_value_cnt[domain]["reqt"][slot] += 1
                if "book" in dialogs[d]["goal"][domain]:
                    book_cnt[domain] += 1
                    for slot in dialogs[d]["goal"][domain]["book"]:
                        if "invalid" in slot:
                            continue
                        if "book" not in ind_slot_value_cnt[domain]:
                            ind_slot_value_cnt[domain]["book"] = {}
                        if slot not in ind_slot_value_cnt[domain]["book"]:
                            ind_slot_value_cnt[domain]["book"][
                                slot
                            ] = Counter()
                        if "care" in dialogs[d]["goal"][domain]["book"][slot]:
                            continue
                        ind_slot_value_cnt[domain]["book"][slot][
                            dialogs[d]["goal"][domain]["book"][slot]
                        ] += 1

        # pprint(self.slots_num_dist)
        # pprint(self.slots_combination_dist)
        # for domain in domains:
        #     print(domain, len(self.slots_combination_dist[domain]['info']))
        self.ind_slot_value_dist = deepcopy(ind_slot_value_cnt)
        self.ind_slot_dist = dict([(domain, {}) for domain in domains])
        self.book_dist = {}
        for domain in domains:
            if "info" in ind_slot_value_cnt[domain]:
                for slot in ind_slot_value_cnt[domain]["info"]:
                    if "info" not in self.ind_slot_dist[domain]:
                        self.ind_slot_dist[domain]["info"] = {}
                    if slot not in self.ind_slot_dist[domain]["info"]:
                        self.ind_slot_dist[domain]["info"][slot] = {}
                    self.ind_slot_dist[domain]["info"][slot] = (
                        sum(ind_slot_value_cnt[domain]["info"][slot].values())
                        / domain_cnt[domain]
                    )
                    slot_total = sum(
                        ind_slot_value_cnt[domain]["info"][slot].values()
                    )
                    for val in self.ind_slot_value_dist[domain]["info"][slot]:
                        self.ind_slot_value_dist[domain]["info"][slot][val] = (
                            ind_slot_value_cnt[domain]["info"][slot][val]
                            / slot_total
                        )
            if "reqt" in ind_slot_value_cnt[domain]:
                for slot in ind_slot_value_cnt[domain]["reqt"]:
                    if "reqt" not in self.ind_slot_dist[domain]:
                        self.ind_slot_dist[domain]["reqt"] = {}
                    self.ind_slot_dist[domain]["reqt"][slot] = (
                        ind_slot_value_cnt[domain]["reqt"][slot]
                        / domain_cnt[domain]
                    )
                    self.ind_slot_value_dist[domain]["reqt"][slot] = (
                        ind_slot_value_cnt[domain]["reqt"][slot]
                        / domain_cnt[domain]
                    )
            if "book" in ind_slot_value_cnt[domain]:
                for slot in ind_slot_value_cnt[domain]["book"]:
                    if "book" not in self.ind_slot_dist[domain]:
                        self.ind_slot_dist[domain]["book"] = {}
                    if slot not in self.ind_slot_dist[domain]["book"]:
                        self.ind_slot_dist[domain]["book"][slot] = {}
                    self.ind_slot_dist[domain]["book"][slot] = (
                        sum(ind_slot_value_cnt[domain]["book"][slot].values())
                        / domain_cnt[domain]
                    )
                    slot_total = sum(
                        ind_slot_value_cnt[domain]["book"][slot].values()
                    )
                    for val in self.ind_slot_value_dist[domain]["book"][slot]:
                        self.ind_slot_value_dist[domain]["book"][slot][val] = (
                            ind_slot_value_cnt[domain]["book"][slot][val]
                            / slot_total
                        )
            self.book_dist[domain] = book_cnt[domain] / len(dialogs)

        pickle.dump(
            (
                self.ind_slot_dist,
                self.ind_slot_value_dist,
                self.domain_ordering_dist,
                self.book_dist,
                self.slots_num_dist,
                self.slots_combination_dist,
            ),
            open(self.goal_model_path, "wb"),
        )
        return domains

    def _get_domain_goal(self, domain):
        cnt_slot = self.ind_slot_dist[domain]
        cnt_slot_value = self.ind_slot_value_dist[domain]
        pro_book = self.book_dist[domain]

        while True:
            # domain_goal = defaultdict(lambda: {})
            # domain_goal = {'info': {}, 'fail_info': {}, 'reqt': {}, 'book': {}, 'fail_book': {}}
            domain_goal = {"info": {}}
            # inform
            if "info" in cnt_slot:
                if self.sample_info_from_trainset:
                    slots = self.seeder.get("py", random).choices(
                        list(
                            self.slots_combination_dist[domain]["info"].keys()
                        ),
                        list(
                            self.slots_combination_dist[domain][
                                "info"
                            ].values()
                        ),
                    )[0]
                    for slot in slots:
                        domain_goal["info"][slot] = nomial_sample(
                            cnt_slot_value["info"][slot], seeder=self.seeder
                        )
                else:
                    for slot in cnt_slot["info"]:
                        if (
                            self.seeder.get("py", random).random()
                            < cnt_slot["info"][slot] + pro_correction["info"]
                        ):
                            domain_goal["info"][slot] = nomial_sample(
                                cnt_slot_value["info"][slot],
                                seeder=self.seeder,
                            )

                if (
                    domain in ["hotel", "restaurant", "attraction"]
                    and "name" in domain_goal["info"]
                    and len(domain_goal["info"]) > 1
                ):
                    if (
                        self.seeder.get("py", random).random()
                        < cnt_slot["info"]["name"]
                    ):
                        domain_goal["info"] = {
                            "name": domain_goal["info"]["name"]
                        }
                    else:
                        del domain_goal["info"]["name"]

                if (
                    domain in ["taxi", "train"]
                    and "arriveBy" in domain_goal["info"]
                    and "leaveAt" in domain_goal["info"]
                ):
                    if self.seeder.get("py", random).random() < (
                        cnt_slot["info"]["leaveAt"](
                            cnt_slot["info"]["arriveBy"]
                            + cnt_slot["info"]["leaveAt"]
                        )
                    ):
                        del domain_goal["info"]["arriveBy"]
                    else:
                        del domain_goal["info"]["leaveAt"]

                if (
                    domain in ["taxi", "train"]
                    and "arriveBy" not in domain_goal["info"]
                    and "leaveAt" not in domain_goal["info"]
                ):
                    if self.seeder.get("py", random).random() < (
                        cnt_slot["info"]["arriveBy"](
                            cnt_slot["info"]["arriveBy"]
                            + cnt_slot["info"]["leaveAt"]
                        )
                    ):
                        domain_goal["info"]["arriveBy"] = nomial_sample(
                            cnt_slot_value["info"]["arriveBy"],
                            seeder=self.seeder,
                        )
                    else:
                        domain_goal["info"]["leaveAt"] = nomial_sample(
                            cnt_slot_value["info"]["leaveAt"],
                            seeder=self.seeder,
                        )

                # if domain in ["train"]:
                #     random_train = self.seeder.get("py", random).choice(
                #         self.train_database
                #     )
                #     domain_goal["info"]["departure"] = random_train[
                #         "departure"
                #     ]
                #     domain_goal["info"]["destination"] = random_train[
                #         "destination"
                #     ]

                if (
                    domain in ["taxi"]
                    and "departure" not in domain_goal["info"]
                ):
                    domain_goal["info"]["departure"] = nomial_sample(
                        cnt_slot_value["info"]["departure"], seeder=self.seeder
                    )

                if (
                    domain in ["taxi"]
                    and "destination" not in domain_goal["info"]
                ):
                    domain_goal["info"]["destination"] = nomial_sample(
                        cnt_slot_value["info"]["destination"],
                        seeder=self.seeder,
                    )

                if (
                    domain in ["taxi"]
                    and "departure" in domain_goal["info"]
                    and "destination" in domain_goal["info"]
                    and domain_goal["info"]["departure"]
                    == domain_goal["info"]["destination"]
                ):
                    if self.seeder.get("py", random).random() < (
                        cnt_slot["info"]["departure"](
                            cnt_slot["info"]["departure"]
                            + cnt_slot["info"]["destination"]
                        )
                    ):
                        domain_goal["info"]["departure"] = nomial_sample(
                            cnt_slot_value["info"]["departure"],
                            seeder=self.seeder,
                        )
                    else:
                        domain_goal["info"]["destination"] = nomial_sample(
                            cnt_slot_value["info"]["destination"],
                            seeder=self.seeder,
                        )
                if domain_goal["info"] == {}:
                    continue
            # request
            if "reqt" in cnt_slot:
                if self.sample_reqt_from_trainset:
                    not_in_info_slots = {}
                    for slots in self.slots_combination_dist[domain]["reqt"]:
                        for slot in slots:
                            if slot in domain_goal["info"]:
                                break
                        else:
                            not_in_info_slots[
                                slots
                            ] = self.slots_combination_dist[domain]["reqt"][
                                slots
                            ]
                    pprint(not_in_info_slots)
                    reqt = list(
                        self.seeder.get("py", random).choices(
                            list(not_in_info_slots.keys()),
                            list(not_in_info_slots.values()),
                        )[0]
                    )
                else:
                    reqt = [
                        slot
                        for slot in cnt_slot["reqt"]
                        if self.seeder.get("py", random).random()
                        < cnt_slot["reqt"][slot] + pro_correction["reqt"]
                        and slot not in domain_goal["info"]
                    ]
                if len(reqt) > 0:
                    domain_goal["reqt"] = reqt

            # book (ignore booking for now)
            # if (
            #     "book" in cnt_slot
            #     and self.seeder.get("py", random).random()
            #     < pro_book + pro_correction["book"]
            # ):
            #     if "book" not in domain_goal:
            #         domain_goal["book"] = {}

            #     for slot in cnt_slot["book"]:
            #         if (
            #             self.seeder.get("py", random).random()
            #             < cnt_slot["book"][slot] + pro_correction["book"]
            #         ):
            #             domain_goal["book"][slot] = nomial_sample(
            #                 cnt_slot_value["book"][slot], seeder=self.seeder
            #             )

            # # makes sure that there are all necessary slots for booking
            # if (
            #     domain == "restaurant"
            #     and "time" not in domain_goal["book"]
            # ):
            #     domain_goal["book"]["time"] = nomial_sample(
            #         cnt_slot_value["book"]["time"], seeder=self.seeder
            #     )

            # if domain == "hotel" and "stay" not in domain_goal["book"]:
            #     domain_goal["book"]["stay"] = nomial_sample(
            #         cnt_slot_value["book"]["stay"], seeder=self.seeder
            #     )

            # if (
            #     domain in ["hotel", "restaurant"]
            #     and "day" not in domain_goal["book"]
            # ):
            #     domain_goal["book"]["day"] = nomial_sample(
            #         cnt_slot_value["book"]["day"], seeder=self.seeder
            #     )

            # if (
            #     domain in ["hotel", "restaurant"]
            #     and "people" not in domain_goal["book"]
            # ):
            #     domain_goal["book"]["people"] = nomial_sample(
            #         cnt_slot_value["book"]["people"], seeder=self.seeder
            #     )

            # if domain == "train" and len(domain_goal["book"]) <= 0:
            #     domain_goal["book"]["people"] = nomial_sample(
            #         cnt_slot_value["book"]["people"], seeder=self.seeder
            #     )

            # fail_info
            domain_goal = self.process_sgd_goal_slot_value(domain_goal, domain)
            if (
                "info" in domain_goal
                and len(self.db.query(domain, domain_goal["info"].items()))
                == 0
            ):
                num_trial = 0
                while num_trial < 10:
                    adjusted_info = self._adjust_info(
                        domain, domain_goal["info"]
                    )
                    adjusted_info = self.process_sgd_goal_slot_value(
                        {"info": adjusted_info}, domain
                    )["info"]
                    if len(self.db.query(domain, adjusted_info.items())) > 0:
                        if domain == "train":
                            domain_goal["info"] = adjusted_info
                        else:
                            # first ask fail_info which return no result then ask info
                            if adjusted_info != domain_goal["info"]:
                                domain_goal["fail_info"] = domain_goal["info"]
                                domain_goal["info"] = adjusted_info

                        break
                    num_trial += 1

                if num_trial >= 10:
                    # continue
                    self.seeder.get("py", random)
                    num_trial2 = 0
                    while num_trial2 < 10:
                        entry = self.seeder.get("py", random).choice(
                            self.db.db[domain]
                        )
                        if all(
                            [
                                slot in entry
                                for slot in domain_goal.get("info", [])
                            ]
                        ) and all(
                            [
                                slot in entry
                                for slot in domain_goal.get("reqt", [])
                            ]
                        ):
                            for slot in domain_goal["info"]:
                                domain_goal["info"][slot] = entry[slot]
                            break
                        num_trial2 += 1
                    if num_trial2 >= 10:
                        continue

            # at least there is one request and book
            if "reqt" in domain_goal or "book" in domain_goal:
                break

        return domain_goal

    def get_user_goal(self):
        domain_ordering = ()
        while len(domain_ordering) <= 0:
            domain_ordering = nomial_sample(
                self.domain_ordering_dist, seeder=self.seeder
            )

        # domain_ordering = [d.lower() for d in domain_ordering]
        domain_ordering = [d for d in domain_ordering]
        user_goal = {
            dom: self._get_domain_goal(dom) for dom in domain_ordering
        }
        assert len(user_goal.keys()) > 0

        for domain in user_goal:
            if not user_goal[domain]["info"]:
                user_goal[domain]["info"] = {"none": "none"}
            new_domain_goal = self.process_sgd_goal_slot_value(
                user_goal[domain], domain
            )
            user_goal[domain] = copy.deepcopy(new_domain_goal)

        user_goal["domain_ordering"] = domain_ordering
        return user_goal

    # process sgd goal to make sure the slots and slot values are in dbs.
    def process_sgd_goal_slot_value(self, domain_goal, domain):
        def process_time(t):
            num_dict = {
                "one": "1",
                "two": "2",
                "three": "3",
                "four": "4",
                "five": "5",
                "six": "6",
                "seven": "7",
                "eight": "8",
                "nine": "9",
                "ten": "10",
                "eleven": "11",
                "twelve": "12",
            }

            def half_past(t):
                t = t.replace("half past", "")
                if ":" in t:
                    t = t.split(":")
                    t2 = int(t[1]) + 30
                    ad = t2 / 60
                    t0 = int(t[0]) + ad
                    t1 = t2 % 60
                    if t1 != 0:
                        t = str(t0) + ":" + str(t1)
                    else:
                        t = str(t0)
                else:
                    t = t + ":30"
                return t

            def quarter_to(t):
                num = re.findall(r"\d+\.?\d*", t)
                return str(int(num[0]) - 1) + ":45"

            def quarter_past(t):
                num = re.findall(r"\d+\.?\d*", t)
                return num[0] + ":15"

            if t in NUL_VALUE:
                return t
            for n in num_dict:
                t = t.replace(n, num_dict[n])
            t = t.replace('o"clock', "")
            t = t.replace("o'clock", "")
            if "am" in t or "morning" in t:
                t = t.replace("am", "").strip()
                if "in the morning" in t:
                    t = t.replace("in the morning", "").strip()
                else:
                    t = t.replace("morning", "").strip()
            elif (
                "pm" in t or "evening" in t or "night" in t or "afternoon" in t
            ):
                t = t.replace("pm", "").strip()
                t = t.replace("in the afternoon", "").strip()
                t = t.replace("afternoon", "").strip()
                t = t.replace("in the night", "").strip()
                t = t.replace("night", "").strip()
                t = t.replace("in the evening", "").strip()
                t = t.replace("evening", "").strip()
                t = t.replace(",", "").strip()  # handle one bug

                if "half past" in t:
                    t = half_past(t)
                if "quarter to" in t:
                    t = quarter_to(t)
                if "quarter past" in t:
                    t = quarter_past(t)
                if ":" in t:
                    t3 = t.split(":")
                    if int(t3[0]) < 12 or (
                        int(t3[0]) == "12" and int(t3[1]) == 0
                    ):
                        t1 = str(int(t3[0]) + 12)
                        t = t1 + ":" + t3[1]
                else:
                    if int(t) <= 12:
                        t = str(int(t) + 12)
            if "half past" in t:
                t = half_past(t)
            if "quarter to" in t:
                t = quarter_to(t)
            if "quarter past" in t:
                t = quarter_past(t)
            t = t.strip()
            if ":" not in t:
                if domain.lower() != "ridesharing":
                    # if t not in [
                    #     "5",
                    #     "2",
                    #     "14",
                    #     "3",
                    #     "11",
                    #     "10",
                    #     "4",
                    #     "13",
                    #     "6",
                    #     "7",
                    #     "8",
                    #     "12",
                    #     "9",
                    # ]:
                    t = t + ":00"
            else:
                t1, t2 = t.split(":")
                if len(t1) == 1:
                    t = "0" + t
            if t == "24:00":
                t = "00:00"
            t = t.strip(",")
            t = t.strip()
            if t in ["03:30"]:
                t = "dontcare"
            return t
            # new_time.append(t.strip())

        def process_day(d):
            d = d.replace("2019", "")
            num = re.findall(r"\d+\.?\d*", d)
            if len(num) > 0:
                d = f"2019-03-{int(num[-1]):02d}"
            if d in [
                "next friday",
                "monday next week",
                "next monday",
                "next tuesday",
                "tuesday next week",
                "thursday next week",
                "next thursday",
                "saturday this week",
                "this saturday",
                "tomorrow",
                "sunday this week",
                "friday next week",
                "later today",
                "wednesday next week",
                "today",
                "next wednesday",
                "day after tomorrow",
                "this sunday",
                "thursday, next week",
                "tuesday, next week",
                "friday, next week",
            ]:
                d = "dontcare"
            return d

        def process_amount(a):
            new_a = copy.deepcopy(a)
            a = a.replace(",", "")
            num_dict1 = {
                "one": "1",
                "two": "2",
                "three": "3",
                "four": "4",
                "five": "5",
                "six": "6",
                "seven": "7",
                "eight": "8",
                "nine": "9",
                "ten": "10",
                "eleven": "11",
                "twelve": "12",
            }
            num_dict2 = {
                "eighty": "80",
                "ninety": "90",
                "forty": "40",
                "thirty": "30",
                "sixty": "60",
                "twenty": "20",
                "fifty": "50",
                "seventy": "70",
            }
            for n in num_dict2:
                a = a.replace(n, num_dict2[n])
            for n in num_dict1:
                a = a.replace(n, num_dict1[n])
            a = a.replace(" ", "")
            num = re.findall(r"\d+\.?\d*", a)
            num = [int(n) for n in num]
            if len(num) == 1:
                if "thousand" in new_a:
                    number = num[0] * 1000
                elif "hundred" in new_a:
                    number = num[0] * 100
                else:
                    number = num[0]
            elif len(num) == 2:
                if "thousand" in new_a and "hundred" in new_a:
                    number = num[0] * 1000 + num[1] * 100
                elif "thousand" in new_a and "hundred" not in new_a:
                    number = num[0] * 1000 + num[1]
                elif "thousand" not in new_a and "hundred" in new_a:
                    number = num[0] * 100 + num[1]
                else:
                    number = num[0] + num[1]
                    print(
                        "thousand not in new_a and hundred in new_a: " + new_a
                    )
            elif len(num) == 3:
                number = num[0] * 1000 + num[1] * 100 + num[2]
            else:
                number = new_a
                print("***", new_a)
            return str(number)

        # unmatche_slot = {}
        new_domain_goal = {}
        # for d in goal["goal"]:
        # if d in ["message"]:
        #     new_goal["goal"] = {}
        #     new_goal["goal"]["message"] = goal["goal"]["message"]
        #     continue
        # domain_goal = goal["goal"][d]
        constrains = {}
        for name in domain_goal:
            if name == "info":
                new_domain_goal[name] = {}
                for k in domain_goal[name]:
                    constrains[k.lower()] = (
                        domain_goal[name][k].lower().strip()
                    )
            elif name == "book":
                new_domain_goal[name] = domain_goal[name]
            elif name == "reqt":
                new_domain_goal[name] = []
                for s in domain_goal[name]:
                    if s.lower() in self.dbs_slot:
                        new_domain_goal[name].append(s.lower())
                    else:
                        print("unresolved slot: ", name, s)
            else:
                new_domain_goal[name] = {}
                for s in domain_goal[name]:
                    if s.lower() in self.dbs_slot:
                        new_domain_goal[name][s] = domain_goal[name][s]
                    else:
                        print("unresolved slot: ", name, s)

            # if name == "reqt":
            #     for k in domain_goal[name]:
            #         constrains[k.lower()] = "?"

        for slot in constrains:
            # dbvalue = dbs_slot[slot]
            # if slot == "depart":
            #     dbvalue = [v.replace(" international airport", "") for v in dbvalue]
            #     dbvalue = [v.replace(" airport", "").strip() for v in dbvalue]
            gv = constrains[slot]
            if gv == "":
                new_domain_goal["info"][slot] = gv
                continue
            # unmatche_slot[slot] = []
            # if gv == "?":
            #     continue
            if slot in "stay":
                e_to_a = {
                    "one": "1",
                    "two": "2",
                    "three": "3",
                    "four": "4",
                    "five": "5",
                    "six": "6",
                    "seven": "7",
                    "eight": "8",
                    "nine": "9",
                    "ten": "10",
                    "twelve": "12",
                }
                if gv in e_to_a:
                    gv = e_to_a[gv]
            if slot == "area":
                if gv in ["san fran", "sfo", "sf"]:
                    gv = "san francisco"
            if slot in ["city", "addr", "city_of_event", "dest", "depart"]:
                if "," in gv:
                    gv = gv.split(",")[0]
                city_dict = {
                    "vancouver bc": "vancouver",
                    "ciudad de mexico": "ciudad de",
                    "atlanta ga": "atlanta",
                    "london uk": "london",
                    "sydney australia": "sydney",
                    "phoenix az": "phoenix",
                    "seattle wa": "seattle",
                    "dc": "washington d.c.",
                    "washington": "washington d.c.",
                    "atl": "atlanta",
                    "la": "los angeles",
                    "london england": "london",
                    "paris france": "paris",
                    "kl": "kuala lumpur",
                    "sd": "sydney",
                    "olema": "olema, california",
                    "new delhi": "delhi",
                    "philly": "philadelphia",
                    "sonoma ave": "sonoma",
                    "sacramento ca": "sacramento",
                    "delhi india": "delhi",
                    "anaheim ca": "anaheim",
                    "hollywood wax museum(r)": "hollywood wax museum",
                    "long beach ca": "long beach",
                }
                if gv in city_dict:
                    gv = city_dict[gv]
                if gv in ["san fran", "sfo", "sf"]:
                    gv = "san francisco"
                elif gv in ["ny", "new york city", "nyc"]:
                    gv = "new york"
                elif gv in [
                    "district of columbia",
                    "lax",
                    "chi-town",
                    "vegas",
                    "ciudad de",
                ]:
                    gv = "dontcare"
                if slot == "depart":
                    gv = gv.replace(" international airport", "")
                    gv = gv.replace(" airport", "").strip()
                    depart_dict = {
                        "washington d.c.": "washington",
                        "fresno": "fresno station",
                        "sacramento": "sacramento valley station",
                        "anaheim": "anaheim intermodal center",
                        "hartsfield": "hartsfield-jackson",
                        "portland or": "portland",
                        "san jose": "san",
                        "fresno ca": "fresno station",
                        "toronto ontario": "toronto",
                        "walnut creek": "walnut creek bart station",
                        "fremont": "fremont bart station",
                        "sunnyvale": "sunnyvale caltrain station",
                    }
                    if gv in depart_dict:
                        gv = depart_dict[gv]
                    if gv in [
                        "cdmx",
                        "long beach",
                        "roissy",
                        "kila",
                        "john f. kennedy",
                        "phoenix sky harbor",
                        "concord",
                    ]:
                        gv = "dontcare"
                elif slot == "dest":
                    dest_dict = {
                        "17 eastern parkway": "17 eastern parkway, brooklyn",
                        "1 tennis place forest hills": (
                            "1 tennis place, forest hills"
                        ),
                        "795 el camino real jamplis building": (
                            "795 el camino real jamplis building, level 1"
                        ),
                        "1820 ogden avenue floor": (
                            "1820 ogden avenue floor, 2"
                        ),
                        "champion hill stadium": (
                            "champion hill stadium, edgar kail way"
                        ),
                        "4812": "4812, 1144 sonoma avenue",
                        "sonoma": "4776 sonoma highway",
                        "portland or": "portland",
                    }
                    if gv in dest_dict:
                        gv = dest_dict[gv]
                    if gv in ["cdmx", "2", "hollywood wax museum"]:
                        gv = "dontcare"
                elif slot == "addr":
                    addr_dict = {
                        "tiburon": "1881 tiburon boulevard",
                        "cloverdale": "105 north cloverdale boulevard",
                        "261 driggs avenue": "261 driggs avenue, brooklyn",
                    }
                    if gv in addr_dict:
                        gv = addr_dict[gv]
                    elif gv in ["hillsborough", "4175", "1", "cdmx"]:
                        gv = "dontcare"
            if slot == "time":
                gv = process_time(gv)
            if slot == "day":
                gv = process_day(gv)
            if slot == "title":
                if gv in [
                    "upside",
                    "the birds movie",
                    "innocent",
                    "how to train your dragon",
                    "mad max",
                    "fighting with family",
                    "shazam",
                    "last dragon",
                    "wild nights",
                    "josie and the pussycats  the pussycats movie",
                    "curse of la llorona",
                    "family funeral",
                    "madmax",
                    "aftermath",
                    "man who knew too much",
                    "visitor",
                    "the vegas movie",
                    "poseidon adventure",
                    "the pussycats movie",
                ]:
                    gv = "dontcare"
            if slot == "amount":
                gv = process_amount(gv)
            if slot == "food":
                if gv in [
                    "freshwater fish",
                    "lobster",
                    "pizza and pasta",
                    "asian fusion",
                    "pick-up",
                    "fish",
                    "dumplings",
                    "burger",
                    "pizza",
                    "noodles",
                    "pasta",
                    "spicy indian",
                    "curry",
                    "punjabi",
                    "diner",
                    "southern",
                    "latin american",
                    "korean barbeque",
                    "middle eastern",
                    "tacos",
                    "spicy noodles",
                    "burgers",
                    "gastrobar",
                    "to-go",
                    "oriental",
                    "szcheuan",
                    "cafe",
                    "parisian",
                    "non meat",
                    "breakfast & brunch",
                    "light meal",
                    "persian",
                    "quick meal",
                    "veggie",
                    "himalayan",
                    "sushi bar",
                    "coffee & light bites",
                    "fast food",
                    "tapas bar",
                    "korean hot pot",
                    "snacks",
                    "soup & salad",
                    "small plates",
                    "healthy meal",
                    "salad bar",
                    "soul food",
                    "deli",
                    "unlimited",
                    "comfort food",
                ]:
                    gv = "dontcare"
            if slot == "type":
                if gv in [
                    "football",
                    "baseball",
                    "soccer",
                    "hip hop",
                    "christian",
                    "international",
                    "basketball",
                    "suspense",
                    "scary",
                    "funny",
                    "non-fiction",
                    "comic",
                    "rom-com",
                    "ghost",
                    "gangster",
                    "detective",
                    "violent",
                    "life history",
                    "fight",
                    "cartoon",
                    "love story",
                    "kids",
                ]:
                    gv = "dontcare"
            if slot == "name":
                if gv in [
                    "palmer's",
                    "marnee",
                    "eric's",
                    "clementine's",
                    "hisui",
                    "yoshio",
                    "aquitaine",
                    "agave grill",
                    "state bird",
                    "thanh long",
                    "the mexican restaurant",
                    "lotus cuisine",
                    "kampai bar",
                    "bear republic",
                    "crab house",
                    "sukho",
                    "burma ruby",
                    "el rancho",
                    "village",
                    "darda",
                    "rangecafe",
                    "spencer's",
                    "will",
                    "yum yum",
                    "luna loca",
                    "ta",
                    "lemongrass",
                    "mendoza's",
                    "paul martin's",
                    "sumac",
                    "kabuto",
                    "dong que",
                    "barrel head",
                    "chili's grill",
                    "wence's",
                    "west park",
                    "steelhead",
                    "la vera",
                    "navin",
                    "crouching tiger",
                    "bob's steakhouse",
                    "amami",
                    "flames eatery",
                    "jing-jing szechwan",
                    "broadway",
                    "my no.1 sushi",
                    "kana",
                    "wooden charcoal barbecue",
                    "the view",
                    "mimi's",
                    "the van's",
                    "jojo restaurant",
                    "hikari sushi",
                    "willi's",
                    "shiki",
                    "parkside",
                    "la panotiq",
                    "mua",
                    "limon",
                    "lark creek",
                    "asya",
                    "black angus",
                    "pearl river",
                    "paradise sushi",
                    "villa d'este",
                    "bj's",
                    "firehouse grill",
                    "the park",
                    "el tesoro",
                    "coupa cafe",
                    "station house",
                    "rivoli",
                    "yayume",
                    "babu ji",
                    "chef zhao",
                    "peacock's koriander",
                    "curse of la llorona",
                    "the curse of la llorona",
                    "wild nights with emily",
                    "knock down the house",
                    "the upside",
                    "shazam",
                    "stockholm",
                    "ash is purest white",
                    "aftermath",
                    "transit",
                    "dr. thomas stodgel",
                    "dr. radhika varma",
                    "dr. andrew sorenson",
                    "dr. werschky ii",
                    "dr. edward lee",
                    "dr. hoang, tuan a.",
                    "dr. roxanne fiscella",
                    "dr. wong hung-kwong",
                    "dr. richard kerbavaz",
                    "dr. carol somersille",
                    "dr. sabi ahmed",
                    "dr. matthew russell",
                    "dr. john smucny",
                    "dr. richard glogau",
                    "dr. claudia pinilla",
                    "dr. david pepper",
                    "dr. anthony boyce",
                    "dr. donna lee",
                    "dr. malini nijagal",
                    "dr. susan logan",
                    "dr. lesley plotke",
                    "dr. jay bansal",
                    "dr. jennifer falk",
                    "dr. silverstein david",
                    "dr. hamblin basil",
                    "dr. vilasini ganesh",
                    "dr. john chiu",
                    "dr. lawrence bruce",
                    "dr. janet bodle",
                    "dr. isaac neuhaus",
                    "dr. carol winton",
                    "dr. gayle sutcliffe",
                    "dr. paley adam",
                    "dr. mauro ruffy",
                    "eyemd of alameda",
                    "dr. amy teng",
                    "dr. shaheen khosla",
                    "dr. ferry james",
                    "dr. gary rust",
                    "dr. edward manche",
                    "dr. donald dossett",
                    "dr. philip lindstrom",
                    "springhill suites fresno",
                    "novotel london tower bridge",
                    "the westin san diego gaslamp quarter",
                    "holiday inn express london",
                    "sls hotel",
                    "travelodge anaheim inn",
                    "homewood suites",
                    "even hotel seattle downtown",
                    "renaissance paris vendome",
                    "extended stay america orange county",
                    "tropicana inn",
                    "ac hotel chicago",
                    "holiday inn express",
                    "fairfield inn central park",
                    "victory house",
                    "travelodge seattle",
                    "clarion inn",
                    "best western jfk",
                    "vibe hotel north",
                    "sanctuary hotel",
                    "super 8 toronto",
                    "towneplace suites downtown",
                    "melia white house",
                    "the cartwright hotel",
                    "best western plus dragon gate",
                    "hotel pullman paris centre - bercy",
                    "homewood suites city avenue",
                    "the queen's gate",
                    "best western plus toronto north york hotel",
                    "catamaran resort hotel",
                    "veriu central",
                    "homewood suites pike street",
                    "concorde hotel",
                    "holiday inn whitechapel",
                    "best western victoria palace",
                    "the kitano hotel",
                    "hotel ibis paris grands boulevards opera 9eme",
                    "meriton suites north",
                    "embassy suites biltmore",
                    "sheba piano",
                    "artisan",
                    "palermo",
                    "fontana's",
                    "tambo",
                    "kuleto's",
                    "up 2u",
                    "koryo",
                    "opera plaza",
                    "amc newpark",
                    "amc saratoga",
                    "century oakridge",
                    "invisibles",
                    "pruneyard cinemas",
                    "the lalit new",
                    "seraphine hammersmith",
                    "north sydney harbourview",
                    "la quinta inn lax",
                    "days inn san francisco downtown",
                    "parc 55 san francisco",
                    "ac hotel downtown",
                    "homewood suites del mar",
                    "embassy suites buckhead",
                    "four points soho",
                    "hotel zeppelin",
                    "best western bowery hanbee",
                    "the westbury mayfair",
                    "hilton garden inn new york central park south",
                    "fairfield inn magnificent mile",
                    "holiday inn vancouver-centre",
                    "club quarters hotel rockefeller center",
                    "the old clare",
                    "embassy suites portland",
                    "karma sanctum soho",
                    "citadines place d'italie paris",
                    "the queens park",
                    "sure hotel",
                    "the pelham london",
                    "the gregory hotel",
                    "ac hotel porte maillot",
                    "hampton inn downtown",
                    "fairfield inn buckhead",
                    "gec granville suites",
                    "point a hotel kings cross",
                    "the whitley",
                    "hotel indigo kensington",
                    "hotel du printemps",
                    "hilton garden inn jalan tuanku abdul rahman north",
                    "citadines barbican london",
                    "abc hyde park",
                    "holiday inn express swiss cottage",
                    "best western plus san pedro hotel",
                    "blue sea beach",
                    "hotel indigo paddington",
                    "andaz delhi",
                    "embassy suites downtown north",
                    "residence inn portland",
                    "adria hotel",
                    "la clef louvre",
                    "sofitel philadelphia",
                    "kimpton hotel seattle",
                    "doubletree by hilton hotel san pedro",
                    "the american hotel atlanta downtown - a doubletree",
                    "country inn portland",
                    "vivanta new delhi",
                    "st christopher's inn village | hostel",
                    "holiday inn new delhi airport",
                    "point a hotel shoreditch",
                    "days inn broadway",
                    "quality inn",
                    "how to train your dragon",
                    "poseidon adventure",
                    "madmax",
                    "upside",
                    "reservation at gumbas",
                    "innocent",
                    "mad max",
                    "family funeral",
                    "the gore london",
                    "holiday inn anaheim",
                    "holiday inn sacramento convention center",
                    "marriott vacation club mayflower",
                    "travelodge by wyndham downtown",
                    "four points downtown",
                    "citadines tour eiffel paris",
                    "mercure london hyde park",
                    "comfort suites michigan avenue",
                    "homewood suites chicago-downtown",
                    "k+k hotel cayre",
                    "la quinta inn seaworld",
                    "the westin kierland resort",
                    "the muse sarovar portico kapashera",
                    "holiday inn express & suites la jolla",
                    "motel 6 san diego hotel circle",
                    "hyatt centric the pike",
                    "hilton garden inn saket",
                    "radisson blu kenilworth",
                    "sheraton philadelphia society hill",
                    "hotel vertigo",
                    "the arctic club seattle - a doubletree",
                    "toronto marriott city centre",
                    "holiday inn express victoria",
                    "the district",
                    "best western burns",
                    "hub by premier inn westminster",
                    "ibis london city",
                    "hilton checkers",
                    "hotel marignan champs-elysees",
                    "residence inn downtown",
                    "the marcel",
                    "la quinta inn seattle",
                    "comfort inn portland",
                    "le roch hotel",
                    "homewood suites mission valley",
                    "tivoli garden resort",
                    "the gwen",
                    "dorsett city",
                    "pier one sydney harbour",
                    "the sydney boulevard",
                    "sea containers london",
                    "residence inn river north",
                    "residence inn midtown east",
                    "maison breguet",
                    "hotel current",
                    "green tortoise hostel",
                    "venice on the beach",
                    "hilton woodland hills",
                    "citadines trocadero paris (apart hotel paris)",
                    "montecassino hotel",
                    "best western plus plaza",
                    "one washington circle",
                    "springhill suites seattle",
                    "hilton garden inn midtown park ave",
                    "the tuscany",
                    "comfort inn victoria",
                    "the pussycats movie",
                    "playa",
                    "last dragon",
                    "le tasty",
                    "the vegas movie",
                    "prima",
                    "fradelizio's",
                    "bowl'd",
                    "senro",
                    "anjappar",
                    "roya",
                    "thai chili",
                    "la finestra",
                    "alfred's",
                    "mei-don",
                    "sinaloa cafe",
                    "trader vic's",
                    "yellow chilli",
                    "dickey's",
                    "tony & alba's",
                    "mr mom's",
                    "tai yuan",
                    "vanessa's bistro 2",
                    "j's garden",
                    "kettles",
                    "montesacro pinseria",
                    "mcdonalds",
                    "dinah's",
                    "le petit",
                    "dj's",
                    "union",
                    "crocodile",
                    "roppongi",
                    "eiko's",
                    "napoli pizza",
                    "mary's pizza",
                    "osha",
                    "everest",
                    "wine cellar",
                    "aslam's",
                    "layang layang",
                    "hatcho",
                    "kamakura",
                    "royal thai",
                    "noir",
                    "spalti",
                    "sakana sushi",
                    "krua",
                    "frida's",
                    "xiao loong",
                    "sala",
                    "8 dragons",
                    "favela's",
                    "tachi",
                    "tuba",
                    "trancas",
                    "truya",
                    "picco",
                    "banana leaf",
                    "lapats noodles",
                    "regent",
                    "filippi's pizza",
                    "rue lepic",
                    "seven restaurant",
                    "myriad",
                    "lotus chaat",
                    "lotus garden",
                    "wildfox",
                    "ace wasabi",
                    "suraj",
                    "the pullman",
                    "pasquale's",
                    "picnic on third",
                    "kincaid's",
                    "may lee",
                    "tribu",
                    "novy",
                    "addis",
                    "scoma's",
                    "maya palenque",
                    "saigon seafood",
                    "fiery hot pot",
                    "old mandarin",
                    "poorboy's",
                    "amaravati",
                    "sakura",
                    "sideshow",
                    "dishdash",
                    "mount everest",
                    "madhuban",
                    "estampas",
                    "bashamichi",
                    "sessions",
                    "la fogata",
                    "coconut bay",
                    "masa's",
                    "papa ray's",
                    "zakuro",
                    "la viga seafood",
                    "la mar",
                    "sumo",
                    "tamarine",
                    "shido",
                    "brickhouse cafe",
                    "hokkaido",
                    "there sushi",
                    "gen korean bbq",
                    "won kee",
                    "paradise",
                    "siam palace",
                    "hanami",
                    "uncle buck's",
                    "barranco cocina",
                    "italico",
                    "imperial",
                    "lighthouse bar",
                    "amc eastridge",
                    "wild nights",
                ]:
                    gv = "dontcare"
            if slot == "stars":
                gv = gv + "0" if "." in gv else gv + ".00"
            new_domain_goal["info"][slot] = gv
            # if gv not in dbvalue and gv not in ['dontcare']:
            #     if gv not in unmatche_slot[slot]:
            #         unmatche_slot[slot].append(gv)
            #         pprint(goal)
            #         pprint(unmatche_slot)
        return new_domain_goal

    # to get the slot_value sets for each slot of the domains
    def get_dbs_slot(self, domain_dbs):
        dbs_slot = {}
        for dm in domain_dbs:
            if dm in ["taxi", "topic"]:
                continue
            dbs = domain_dbs[dm]
            for db in dbs:
                new_db = {}
                for k in db:
                    new_db[k.lower()] = str(db[k]).lower().strip()
                cslot_dbs = list(new_db.keys())
                for c in cslot_dbs:
                    if c not in dbs_slot:
                        dbs_slot[c] = [new_db[c]]
                    else:
                        if new_db[c] not in dbs_slot[c]:
                            dbs_slot[c].append(new_db[c])
        return dbs_slot

    def _adjust_info(self, domain, info):
        # adjust one of the slots of the info
        adjusted_info = deepcopy(info)
        slot = self.seeder.get("py", random).choice(list(info.keys()))
        value = self.seeder.get("py", random).choice(
            list(self.ind_slot_value_dist[domain]["info"][slot].keys())
        )
        if adjusted_info[slot] != value:  # new added on 2023,01,02
            adjusted_info[slot] = copy.deepcopy(value)
        return adjusted_info

    def build_message(self, user_goal, boldify=null_boldify):
        message = []
        message_by_domain = []
        mess_ptr4domain = 0
        state = deepcopy(user_goal)

        for dom in user_goal["domain_ordering"]:
            dom_msg = []
            state = deepcopy(user_goal[dom])
            num_acts_in_unit = 0

            if not (dom == "taxi" and len(state["info"]) == 1):
                # intro
                m = [templates[dom]["intro"]]

            # info
            def fill_info_template(user_goal, domain, slot, info):
                if slot != "area" or not (
                    "restaurant" in user_goal
                    and "attraction" in user_goal
                    and info in user_goal["restaurant"].keys()
                    and info in user_goal["attraction"].keys()
                    and "area" in user_goal["restaurant"][info]
                    and "area" in user_goal["attraction"][info]
                    and user_goal["restaurant"][info]["area"]
                    == user_goal["attraction"][info]["area"]
                ):
                    return templates[domain][slot].format(
                        self.boldify(user_goal[domain][info][slot])
                    )
                else:
                    restaurant_index = user_goal["domain_ordering"].index(
                        "restaurant"
                    )
                    attraction_index = user_goal["domain_ordering"].index(
                        "attraction"
                    )
                    if (
                        restaurant_index > attraction_index
                        and domain == "restaurant"
                    ):
                        return templates[domain][slot].format(
                            self.boldify("same area as the attraction")
                        )
                    elif (
                        attraction_index > restaurant_index
                        and domain == "attraction"
                    ):
                        return templates[domain][slot].format(
                            self.boldify("same area as the restaurant")
                        )
                return templates[domain][slot].format(
                    self.boldify(user_goal[domain][info][slot])
                )

            info = "info"
            if "fail_info" in user_goal[dom]:
                info = "fail_info"
            if dom == "taxi" and len(state[info]) == 1:
                taxi_index = user_goal["domain_ordering"].index("taxi")
                places = [
                    dom
                    for dom in user_goal["domain_ordering"][:taxi_index]
                    if dom in ["attraction", "hotel", "restaurant"]
                ]
                if len(places) >= 2:
                    self.seeder.get("py", random).shuffle(places)
                    m.append(templates["taxi"]["commute"])
                    if "arriveBy" in state[info]:
                        m.append(
                            "The taxi should arrive at the {} from the {}"
                            " by {}.".format(
                                self.boldify(places[0]),
                                self.boldify(places[1]),
                                self.boldify(state[info]["arriveBy"]),
                            )
                        )
                    elif "leaveAt" in state[info]:
                        m.append(
                            "The taxi should leave from the {} to the {}"
                            " after {}.".format(
                                self.boldify(places[0]),
                                self.boldify(places[1]),
                                self.boldify(state[info]["leaveAt"]),
                            )
                        )
                    message.append(" ".join(m))
            else:
                while len(state[info]) > 0:
                    num_acts = self.seeder.get("py", random).randint(
                        1, min(len(state[info]), 3)
                    )
                    slots = self.seeder.get("py", random).sample(
                        list(state[info].keys()), num_acts
                    )
                    sents = [
                        fill_info_template(user_goal, dom, slot, info)
                        for slot in slots
                        if slot not in ["parking", "internet"]
                    ]
                    if "parking" in slots:
                        sents.append(
                            templates[dom]["parking " + state[info]["parking"]]
                        )
                    if "internet" in slots:
                        sents.append(
                            templates[dom][
                                "internet " + state[info]["internet"]
                            ]
                        )
                    m.extend(sents)
                    message.append(" ".join(m))
                    m = []
                    for slot in slots:
                        del state[info][slot]

            # fail_info
            if "fail_info" in user_goal[dom]:
                # if 'fail_info' in user_goal[dom]:
                adjusted_slot = list(
                    filter(
                        lambda x: x[0][1] != x[1][1],
                        zip(
                            user_goal[dom]["info"].items(),
                            user_goal[dom]["fail_info"].items(),
                        ),
                    )
                )[0][0][0]
                if adjusted_slot in ["internet", "parking"]:
                    message.append(
                        templates[dom][
                            "fail_info "
                            + adjusted_slot
                            + " "
                            + user_goal[dom]["info"][adjusted_slot]
                        ]
                    )
                else:
                    message.append(
                        templates[dom]["fail_info " + adjusted_slot].format(
                            self.boldify(user_goal[dom]["info"][adjusted_slot])
                        )
                    )

            # reqt
            if "reqt" in state:
                slot_strings = []
                for slot in state["reqt"]:
                    if slot in ["internet", "parking", "food"]:
                        continue
                    slot_strings.append(
                        slot
                        if slot not in request_slot_string_map
                        else request_slot_string_map[slot]
                    )
                if len(slot_strings) > 0:
                    message.append(
                        templates[dom]["request"].format(
                            self.boldify(", ".join(slot_strings))
                        )
                    )
                if "internet" in state["reqt"]:
                    message.append(
                        "Make sure to ask if the hotel includes free wifi."
                    )
                if "parking" in state["reqt"]:
                    message.append(
                        "Make sure to ask if the hotel includes free parking."
                    )
                if "food" in state["reqt"]:
                    message.append(
                        "Make sure to ask about what food it serves."
                    )

            def get_same_people_domain(user_goal, domain, slot):
                if slot not in ["day", "people"]:
                    return None
                domain_index = user_goal["domain_ordering"].index(domain)
                previous_domains = user_goal["domain_ordering"][:domain_index]
                for prev in previous_domains:
                    if (
                        prev in ["restaurant", "hotel", "train"]
                        and "book" in user_goal[prev]
                        and slot in user_goal[prev]["book"]
                        and user_goal[prev]["book"][slot]
                        == user_goal[domain]["book"][slot]
                    ):
                        return prev
                return None

            # book
            book = "book"
            if "fail_book" in user_goal[dom]:
                book = "fail_book"
            if "book" in state:
                slot_strings = []
                for slot in ["people", "time", "day", "stay"]:
                    if slot in state[book]:
                        if slot == "people":
                            same_people_domain = get_same_people_domain(
                                user_goal, dom, slot
                            )
                            if same_people_domain is None:
                                slot_strings.append(
                                    "for {} people".format(
                                        self.boldify(state[book][slot])
                                    )
                                )
                            else:
                                slot_strings.append(
                                    self.boldify(
                                        "for the same group of people as the"
                                        " {} booking".format(
                                            same_people_domain
                                        )
                                    )
                                )
                        elif slot == "time":
                            slot_strings.append(
                                "at {}".format(self.boldify(state[book][slot]))
                            )
                        elif slot == "day":
                            same_people_domain = get_same_people_domain(
                                user_goal, dom, slot
                            )
                            if same_people_domain is None:
                                slot_strings.append(
                                    "on {}".format(
                                        self.boldify(state[book][slot])
                                    )
                                )
                            else:
                                slot_strings.append(
                                    self.boldify(
                                        "on the same day as the {} booking"
                                        .format(same_people_domain)
                                    )
                                )
                        elif slot == "stay":
                            slot_strings.append(
                                "for {} nights".format(
                                    self.boldify(state[book][slot])
                                )
                            )
                        del state[book][slot]

                assert len(state[book]) <= 0, state[book]

                if len(slot_strings) > 0:
                    message.append(
                        templates[dom]["book"].format(" ".join(slot_strings))
                    )

            # fail_book
            if "fail_book" in user_goal[dom]:
                adjusted_slot = list(
                    filter(
                        lambda x: x[0][1] != x[1][1],
                        zip(
                            user_goal[dom]["book"].items(),
                            user_goal[dom]["fail_book"].items(),
                        ),
                    )
                )[0][0][0]

                if adjusted_slot in ["internet", "parking"]:
                    message.append(
                        templates[dom][
                            "fail_book "
                            + adjusted_slot
                            + " "
                            + user_goal[dom]["book"][adjusted_slot]
                        ]
                    )
                else:
                    message.append(
                        templates[dom]["fail_book " + adjusted_slot].format(
                            self.boldify(user_goal[dom]["book"][adjusted_slot])
                        )
                    )

            dm = message[mess_ptr4domain:]
            mess_ptr4domain = len(message)
            message_by_domain.append(" ".join(dm))

        if boldify == do_boldify:
            for i, m in enumerate(message):
                message[i] = message[i].replace("wifi", "<b>wifi</b>")
                message[i] = message[i].replace("internet", "<b>internet</b>")
                message[i] = message[i].replace("parking", "<b>parking</b>")

        return message, message_by_domain


if __name__ == "__main__":
    goal_generator = GoalGenerator(
        # corpus_path=os.path.join(get_root_path(), "data/multiwoz/test.json"),
        corpus_path="sgd_goal.json",
        sample_reqt_from_trainset=False,
        data_set="sgd",
    )
    print("finished build goal!")
    user_goal = goal_generator.get_user_goal()
    pprint(user_goal)
    # message = goal_generator.build_message(user_goal)
    c = 0
