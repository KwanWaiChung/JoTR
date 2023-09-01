"""
This evaluates the generated dialogues from a model with the gold
corpus dialogues.

Mostly inspired by the Marco code since it's cleaner.
"""
from typing import Dict, Union, List, Any, Tuple
from convlab.data.multiwoz.dbquery2 import Database
from nltk.util import ngrams
from collections import Counter
import json
import math

domains = [
    "restaurant",
    "hotel",
    "attraction",
    "train",
    "taxi",
    "hospital",
    "police",
]
requestables = ["phone", "address", "postcode", "reference", "id"]
null_values = [
    "",
    "dontcare",
    "not mentioned",
    "don't caore",
    "dont care",
    "do n't care",
    "none",
]


class BLEUScorer:
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def score(self, hypothesis, corpus, n=1):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]

            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(
                                max_counts.get(ng, 0), refcnts[ng]
                            )
                    clipcnt = dict(
                        (ng, min(count, max_counts[ng]))
                        for ng, count in hypcnts.items()
                    )
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0:
                        break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [
            float(clip_count[i]) / float(count[i] + p0) + p0 for i in range(4)
        ]
        s = math.fsum(
            w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n
        )
        bleu = bp * math.exp(s)
        return bleu


class MultiwozCorpusEvaluator:
    def __init__(self):
        self.db = Database()
        self.bs_scorer = BLEUScorer()

    def _query(self, domain: str, constraints: Dict[str, str]) -> List[str]:
        """Wrapper of the db.query method. Inspiried by UBAR implimentation.
            See queryJsons in the following link
            https://github.com/TonyNemo/UBAR-MultiWOZ/blob/master/eval.py

        Args:
            domain (str)
            constraints (Dict[str,str])

        """
        match_results = self.db.query(domain, tuple(constraints.items()))
        if domain == "train":
            match_results = [e["trainID"] for e in match_results]
        else:
            match_results = [e["name"] for e in match_results]
        return match_results

    def _parseGoal(
        self,
        user_goal: Dict[str, Dict[str, Union[Dict[str, str], List[str]]]],
        domain: str,
    ) -> Dict[str, Union[Dict[str, str], List[str]]]:
        """Parses user goal into dictionary format."""
        domain_goal = {"informable": {}, "requestable": [], "booking": []}
        if "info" in user_goal[domain]:
            if domain == "train":
                # we consider dialogues only where train had to be booked!
                if "book" in user_goal[domain]:
                    domain_goal["requestable"].append("reference")
                if "reqt" in user_goal[domain]:
                    if "trainID" in user_goal[domain]["reqt"]:
                        domain_goal["requestable"].append("id")
            else:
                if "reqt" in user_goal[domain]:
                    for s in user_goal[domain]["reqt"]:  # addtional requests:
                        if s in [
                            "phone",
                            "address",
                            "postcode",
                            "reference",
                            "id",
                        ]:
                            # ones that can be easily delexicalized
                            domain_goal["requestable"].append(s)
                if "book" in user_goal[domain]:
                    domain_goal["requestable"].append("reference")

            domain_goal["informable"] = user_goal[domain]["info"]
            if "book" in user_goal[domain]:
                domain_goal["booking"] = user_goal[domain]["book"]
        return domain_goal

    def _evaluateDST(
        self,
        real_dialog: Dict[str, Any],
        pred_belief: List[Dict[str, str]] = None,
    ):
        """
        Args:
            pred_dialog (List[str]): List of predicted sys response.
            real_dialog (Dict[str, Any]): It contains key `goal` and `log`.
                `goal` points Dict of domain to Dict of `info`, `book` etc.
                `log` points to List of Dict of `text` and `metadata`.

        """
        dst_match = 0
        dst_total = 0
        if pred_belief is None:
            return 0, 0
        for t, turn_pred_belief in enumerate(pred_belief):
            # evaluateDST here
            turn_gt_belief = {}
            for domain in domains:
                gt_belief = {
                    s.lower(): v.lower()
                    for s, v in real_dialog["log"][t * 2 + 1]["metadata"][
                        domain
                    ]["semi"].items()
                    if v not in null_values
                }
                if gt_belief:
                    turn_gt_belief[domain] = gt_belief

                gt_belief_book = {
                    s.lower(): v.lower()
                    for s, v in real_dialog["log"][t * 2 + 1]["metadata"][
                        domain
                    ]["book"].items()
                    if v not in null_values and s != "booked"
                }
                if gt_belief_book:
                    turn_gt_belief.setdefault("booking", {}).update(
                        gt_belief_book
                    )
                dst_match += turn_gt_belief == turn_pred_belief
                dst_total += 1
            return dst_match, dst_total

    def _evaluateDialogue(
        self,
        pred_dialog: List[str],
        real_dialog: Dict[str, Any],
        pred_belief: List[Dict[str, str]] = None,
    ):
        """
        Args:
            pred_dialog (List[str]): List of predicted sys response.
            real_dialog (Dict[str, Any]): It contains key `goal` and `log`.
                `goal` points Dict of domain to Dict of `info`, `book` etc.
                `log` points to List of Dict of `text` and `metadata`.

        """
        # get the list of domains in the goal
        goal = {}
        for domain in domains:
            if real_dialog["goal"][domain]:
                goal[domain] = self._parseGoal(real_dialog["goal"], domain)

        real_requestables = {}
        for domain in goal.keys():
            real_requestables[domain] = goal[domain]["requestable"]

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []

        for t, sent_t in enumerate(pred_dialog):
            for domain in goal.keys():
                # Search for the only restaurant, hotel, attraction or train with an ID
                if "[" + domain + "_name]" in sent_t or "trainid]" in sent_t:
                    if domain in [
                        "restaurant",
                        "hotel",
                        "attraction",
                        "train",
                    ]:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        if pred_belief is not None:
                            venues = self._query(
                                domain=domain,
                                constraints=pred_belief[t].get(domain, {}),
                            )
                        else:
                            venues = self._query(
                                domain=domain,
                                constraints=real_dialog["log"][t * 2 + 1][
                                    "metadata"
                                ][domain]["semi"],
                            )
                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[
                                domain
                            ] = venues  # random.sample(venues, 1)
                        else:
                            flag = True
                            for ven in venue_offered[domain]:
                                if ven not in venues:
                                    flag = False
                                    break
                            if (
                                not flag and venues
                            ):  # sometimes there are no results so sample won't work
                                venue_offered[domain] = venues
                    else:
                        venue_offered[domain] = "[" + domain + "_name]"

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == "reference":
                        if domain + "_reference" in sent_t:
                            if "restaurant_reference" in sent_t:
                                if (
                                    real_dialog["log"][t * 2]["db_pointer"][-5]
                                    == 1
                                ):  # if pointer was allowing for that?
                                    provided_requestables[domain].append(
                                        "reference"
                                    )

                            elif "hotel_reference" in sent_t:
                                if (
                                    real_dialog["log"][t * 2]["db_pointer"][-3]
                                    == 1
                                ):  # if pointer was allowing for that?
                                    provided_requestables[domain].append(
                                        "reference"
                                    )

                            elif "train_reference" in sent_t:
                                if (
                                    real_dialog["log"][t * 2]["db_pointer"][-1]
                                    == 1
                                ):  # if pointer was allowing for that?
                                    provided_requestables[domain].append(
                                        "reference"
                                    )

                            else:
                                provided_requestables[domain].append(
                                    "reference"
                                )
                    else:
                        if domain + "_" + requestable + "]" in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            if "name" in goal[domain]["informable"]:
                venue_offered[domain] = "[" + domain + "_name]"

            # special domains - entity does not need to be provided
            if domain in ["taxi", "police", "hospital"]:
                venue_offered[domain] = "[" + domain + "_name]"

            if domain == "train":
                if not venue_offered[domain]:
                    if (
                        goal[domain]["requestable"]
                        and "id" not in goal[domain]["requestable"]
                    ):
                        venue_offered[domain] = "[" + domain + "_name]"
        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {
            "restaurant": [0, 0, 0],
            "hotel": [0, 0, 0],
            "attraction": [0, 0, 0],
            "train": [0, 0, 0],
            "taxi": [0, 0, 0],
            "hospital": [0, 0, 0],
            "police": [0, 0, 0],
        }

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ["restaurant", "hotel", "attraction", "train"]:
                if (
                    type(venue_offered[domain]) is str
                    and "_name" in venue_offered[domain]
                ):
                    match += 1
                    match_stat = 1
                elif venue_offered[domain]:
                    goal_venues = self._query(
                        domain=domain, constraints=goal[domain]["informable"]
                    )
                    if set(venue_offered[domain]).issubset(set(goal_venues)):
                        match += 1
                        match_stat = 1
            else:
                if "[" + domain + "_name]" in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if match == len(goal):
            match = 1
        else:
            match = 0

        # SUCCESS
        if match:
            for domain in goal.keys():
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            if success >= len(real_requestables):
                success = 1
            else:
                success = 0
        return success, match, stats

    def evaluateModel(
        self,
        pred_dialog: Dict[str, List[str]],
        real_dialog: Dict[str, Dict[str, Any]],
        pred_belief: Dict[str, List[Dict[str, str]]] = None,
    ) -> Tuple[float, float, float, float]:
        """Gathers statistics for the whole sets."""
        # fin1 = open("data/multiwoz/delex.json")
        # real_dialog = json.load(fin1)
        successes, matches = 0, 0
        dst_matches, dst_total = 0, 0
        total = 0

        domain_success = {}
        corpus = []
        gold_corpus = []
        for filename, dial in pred_dialog.items():
            domain_success[filename] = {}
            data = real_dialog[filename]
            belief = pred_belief[filename] if pred_belief is not None else None
            success, match, _ = self._evaluateDialogue(dial, data, belief)
            _dst_matches, _dst_total = self._evaluateDST(data, belief)
            domain_success[filename]["inform"] = match
            domain_success[filename]["request"] = success
            dst_matches += _dst_matches
            dst_total += _dst_total
            successes += success
            matches += match
            total += 1

            # turns = [
            #     turn["text"]
            #     for i, turn in enumerate(data["log"])
            #     if i % 2 == 1
            # ]
            turns = []
            gold_turns = []
            for i, turn in enumerate(data["log"]):
                if i % 2 == 1:
                    gold_turns.append([turn["text"]])
            for turn in dial:
                turns.append([turn])
            assert len(turns) == len(gold_turns)
            corpus.extend(turns)
            gold_corpus.extend(gold_turns)
        bleu_score = self.bs_scorer.score(corpus, gold_corpus)

        # Print results
        matches = matches / float(total) * 100
        successes = successes / float(total) * 100
        dst_matches = dst_matches / float(dst_total) * 100

        # print("Corpus Inform Success : %2.2f%%" % (matches))
        # print("Corpus Requestable Success : %2.2f%%" % (successes))
        # return "{}_{}".format("%2.2f"%bleu, matches, successes)
        return (matches, successes, bleu_score, dst_matches)

