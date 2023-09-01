import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import logging
import logging.config
import datetime
import os
import torch.nn.functional as F
import wandb
from typing import List, Dict, Tuple, Union
from torch import nn

# plt.figure(dpi=300)
sns.set_theme()


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


level_map = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def get_random_state():
    return (
        torch.random.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
    )


def set_random_state(states):
    torch.random.set_rng_state(states[0])
    np.random.set_state(states[1])
    random.setstate(states[2])


def format_size(size: int):
    suffix = "B"
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(size) < 1024.0:
            return "%3.1f %s%s" % (size, unit, suffix)
        size /= 1024.0
    return "%.1f %s%s" % (size, "Yi", suffix)


def print_tensor_size(nlargest: int = 10):
    sizes = []
    for name, value in globals().items():
        if isinstance(value, torch.Tensor):
            sizes.append((name, "tensor", sys.getsizeof(value.storage())))
        elif isinstance(value, nn.Module):
            size = sum(
                [
                    sys.getsizeof(param.storage())
                    for param in list(value.parameters())
                ]
            )
            sizes.append((name, "module", size))
    for name, type, size in sorted(
        sizes,
        key=lambda x: x[1],
        reverse=True,
    )[:nlargest]:
        print(
            "{:>30}{:>10}: {:>8}".format(
                name, "(" + type + ")", format_size(size)
            )
        )


def get_seeder(seed):
    pt_g = torch.manual_seed(seed)
    np_g = np.random.default_rng(seed)
    py_g = random.Random(seed)
    return {"pt": pt_g, "np": np_g, "py": py_g}


def lineplot(
    data: pd.DataFrame,
    x: str,
    y: Union[str, List[str]],
    hue: str = None,
    hue_order: List[str] = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    legend_title: str = None,
    out_filename: str = None,
    show: bool = True,
):
    plt.clf()
    if isinstance(y, list):
        data = data.melt(id_vars=[x], value_vars=y)
        ax = sns.lineplot(
            data=data, x=x, y="value", hue="variable", hue_order=hue_order
        )
    else:
        ax = sns.lineplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend_title is not None:
        ax.get_legend().set_title(legend_title)
    if out_filename:
        plt.savefig(out_filename)
    if show:
        plt.show()
    return ax


def get_logger(
    name: str = None,
    logger_level: str = None,
    console_level: str = None,
    file_level: str = None,
    log_path: str = None,
):
    """Configure the logger and return it.

    Args:
        name (str, optional): Name of the logger, usually __name__.
            Defaults to None. None refers to root logger, usually useful
            when setting the default config at the top level script.
        logger_level (str, optional): level of logger. Defaults to None.
            None is treated as `debug`.
        console_level (str, optional): level of console. Defaults to None.
            None is treated as `debug`.
        file_level (str, optional): level of file. Defaults to None.
            None is treated `debug`.
        log_path (str, optional): The path of the log.

    Note that console_level should only be used when configuring the
    root logger.
    """

    logger = logging.getLogger(name)
    if name:
        logger.setLevel(level_map[logger_level or "debug"])
    else:  # root logger default lv should be high to avoid external lib log
        logger.setLevel(level_map[logger_level or "warning"])
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # set up the logfile handler
    if file_level and log_path:
        logTime = datetime.datetime.now()
        fn1, fn2 = os.path.splitext(log_path)
        log_filename = f"{fn1}-{logTime.strftime('%Y%m%d-%H%M%S')}{fn2}"
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        fh = logging.FileHandler(log_filename)
        fh.setLevel(level_map[file_level or "debug"])
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # set up the console/stream handler
    if name and console_level:
        raise ValueError(
            "`console_level` should only be set when configuring root logger."
        )
    if console_level:
        sh = logging.StreamHandler()
        sh.setLevel(level_map[console_level or "debug"])
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger


def get_gradients(model) -> torch.tensor:
    """

    Returns:
        torch.tensor: The flatten gradients.

    """
    return torch.cat(
        [p.grad.reshape(-1) for p in model.parameters() if p is not None],
        dim=0,
    )


def get_params(model) -> torch.tensor:
    """

    Returns:
        torch.tensor: The flatten parameters.

    """
    return torch.cat(
        [p.reshape(-1) for p in model.parameters() if p is not None],
        dim=0,
    )


def get_diaact_list(
    diaacts: Dict[str, List[List[str]]], intent_first=False, add_value=False
) -> List[Tuple[str, str, str]]:
    """_summary_

    Args:
        diaacts (Dict[str, List[List[str]]]): Dict of d-i -> [[s,v]]

    Returns:
        List[Tuple[str, str, str]]:  domain, intent, slot
    """
    new_acts = []
    for di, svs in diaacts.items():
        d, i = di.split("-")
        for s, v in svs:
            act = []
            if intent_first:
                act += [i, d, s]
            else:
                act += [d, i, s]
            if add_value:
                act.append(v)
            new_acts.append(tuple(act))
    return new_acts


def get_diaact_dict(
    acts: List[List[str]], intent_first=True
) -> Dict[str, List[List[str]]]:
    new_acts = {}
    for i, d, s, v in acts:
        if not intent_first:
            d, i = i, d
        di = f"{d}-{i}"
        if di not in new_acts:
            new_acts[di] = []
        new_acts[di].append([s, v])
    return new_acts


def str_to_ids(
    flatten_diaact_str: str,
    tokenizer,
    domains: List[str],
    intents: List[str],
    slots: List[str],
) -> Tuple[List[int], List[int], List[int]]:
    symbols = ["[d]", "[i]", "[s]", "[db]"]
    input_ids = []
    label_ids = []
    cat_ids = []
    prev_symb = -1
    for token in flatten_diaact_str.split():
        if token in symbols:
            input_ids.append(tokenizer.convert_tokens_to_ids(token))
            prev_symb = symbols.index(token)
            cat_ids.append(prev_symb)
            label_ids.append(-100)
        elif prev_symb == 0:  # domain
            _input_ids: List[int] = tokenizer(token, add_prefix_space=True)[
                "input_ids"
            ]
            cat_ids += [-100] * len(_input_ids)
            label_ids += [domains.index(token)] + [-100] * (
                len(_input_ids) - 1
            )
            input_ids += _input_ids
        elif prev_symb == 1:  # intent
            _input_ids: List[int] = tokenizer(token, add_prefix_space=True)[
                "input_ids"
            ]
            cat_ids += [-100] * len(_input_ids)
            label_ids += [intents.index(token)] + [-100] * (
                len(_input_ids) - 1
            )
            input_ids += _input_ids
        elif prev_symb == 2:  # slot
            _input_ids: List[int] = tokenizer(token, add_prefix_space=True)[
                "input_ids"
            ]
            cat_ids += [-100] * len(_input_ids)
            label_ids += [slots.index(token)] + [-100] * (len(_input_ids) - 1)
            input_ids += _input_ids
        elif prev_symb == 3:  # db
            _input_ids: List[int] = tokenizer(token, add_prefix_space=True)[
                "input_ids"
            ]
            cat_ids += [-100] * len(_input_ids)
            label_ids += [-100] * len(_input_ids)
            input_ids += _input_ids
        else:
            raise ValueError("Invalid condition.")
    return input_ids, label_ids, cat_ids


def str_to_diaact(
    sequence: str,
) -> List[Tuple[str, str, str]]:
    """
    This method transform the format for the multi head gpt model.

    Args:
        sequence (str): The dialog act str

    Returns:
        The List of (domain, intent, slot) dialog acts.

    """
    input_str: List[str] = sequence.split()

    symbs = ["[d]", "[i]", "[s]", "[db]"]
    prev_symb = ""
    prev_domain = ""
    prev_intent = ""
    next_symb = ["[d]"]
    i = 0
    n = len(input_str)
    diaact_preds = []

    while i < n:
        token = input_str[i].strip()
        if token in symbs:
            if token not in next_symb:  # invalid order
                prev_symb = ""
                if next_symb != ["[d]"]:  # retry with [d]
                    next_symb = ["[d]"]
                else:
                    i += 1
            elif token == "[db]":
                # end here
                break
            else:  # encountered valid symbol
                prev_symb = token
                next_symb = {
                    "[d]": ["[i]"],
                    "[i]": ["[s]"],
                    "[s]": ["[d]", "[i]", "[s]", "[db]"],
                }[token]
                i += 1
        elif (
            prev_symb == ""
        ):  # is a token but haven't encoutered [d] yet. invalid.
            i += 1
        else:  # is a token and have is under valid structure
            if prev_symb == "[d]":
                prev_domain = token
                prev_intent = ""
            elif prev_symb == "[i]":
                prev_intent = token
            elif prev_symb == "[s]":
                diaact_preds.append((prev_domain, prev_intent, token))
            i += 1
    return diaact_preds


def diaact_to_str(
    diaacts: Dict[str, List[List[str]]]
) -> Tuple[str, List[str]]:
    """_summary_

    Args:
        diaacts (Dict[str, List[List[str]]]): Dict of d-i -> [[s,v]]

    Returns:
        str: The context sequence.
        domains: The List of domains involve.

    """
    turn_str = ""
    prev_domain = ""
    domains = set()
    for di in diaacts.keys():
        domain, intent = di.split("-")
        domains.add(domain)
        if domain != prev_domain:
            turn_str += f"[d] {domain} "
            prev_domain = domain
        turn_str += f"[i] {intent} "
        for sv in diaacts[di]:
            turn_str += f"[s] {str(sv[0])} "
    turn_str += "[db] "
    return turn_str, sorted(list(domains))


def top_k_top_p_filtering(
    logits,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    filter_value=-float("Inf"),
    min_tokens_to_keep=1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    logits = logits / temperature
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = (
            logits < torch.topk(logits, top_k)[0][..., -1, None]
        )
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def watch(
    models: Dict[str, torch.nn.Module],
    log="gradients",
    log_freq=1000,
):
    log_parameters = False
    log_gradients = True
    if log == "all":
        log_parameters = True
    elif log == "parameters":
        log_parameters = True
        log_gradients = False
    elif log is None:
        log_gradients = False

    torch = wandb.util.get_module(
        "torch",
        required="wandb.watch only works with pytorch, couldn't import torch.",
    )

    for model in models.values():
        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                "Expected a pytorch model (torch.nn.Module). Received "
                + str(type(model))
            )

    for prefix, model in models.items():
        if not prefix.endswith("_"):
            prefix += "_"
        wandb.run.history.torch.add_log_hooks_to_pytorch_module(
            model,
            log_parameters=log_parameters,
            log_gradients=log_gradients,
            prefix=prefix,
            log_freq=log_freq,
        )
    return []


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


class RunningMeanStd:
    def __init__(self, epsilon=0, shape=()):
        self._mean = np.zeros(shape, "float64")
        self._var = np.ones(shape, "float64")
        self._std = np.ones(shape, "float64")
        self._count = epsilon
        self.shape = shape

    def update(self, x):
        """x must have shape (-1, self.shape[0], self.shape[1], etc)"""
        assert x.shape[1:] == self.shape, (x.shape, self.shape)
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
        self._std = self._var**0.5

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        (
            self._mean,
            self._var,
            self._count,
        ) = self._update_mean_var_count_from_moments(
            self._mean,
            self._var,
            self._count,
            batch_mean,
            batch_var,
            batch_count,
        )

    def mean(self):
        return self._mean.item()

    def std(self):
        return self._std.item()

    def var(self):
        return self._var.item()

    @staticmethod
    def _update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
    ):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return (new_mean, new_var, new_count)
