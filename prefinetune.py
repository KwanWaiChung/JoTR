import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import importlib
import wandb
import os
import json
import shutil
from einops import repeat
from typing import Dict, Iterable, Any, List, Tuple
from data.dataset import DialogActTurnDataset
from metric import DiaactF1
from util import (
    get_logger,
    set_seed,
    get_random_state,
    set_random_state,
    get_diaact_list,
    freeze_model,
)
from tqdm import tqdm
from model import DiaactTransformer
from torch.utils.data import DataLoader, random_split
from transformers import (
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    BertConfig,
    BertModel,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger(
    logger_level="debug",
    console_level="info",
)


def get_dataset(
    config: Dict[str, Any],
    mode,
    response_tokenizer=None,
    context_tokenizer=None,
):
    return DialogActTurnDataset(
        mode=mode,
        context_tokenizer=context_tokenizer,
        response_tokenizer=response_tokenizer,
        n_samples=config["n_samples"],
        usr_diaact_prefix=config["usr_diaact_prefix"],
        sys_diaact_prefix=config["sys_diaact_prefix"],
        belief_prefix=config["belief_prefix"],
        db_prefix=config["db_prefix"],
        decoder_prefix=config["decoder_prefix"],
        output_belief=config["output_belief"],
        output_belief_prefix=config["output_belief_prefix"],
        output_num_act=config["output_num_act"],
        output_num_act_prefix=config["output_num_act_prefix"],
        output_act_prefix=config["output_act_prefix"],
        max_resp_len=config["max_resp_len"],
        load_path=config["dataset_cache_load_path"],
        save_path=config["dataset_cache_save_path"],
        overwrite_cache=config["overwrite_dataset_cache"],
        character=config["character"],
        remove_belief_value=config.get("remove_belief_value", False),
        add_repeat_act_num=config.get("add_repeat_act_num", False),
    )


def get_context_model(config: Dict[str, Any], remove_dropout=False):
    model_config = BertConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["context_hidden_size"],
        num_hidden_layers=config["context_layer"],
        num_attention_heads=config["context_head"],
        intermediate_size=4 * config["context_hidden_size"],
        hidden_dropout_prob=0 if remove_dropout else 0.1,
        attention_probs_dropout_prob=0 if remove_dropout else 0.1,
    )
    model = BertModel(config=model_config, add_pooling_layer=True)
    return model


def get_response_model(
    config: Dict[str, Any],
    vocab_size: int,
    context_hidden_size: int,
    dropout_p: float = 0.1,
):
    model = DiaactTransformer(
        vocab_size=vocab_size,
        hidden_size=config["hidden_size"],
        context_hidden_size=context_hidden_size,
        n_heads=config["n_heads"],
        n_encoder_layers=config["n_encoder_layers"],
        n_decoder_layers=config["n_decoder_layers"],
        add_encoder_turn_embedding=config["add_encoder_turn_embedding"],
        add_encoder_type_embedding=config["add_encoder_type_embedding"],
        add_decoder_pos_embedding=config["add_decoder_pos_embedding"],
        add_decoder_type_embedding=config["add_decoder_type_embedding"],
        tie_weights=config["tie_weights"],
        max_turn=22,  # the maximum turn in dataset is 21
        dropout_p=dropout_p,
    )
    return model


def get_optim(
    config: Dict[str, Any],
    context_model: nn.Module,
    response_model: nn.Module,
    num_training_steps: int,
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in tuple(context_model.named_parameters())
                + tuple(response_model.named_parameters())
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in tuple(context_model.named_parameters())
                + tuple(response_model.named_parameters())
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config["learning_rate"],
        eps=config["adam_epsilon"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_step_ratio"] * num_training_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, help="The random seed", default=2048
    )
    parser.add_argument(
        "--save_prefix", help="The prefix of the save folder", type=str
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Directory path for checkpoints.",
        default="saved/",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--project_name", type=str, help="Name for wandb")
    parser.add_argument("--random_params", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_val", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        type=str,
        help="Location of the checkpoint for evaluation.",
    )

    # model args
    parser.add_argument(
        "--context_hidden_size",
        type=int,
    )
    parser.add_argument(
        "--context_layer",
        type=int,
    )
    parser.add_argument(
        "--context_head",
        type=int,
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        help="The hidden size for the generation transformer.",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        help="The number of heads for the generation transformer.",
    )
    parser.add_argument(
        "--n_encoder_layers",
        type=int,
        help="The number of layers for the generation transformer's encoder.",
    )
    parser.add_argument(
        "--n_decoder_layers",
        type=int,
        help="The number of layers for the generation transformer's decoder.",
    )
    parser.add_argument("--add_encoder_turn_embedding", action="store_true")
    parser.add_argument("--add_encoder_type_embedding", action="store_true")
    parser.add_argument("--add_decoder_pos_embedding", action="store_true")
    parser.add_argument("--add_decoder_type_embedding", action="store_true")
    parser.add_argument("--tie_weights", action="store_true")
    parser.add_argument("--freeze_context_model", action="store_true")
    parser.add_argument("--constrain_output", action="store_true")

    # data args
    parser.add_argument("--max_resp_len", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=-1)
    parser.add_argument("--usr_diaact_prefix", type=str, default="[usr_act]")
    parser.add_argument("--sys_diaact_prefix", type=str, default="[sys_act]")
    parser.add_argument("--belief_prefix", type=str, default="[belief_state]")
    parser.add_argument("--db_prefix", type=str, default="[db]")
    parser.add_argument("--decoder_prefix", type=str, default="[start]")
    parser.add_argument("--output_belief", action="store_true")
    parser.add_argument(
        "--output_belief_prefix", type=str, default="[belief_change]"
    )
    parser.add_argument("--output_num_act", action="store_true")
    parser.add_argument(
        "--output_num_act_prefix", type=str, default="[num_act]"
    )
    parser.add_argument("--output_act_prefix", type=str, default="[act]")
    parser.add_argument("--end_token", type=str, default="[end]")
    parser.add_argument("--dataset_cache_load_path", type=str)
    parser.add_argument("--dataset_cache_save_path", type=str)
    parser.add_argument("--overwrite_dataset_cache", action="store_true")
    parser.add_argument("--character", type=str, default="all")
    parser.add_argument("--remove_belief_value", action="store_true")
    parser.add_argument("--add_repeat_act_num", action="store_true")

    # optim args, scheduler args
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-5)
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.",
    )
    parser.add_argument(
        "--warmup_step_ratio",
        default=0,
        type=float,
        help="Portion of steps to do warmup. 10% is a good start",
    )
    parser.add_argument(
        "--lr_decay",
        action="store_true",
        help="If true, it will linear decay the lr to 0 at the end.",
    )

    # train args
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--epoch_per_val",
        type=int,
        help="The number of training epochs to perform one validation.",
        default=1,
    )
    parser.add_argument(
        "--step_per_log",
        type=int,
        help="The number of training steps to log batch level statistics.",
        default=1,
    )
    parser.add_argument(
        "--patience",
        type=int,
        help=(
            "Number of epochs with no improvement after which training will be"
            " stopped"
        ),
        default=1e17,
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        help="Number of training epochs",
        default=5,
    )
    parser.add_argument("--validate_before_train", action="store_true")
    parser.add_argument("--overfit_test", action="store_true")

    args = parser.parse_args()
    if args.add_decoder_type_embedding:
        if args.output_belief or args.output_num_act:
            raise ValueError(
                "Currently only support adding decoder type embedding without"
                " output_belief and output_num_act."
            )
        if args.output_act_prefix:
            raise ValueError("Currently only support no output_act_prefix.")

    return args


class Trainer:
    def __init__(
        self,
        context_model: nn.Module,
        response_model: nn.Module,
        context_tokenizer: PreTrainedTokenizer,
        response_tokenizer: PreTrainedTokenizer,
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler = None,
    ):
        """
        Optimizer can be None if constructed only for testing.
        """
        self.context_model = context_model.to(DEVICE)
        self.response_model = response_model.to(DEVICE)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.context_tokenizer = context_tokenizer
        self.response_tokenizer = response_tokenizer
        self.f1 = DiaactF1()
        self.domain_index = response_tokenizer.encode(
            " ".join(DialogActTurnDataset.domains + [config["end_token"]])
        )
        self.intent_index = response_tokenizer.encode(
            " ".join(DialogActTurnDataset.intents)
        )
        self.slot_index = response_tokenizer.encode(
            " ".join(DialogActTurnDataset.slots)
        )
        if config["add_decoder_type_embedding"]:
            assert not config[
                "output_act_prefix"
            ], "act_prefix should be empty for using decoder_type_embedding."
            self.start_token = config["decoder_prefix"]
        else:
            assert config["output_act_prefix"], "Parsing requires act_prefix"
            self.start_token = config["output_act_prefix"]
        self.start_index = response_tokenizer.convert_tokens_to_ids(
            self.start_token
        )

    def _reset_metrics(self):
        self.f1.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor], batch_i: int
    ) -> Dict[str, Any]:
        """Define a update step of the model.

        Args:
            batch: Input for the model.
            batch_i: Current batch idx of this epoch.

        Returns:
            Dict with book keeping variables:

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
        ) = batch

        self.optimizer.zero_grad()
        # shape (B, E)
        last_usr_diaact_embeds = self.context_model(
            input_ids=last_usr_diaact_ids.to(DEVICE),
            attention_mask=last_usr_diaact_attn_mask.to(DEVICE),
        )[1]
        # shape (B, E)
        last_sys_diaact_embeds = self.context_model(
            input_ids=last_sys_diaact_ids.to(DEVICE),
            attention_mask=last_sys_diaact_attn_mask.to(DEVICE),
        )[1]
        # shape (B, E)
        belief_embeds = self.context_model(
            input_ids=belief_ids.to(DEVICE),
            attention_mask=belief_attn_mask.to(DEVICE),
        )[1]
        # shape (B, E)
        db_embeds = self.context_model(
            input_ids=db_ids.to(DEVICE),
            attention_mask=db_attn_mask.to(DEVICE),
        )[1]
        outputs: Dict[str, torch.Tensor] = self.response_model(
            tgt_ids=output_ids.to(DEVICE),
            tgt_key_padding_mask=output_mask.to(DEVICE),
            last_usr_diaact_embeds=last_usr_diaact_embeds,
            last_sys_diaact_embeds=last_sys_diaact_embeds,
            belief_embeds=belief_embeds,
            db_embeds=db_embeds,
            turns=turns.to(DEVICE)
            if config["add_encoder_turn_embedding"]
            else None,
        )
        loss = outputs["loss"]
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return {
            "loss": loss.item(),
            "batch_i": batch_i,
        }

    def _parse_diaacts(
        self, input_ids: torch.Tensor
    ) -> List[Tuple[str, str, str]]:
        """

        Args:
            input_ids (torch.Tensor): The context which probably ends with
                [te]. shape: (T, ).

        Returns:
            List of (domain, intent, slot).

        """
        input_str: str = self.response_tokenizer.decode(input_ids)
        diaacts = DialogActTurnDataset.str_to_diaact(
            input_str,
            start_token=self.start_token,
            end_token=config["end_token"],
        )
        return diaacts

    def validating_step(
        self, batch: Tuple[torch.Tensor], batch_i: int
    ) -> Dict[str, Any]:
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
        ) = batch
        output_ids = output_ids.to(DEVICE)
        # shape (B, E)
        last_usr_diaact_embeds = self.context_model(
            input_ids=last_usr_diaact_ids.to(DEVICE),
            attention_mask=last_usr_diaact_attn_mask.to(DEVICE),
        )[1]
        # shape (B, E)
        last_sys_diaact_embeds = self.context_model(
            input_ids=last_sys_diaact_ids.to(DEVICE),
            attention_mask=last_sys_diaact_attn_mask.to(DEVICE),
        )[1]
        # shape (B, E)
        belief_embeds = self.context_model(
            input_ids=belief_ids.to(DEVICE),
            attention_mask=belief_attn_mask.to(DEVICE),
        )[1]
        # shape (B, E)
        db_embeds = self.context_model(
            input_ids=db_ids.to(DEVICE),
            attention_mask=db_attn_mask.to(DEVICE),
        )[1]
        outputs: Dict[str, torch.Tensor] = self.response_model(
            tgt_ids=output_ids,
            tgt_key_padding_mask=output_mask.to(DEVICE),
            last_usr_diaact_embeds=last_usr_diaact_embeds,
            last_sys_diaact_embeds=last_sys_diaact_embeds,
            belief_embeds=belief_embeds,
            db_embeds=db_embeds,
            turns=turns.to(DEVICE)
            if config["add_encoder_turn_embedding"]
            else None,
        )
        loss = outputs["loss"]
        # shape (B, *, V)
        logits = outputs["logits"]
        # shape (B, *)
        output_ids = torch.cat(
            [output_ids[:, :1], logits.max(dim=2)[1]], dim=1
        )
        for diaact, _output_ids in zip(diaacts, output_ids):
            diaacts_pred: List[Tuple[str, str, str]] = self._parse_diaacts(
                _output_ids
            )
            f1 = self.f1(
                diaacts_pred,
                get_diaact_list(diaact[-1]),
            )
        return {
            "loss": loss.item(),
            "batch_i": batch_i,
            "f1": f1,
        }

    def testing_step(
        self, batch: Tuple[torch.Tensor], batch_i: int
    ) -> Dict[str, Any]:
        write_outputs = []
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
            gt_output_ids,
            output_mask,
            turns,
            diaacts,
            diaact_strs,
            belief_strs,
            change_belief_strs,
            db_strs,
        ) = batch
        bz = len(session_ids)
        # shape (B, E)
        last_usr_diaact_embeds = self.context_model(
            input_ids=last_usr_diaact_ids.to(DEVICE),
            attention_mask=last_usr_diaact_attn_mask.to(DEVICE),
        )[1]
        # shape (B, E)
        last_sys_diaact_embeds = self.context_model(
            input_ids=last_sys_diaact_ids.to(DEVICE),
            attention_mask=last_sys_diaact_attn_mask.to(DEVICE),
        )[1]
        # shape (B, E)
        belief_embeds = self.context_model(
            input_ids=belief_ids.to(DEVICE),
            attention_mask=belief_attn_mask.to(DEVICE),
        )[1]
        # shape (B, E)
        db_embeds = self.context_model(
            input_ids=db_ids.to(DEVICE),
            attention_mask=db_attn_mask.to(DEVICE),
        )[1]
        # shape (B, 1)
        output_ids = repeat(
            torch.tensor(
                self.response_tokenizer.encode(config["decoder_prefix"]),
                dtype=torch.long,
                device=DEVICE,
            ),
            "T -> B T",
            B=bz,
        )
        # shape (B, *)
        output_ids: torch.Tensor = self.response_model.generate(
            tgt_ids=output_ids,
            last_usr_diaact_embeds=last_usr_diaact_embeds,
            last_sys_diaact_embeds=last_sys_diaact_embeds,
            belief_embeds=belief_embeds,
            db_embeds=db_embeds,
            turns=turns.to(DEVICE)
            if config["add_encoder_turn_embedding"]
            else None,
            eos_token_id=self.response_tokenizer.convert_tokens_to_ids(
                config["end_token"]
            ),
            do_sample=False,
            domain_idx=self.domain_index
            if config["constrain_output"]
            else None,
            intent_idx=self.intent_index
            if config["constrain_output"]
            else None,
            slot_idx=self.slot_index if config["constrain_output"] else None,
            start_idx=self.start_index if config["constrain_output"] else None,
        )

        for i, (diaact, _output_ids) in enumerate(zip(diaacts, output_ids)):
            diaacts_pred: List[Tuple[str, str, str]] = self._parse_diaacts(
                _output_ids
            )
            diaacts_gt = get_diaact_list(diaact[-1])
            f1 = self.f1(
                diaacts_pred,
                diaacts_gt,
            )
            write_outputs.append(
                {
                    "session_id": session_ids[i],
                    "last_usr_diaact": get_diaact_list(diaact[-2])
                    if len(diaact) >= 2
                    else [],
                    "last_sys_diaact": get_diaact_list(diaact[-3])
                    if len(diaact) >= 3
                    else [],
                    "belief_state_str": belief_strs[i][-1],
                    "db_str": db_strs[i][-1],
                    "predicted_str": self.response_tokenizer.decode(
                        _output_ids
                    ),
                    "predicted_act": diaacts_pred,
                    "gt_act": diaacts_gt,
                    "f1": f1,
                    "turn": turns[i].item(),
                }
            )
        return write_outputs

    def test(self, test_dl: Iterable):
        bar = tqdm(test_dl, desc="Testing")
        self.context_model.eval()
        self.response_model.eval()
        self._reset_metrics()
        write_outputs = []
        for i, batch_dict in enumerate(bar):
            with torch.no_grad():
                write_outputs += self.testing_step(batch_dict, i)
            f1 = self.f1.compute()
            bar.set_postfix({"test/dia_f1": f"{f1:.3f}"})
        return f1, write_outputs

    def validate(self, val_dl: Iterable):
        bar = tqdm(val_dl, desc="Validating")
        self.context_model.eval()
        self.response_model.eval()
        self._reset_metrics()
        val_outputs = []
        for i, batch_dict in enumerate(bar):
            with torch.no_grad():
                output = self.validating_step(batch_dict, i)
            val_outputs.append(output)
            avg_loss = np.mean([o["loss"] for o in val_outputs])
            dia_f1 = self.f1.compute()
            bar.set_postfix(
                {"val/loss": f"{avg_loss:.3f}", "val/dia_f1": f"{dia_f1:.3f}"}
            )
        return val_outputs

    def fit(self, train_dl: Iterable, val_dl: Iterable = None):
        best_score = 100
        best_dia_f1 = 0
        best_path = None
        wait = 0
        global_step = 0
        global_outputs = []
        if config["validate_before_train"] and val_dl is not None:
            states = get_random_state()
            self.validate(val_dl)
            set_random_state(states)
        for epoch_i in range(1, config["n_epochs"] + 1):
            bar = tqdm(
                train_dl, desc=f"Train epoch {epoch_i}/{config['n_epochs']}"
            )
            self.context_model.train()
            self.response_model.train()
            train_outputs = []
            self._reset_metrics()
            for i, batch_dict in enumerate(bar):
                output = self.training_step(batch_dict, i)
                global_step += 1
                train_outputs.append(output)
                avg_loss = np.mean([o["loss"] for o in train_outputs])
                bar.set_postfix({"train/loss": f"{avg_loss:.3f}"})
                # train batch statistics
                if global_step % config["step_per_log"] == 0:
                    wandb_log = {
                        "train/loss": output["loss"],
                        "trainer/global_step": global_step,
                        "trainer/learning_rate": self.optimizer.param_groups[
                            0
                        ]["lr"],
                    }
                    if config["use_wandb"]:
                        wandb.log(wandb_log)
                    wandb_log["epoch"] = epoch_i
                    global_outputs.append(wandb_log)

            # train epoch statistics
            if config["use_wandb"]:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "epoch": epoch_i,
                        "trainer/learning_rate": self.optimizer.param_groups[
                            0
                        ]["lr"],
                    }
                )

            if val_dl is not None and epoch_i % config["epoch_per_val"] == 0:
                val_outputs = self.validate(val_dl)
                avg_loss = np.mean([o["loss"] for o in val_outputs])
                val_dia_f1 = self.f1.compute()
                wandb_log = {
                    "val/loss": avg_loss,
                    "val/dia_f1": val_dia_f1,
                    "epoch": epoch_i,
                    "trainer/global_step": global_step,
                }
                global_outputs.append(wandb_log)
                if config["use_wandb"]:
                    wandb.log(wandb_log)
                if avg_loss < best_score:
                    logger.info(
                        f"val_loss improved by {(best_score - avg_loss):.3f}"
                    )
                    best_score = avg_loss
                    best_dia_f1 = val_dia_f1
                    wait = 0
                    if config["save_prefix"] and config["save_path"]:
                        if best_path is not None:
                            if os.path.exists(best_path):
                                shutil.rmtree(best_path)
                                logger.info(f"Removed old save {best_path}.")
                        best_path = os.path.join(
                            config["save_path"],
                            f"{config['save_prefix']}"
                            f"-epoch={epoch_i:02d}"
                            f"-val_loss={best_score:.3f}"
                            f"-val_dia_f1={best_dia_f1:.3f}",
                        )
                        os.makedirs(best_path, exist_ok=True)
                        torch.save(
                            {
                                "context_model": self.context_model.state_dict(),
                                "response_model": self.response_model.state_dict(),
                                "context_tokenizer": self.context_tokenizer,
                                "response_tokenizer": self.response_tokenizer,
                            },
                            os.path.join(best_path, "ckpt.pth"),
                        )
                        json.dump(
                            config,
                            open(
                                os.path.join(best_path, "config.json"),
                                "w",
                            ),
                        )
                        pd.DataFrame(global_outputs).to_csv(
                            os.path.join(best_path, "outputs.csv"), index=False
                        )
                        logger.info(
                            f"Saved best with best score: {best_score:.2f}"
                        )

                elif wait >= config["patience"]:
                    logger.info("Out of patience. Early stop now.")
                    # maybe save sth
                    break
                else:
                    wait += 1
                    logger.debug(
                        f"Not improved on epoch {epoch_i}, wait={wait}",
                    )
        # End training
        config["checkpoint"] = best_path
        if config["use_wandb"]:
            wandb.run.summary["best_path"] = best_path
            wandb.run.summary["best_val_loss"] = best_score
            wandb.run.summary["best_val_dia_f1"] = best_dia_f1

    def overfit_test(self, train_dl):
        self.context_model.eval()
        self.response_model.eval()
        batch_dict = next(iter(train_dl))
        i = 0
        x = []
        y = []
        while True:
            i += 1
            output = self.training_step(batch_dict, i)
            if i % 5 == 0:
                x.append(i)
                y.append(output["loss"])
                print(f"{i}: {output}")
            if i % 50 == 0:
                self.f1.reset()
                output = self.testing_step(batch_dict, i)
                print(self.f1.compute())
                self.f1.reset()
                output = self.validating_step(batch_dict, i)
                print(self.f1.compute())
                pass
            #      plt.plot(x, y)
            #      plt.xlabel("train iter")
            #      plt.ylabel("loss")
            #      plt.pause(0.05)


def train(config):
    # create dataset and dataloader object
    ds = get_dataset(config, mode="train")
    response_tokenizer = ds.build_response_tokenizer()
    context_tokenizer = ds.build_context_tokenizer()
    ds.response_tokenizer = response_tokenizer
    ds.context_tokenizer = context_tokenizer
    config["vocab_size"] = len(context_tokenizer)
    context_model = get_context_model(config)
    response_model = get_response_model(
        config,
        vocab_size=len(response_tokenizer),
        context_hidden_size=context_model.config.hidden_size,
    )
    n_train = int(0.7 * len(ds))
    train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train])
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=config["batch_size"],
        num_workers=4,
        collate_fn=ds.collate_fn,
        shuffle=True,
    )
    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=50,
        num_workers=4,
        collate_fn=ds.collate_fn,
        shuffle=False,
    )

    optimizer, scheduler = get_optim(
        config=config,
        context_model=context_model,
        response_model=response_model,
        num_training_steps=len(train_dl) * config["n_epochs"],
    )
    if not config["lr_decay"]:
        scheduler = None

    trainer = Trainer(
        context_model=context_model,
        response_model=response_model,
        context_tokenizer=context_tokenizer,
        response_tokenizer=response_tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if config["overfit_test"]:
        trainer.overfit_test(train_dl=train_dl)
    if config["use_wandb"]:
        if not config["project_name"]:
            raise ValueError("Must specify `project_name` if using wandb.")
        wandb.init(
            project=config["project_name"],
            name=config["save_prefix"],
            config=config,
        )
        wandb.watch(context_model, log="all")
        wandb.watch(response_model, log="all")
    trainer.fit(train_dl=train_dl, val_dl=val_dl)

    # test
    config["n_samples"] = -1
    test_ds = get_dataset(
        config,
        context_tokenizer=context_tokenizer,
        response_tokenizer=response_tokenizer,
        mode="test",
    )
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=config["batch_size"],
        num_workers=4,
        collate_fn=test_ds.collate_fn,
        shuffle=False,
    )
    dia_f1, write_outputs = trainer.test(test_dl=test_dl)
    if config.get("checkpoint"):
        filepath = os.path.join(config["checkpoint"], "test_outputs.json")
        with open(filepath, "w") as f:
            json.dump(write_outputs, f, indent=2)
            logger.info(f"test outputs saved at {filepath}.")

    if config["use_wandb"]:
        wandb.run.summary["dia_f1"] = dia_f1


def validate(config):
    # TODO: modify the whole code
    config_saved = json.load(
        open(os.path.join(config["checkpoint"], "config.json"), "r")
    )
    torch_saved = torch.load(os.path.join(config["checkpoint"], "ckpt.pth"))
    tokenizer = get_tokenizer(config_saved)
    model = get_model(config_saved, tokenizer)
    model.load_state_dict(torch_saved["model"])

    raise NotImplementedError()
    val_ds = get_dataset(config=config_saved, tokenizer=tokenizer, mode="test")
    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=config["batch_size"],
        num_workers=4,
        collate_fn=val_ds.collate_fn,
        shuffle=False,
    )
    trainer = Trainer(model=model, tokenizer=tokenizer)
    outputs = trainer.validate(val_dl=val_dl)
    # do some processing to outputs if needed
    print(outputs)


def test(config):
    config_saved = json.load(
        open(os.path.join(config["checkpoint"], "config.json"), "r")
    )
    torch_saved = torch.load(
        os.path.join(config["checkpoint"], "ckpt.pth"), map_location=DEVICE
    )
    context_model, _ = get_context_model(config_saved)
    context_model.load_state_dict(torch_saved["context_model"])
    context_tokenizer = torch_saved["context_tokenizer"]
    response_tokenizer = torch_saved["response_tokenizer"]
    config_saved["n_samples"] = -1
    config["add_encoder_turn_embedding"] = config_saved[
        "add_encoder_turn_embedding"
    ]
    config["decoder_prefix"] = config_saved["decoder_prefix"]
    config["end_token"] = config_saved["end_token"]
    config_saved["dataset_cache_load_path"] = config["dataset_cache_load_path"]
    test_ds = get_dataset(
        config_saved,
        context_tokenizer=context_tokenizer,
        response_tokenizer=response_tokenizer,
        mode="test",
    )

    response_model = get_response_model(
        config_saved,
        vocab_size=len(response_tokenizer),
        context_hidden_size=context_model.config.hidden_size,
    )
    response_model.load_state_dict(torch_saved["response_model"])
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=config["batch_size"],
        num_workers=4,
        collate_fn=test_ds.collate_fn,
        shuffle=False,
    )
    trainer = Trainer(
        context_model=context_model,
        response_model=response_model,
        context_tokenizer=context_tokenizer,
        response_tokenizer=response_tokenizer,
    )
    f1, write_outputs = trainer.test(test_dl=test_dl)
    # do some processing to outputs if needed
    filepath = os.path.join(config["checkpoint"], "test_outputs.json")
    with open(filepath, "w") as f:
        json.dump(write_outputs, f, indent=2)
        logger.info(f"test outputs saved at {filepath}.")

    print(f1)
    return f1


def get_random_params(config):
    seeder = random.Random()
    config["batch_size"] = seeder.choice([8, 16, 32])
    config["hidden_size"] = seeder.choice([128, 256])
    config["n_encoder_layers"] = seeder.randrange(1, 3)
    config["n_decoder_layers"] = seeder.randrange(1, 3)
    config["n_heads"] = 1
    config["learning_rate"] = seeder.choice([3e-4, 1e-4])
    logger.info(
        f"Random sampled parameters: batch_size={config['batch_size']},"
        f" hidden_size={config['hidden_size']},"
        f" n_encoder_layers={config['n_encoder_layers']},"
        f" n_decoder_layers={config['n_decoder_layers']},"
        f" n_heads={config['n_heads']},"
        f" learning_rate={config['learning_rate']}."
    )


if __name__ == "__main__":
    config = vars(get_args())
    set_seed(config["seed"])
    if importlib.util.find_spec("wandb") is None:
        config["use_wandb"] = False

    if config["random_params"]:
        get_random_params(config)
    if config["do_train"]:
        train(config)
    elif config["do_val"]:
        validate(config)
    elif config["do_test"]:
        test(config)
