"""
    The file contains the PPO training code for prefinetune2
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg

    Code adopted from https://github.com/ericyangyu/PPO-for-Beginners/blob/master/ppo.py

"""

import sys
import os
import random
import json
import shutil

import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import importlib
from einops import rearrange
from time import time
from tqdm import tqdm
from torch.distributions import Categorical
from pprint import pprint
from typing import Dict, Any, Tuple, List, Union
from model import GPT2MultiHead, TransformerSingleHead, MLP, CriticTransformer
from util import (
    get_logger,
    set_seed,
    get_seeder,
    get_random_state,
    set_random_state,
    lineplot,
    RunningMeanStd,
    diaact_to_str,
    str_to_ids,
    freeze_model,
)
from convlab.model.policy.usr_sgd_rule import UserPolicyAgendaSGD
from sgd_prefinetune import get_context_model, get_response_model

# from prefinetune2_scratch import get_context_model  # Use this for prefinetun2_scratch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, GPT2Model
from convlab.model.state_encoder.dagpt_state_encoder import SGDStateEncoder
from convlab.model.action_encoder.dagpt_action_encoder import (
    SGDActionEncoder,
)
from convlab.dialog_agent.agent import PipelineAgent
from convlab.model.dst.sgd_rule_dst import RuleDST
from convlab.model.policy.multiwoz_rule import RulePolicy
from convlab.env.multiwoz_evaluator import MultiwozEvaluator
from convlab.env.env import Environment
from env import (
    AggressiveRewardEnv,
    ConservativeRewardEnv,
    ActionNumEnvironment,
)
from convlab3.policy.genTUS.stepGenTUS import UserPolicy
from torch.nn.utils.rnn import pad_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORWARD_BATCH_SIZE = 8
DB_DIM = 24
SGD_CKPT = "convlab3/policy/genTUS/unify/experiments/sgd_0_1/22-12-19-19-58"


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
        "--prefinetuned_path",
        type=str,
        help="This holds the path to the model trained by me.",
    )
    parser.add_argument("--remove_dropout", action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        help=(
            "The gpt model name. Even if not using a pretrained model, still"
            " need to provide one to tokenize."
        ),
        default="gpt2",
    )
    parser.add_argument("--use_critic_transformer", action="store_true")
    parser.add_argument("--constrain_output", action="store_true")
    parser.add_argument("--use_gentus", action="store_true")

    # data args
    parser.add_argument("--max_seq_len", type=int, default=1024)

    # rl env args
    parser.add_argument("--max_turn", type=int, default=20)
    parser.add_argument("--train_env_seed", type=int, default=2049)
    parser.add_argument("--test_env_seed", type=int, default=2050)
    parser.add_argument("--rew_type", type=str, default="normal")

    # optim args, scheduler args
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--actor_learning_rate", "-actor_lr", type=float, default=5e-5
    )
    parser.add_argument(
        "--critic_learning_rate", "-critic_lr", type=float, default=5e-5
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.",
    )
    parser.add_argument("--grad_clip", type=float, default=5.0)

    # rl train args
    parser.add_argument(
        "--minibatch_per_epoch",
        type=int,
        default=4,
        help="How many minibatches for each update epoch.",
    )
    parser.add_argument(
        "--epoch_per_update",
        type=int,
        default=5,
        help="Total number of updates per sample.",
    )
    parser.add_argument(
        "--step_per_collect",
        type=int,
        default=400,
        help="Number of env steps before one update.",
    )
    parser.add_argument(
        "--n_train_steps",
        type=int,
        default=800000,
        help="The total number of training steps.",
    )
    parser.add_argument(
        "--n_warmup_steps",
        type=int,
        default=0,
        help="Number of steps to warm up.",
    )
    parser.add_argument(
        "--update_per_log",
        type=int,
        default=4,
        help="Number of update steps before one log.",
    )
    parser.add_argument(
        "--update_per_test",
        type=int,
        default=50,
        help="Number of update steps before one test.",
    )
    parser.add_argument(
        "--episode_per_test",
        type=int,
        default=100,
        help="Number of episodes to test.",
    )
    parser.add_argument(
        "--norm_adv", action="store_true", help="Normalize the advantage."
    )
    parser.add_argument(
        "--norm_value",
        action="store_true",
        help="Normalize the value function target.",
    )
    parser.add_argument(
        "--vf_coef", type=float, help="The critic coefficient", default=0.5
    )
    parser.add_argument(
        "--et_coef", type=float, help="The entropy coefficient", default=0.01
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="The gamma in gae calculation.",
        default="0.99",
    )
    parser.add_argument(
        "--lamb",
        type=float,
        help="The lambda in gae calculation.",
        default="0.95",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        help="The clip range for ppo loss.",
        default=0.2,
    )
    parser.add_argument(
        "--critic_hidden_sizes",
        nargs="+",
        help="The hidden size of critic net.",
        type=int,
        default=[],
    )
    parser.add_argument("--recompute_adv", action="store_true")
    parser.add_argument("--test_before_train", action="store_true")
    parser.add_argument("--random_params", action="store_true")

    args = parser.parse_args()
    assert args.n_warmup_steps == 0, "warmup is currently not supported. "
    return args


class PPOTrainer:
    def __init__(
        self,
        context_model: nn.Module,
        response_model: nn.Module,
        critic: nn.Module,
        context_tokenizer,
        response_tokenizer,
        train_env,
        test_env,
        actor_lr: float,
        critic_lr: float,
        epoch_per_update: int,
        minibatch_per_epoch: int,
        step_per_collect: int,
        update_per_test: int,
        episode_per_test: int,
        update_per_log: int,
        norm_adv: bool,
        norm_value: bool,
        gamma: float,
        lamb: float,
        clip_range: float,
        vf_coef: float,
        et_coef: float,
    ):
        """
        Initializes the PPO model, including hyperparameters.

        Args:
            context_model (torch.Module)
            response_model (torch.Module)
            critic (torch.Module)
            context_tokenizer
            response_tokenizer
            train_env: The train environment.
            test_env: The train environment.
            actor_lr(float): Learning rate for the policy, default: 5e-5
            critic_lr(float): Learning rate for the value network,
                default: 1e-4.
            epoch_per_update (int): The number of repeat time for policy
                learning, for example, set it to 2 means the policy needs to
                learn each given batch data twice.
            step_per_collect (int): The number of transitions the collector
                would collect before the network update, i.e., trainer will
                collect "step_per_collect" transitions and do some policy
                network update repeatedly in each epoch.
            update_per_test (int): Frequnecy to test the policy. Should be multiple
                of epoch_per_update.
            episode_per_test (int): Number of episodes to test.
            update_per_log (int): Number of updates before one log.
            gamma: Gamma parameter for advantage calculation, default: 1.
            lamb: Lambda parameter for advantage calcualation, default: 0.95
            dis_thres (float): If disease prob is larger than this value, it
                will predict disease and early terminate.
            vf_coef: Scaling factor for value loss, default: 0.1
            et_coef: Scaling factor for entropy loss, default: 0.01.

        """
        self.context_model = context_model
        self.response_model = response_model
        self.context_tokenizer = context_tokenizer
        self.response_tokenizer = response_tokenizer
        self.critic = critic
        self.actor_optim = optim.Adam(
            list(context_model.parameters())
            + list(response_model.parameters()),
            lr=actor_lr,
        )
        if critic is not None:
            self.critic_optim = optim.Adam(
                self.critic.parameters(), lr=critic_lr
            )
        self.epoch_per_update = epoch_per_update
        self.step_per_collect = step_per_collect
        self.minibatch_size = step_per_collect // minibatch_per_epoch
        config["minibatch_size"] = self.minibatch_size
        logger.info(f"The minibatch size is around {self.minibatch_size}.")
        self.update_per_test = update_per_test
        self.episode_per_test = episode_per_test
        self.last_log_step = 0
        self.update_per_log = update_per_log
        self.norm_adv = norm_adv
        self.norm_value = norm_value
        self.gamma = gamma
        self.lamb = lamb
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.et_coef = et_coef
        self.saves = []
        self.ret_stat = RunningMeanStd()

        self.train_env = train_env
        self.test_env = test_env

        self.start_token = (
            config["output_act_prefix"] or config["decoder_prefix"]
        )
        self.start_index = response_tokenizer.convert_tokens_to_ids(
            self.start_token
        )

    def test(self):
        (
            batch_usr_embeds,
            batch_sys_embeds,
            batch_belief_embeds,
            batch_db_embeds,
            batch_turns,
            batch_response_ids,
            batch_response_mask,
            batch_log_probs,
            batch_values,
            batch_reward,
            batch_done,
            ep_len,
            ep_rew,
            ep_info,
            ep_history,
            ep_goal,
        ) = self.rollout(
            episode_per_collect=config["episode_per_test"], is_train=False
        )  # ALG STEP 3
        avg_succ = np.mean([i["success"] for i in ep_info])
        avg_len = ep_len.mean().item()
        avg_rew = ep_rew.mean().item()
        log_dict = {
            "trainer/env_step": self.t_so_far,
            "trainer/global_step": self.u_so_far,
            "test/avg_len": avg_len,
            "test/avg_rew": avg_rew,
            "test/avg_succ": avg_succ,
        }
        self.global_outputs.append(log_dict)
        self.prev_avg_succ = avg_succ.item()
        if config["use_wandb"]:
            wandb.log(log_dict)
        if avg_succ > self.best_avg_succ:
            logger.info(
                "Test success rate increase by"
                f" {avg_succ-self.best_avg_succ:.3f}"
            )
            self.best_avg_succ = avg_succ
            self.best_avg_len = avg_len
            self.best_avg_rew = avg_rew
            # save
            if config["save_prefix"] is not None:
                if self.best_path is not None:
                    if os.path.exists(self.best_path):
                        shutil.rmtree(self.best_path)
                        logger.info(f"Removed old save {self.best_path}.")
                self.best_path = os.path.join(
                    config["save_path"],
                    f"{config['save_prefix']}"
                    f"-global_step={self.u_so_far}"
                    f"-env_step={self.t_so_far}"
                    f"-succ={avg_succ:.3f}"
                    f"-rew={avg_rew:.3f}"
                    f"-len={avg_len:.3f}",
                )
                os.makedirs(self.best_path, exist_ok=True)
                torch.save(
                    {
                        "response_model": self.response_model.state_dict(),
                        "context_model": self.context_model.state_dict(),
                        "critic": self.critic.state_dict(),
                        "response_tokenizer": self.response_tokenizer,
                        "context_tokenizer": self.context_tokenizer,
                    },
                    os.path.join(self.best_path, "ckpt.pth"),
                )
                json.dump(
                    config,
                    open(
                        os.path.join(self.best_path, "config.json"),
                        "w",
                    ),
                )
                extract_rollout_output(
                    ep_history,
                    ep_info,
                    ep_rew,
                    ep_goal,
                    ep_len,
                    os.path.join(self.best_path, "test_outputs.txt"),
                )

                logger.info(
                    "Saved best with best success rate: "
                    f"{avg_succ:.2f} "
                    f"in {self.best_path}."
                )
                if config["use_wandb"]:
                    wandb.run.summary["best_path"] = self.best_path
                    wandb.run.summary["best_avg_succ"] = self.best_avg_succ
                    wandb.run.summary["best_avg_rew"] = self.best_avg_rew
                    wandb.run.summary["best_avg_len"] = self.best_avg_len
                    config["checkpoint"] = self.best_path

    def update(
        self,
        batch_usr_embeds,
        batch_sys_embeds,
        batch_belief_embeds,
        batch_db_embeds,
        batch_turns,
        batch_response_ids,
        batch_response_mask,
        batch_log_probs,
        batch_values,
        batch_reward,
        batch_done,
    ):
        """
        Args:
            batch_usr_embeds:  The last usr diaact embeds
                shape: (n_sample, E).
            batch_sys_embeds: The last sys diaact embeds.
                shape: (n_sample, E).
            batch_belief_embeds: The belief embeds.
                shape: (n_sample, E).
            batch_db_embeds: The db embeds.
                shape: (n_sample, E).
            batch_turns: The turn numbers.
                shape: (n_samples).
            batch_response_ids: The response token ids.
                shape: (n_samples, T)
            batch_response_mask: The response mask.
                shape: (n_samples, T)
            batch_log_probs: The log probs of the responses, which are the sum
                of log probs of all the response tokens.
                Shape: (n_samples)
            batch_values: The critic values
                Shape: (n_samples)
            batch_reward: The rewards.
                Shape: (n_samples)
            batch_done: .
                Shape: (n_samples)

        """
        # Calculate advantage at k-th iteration
        v = batch_values
        old_log_probs = batch_log_probs
        if self.norm_value:
            # denormalize
            v = v * self.ret_stat.std() + self.ret_stat.mean()
        adv = self.compute_gae(
            batch_reward,
            v,
            batch_done,
        )
        rets = adv + v.to(DEVICE)
        if self.norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)
        if self.norm_value:
            self.ret_stat.update(rets)
            rets = (rets - self.ret_stat.mean()) / (
                self.ret_stat.std() + 1e-10
            )

        # This is the loop where we update our network for some n epochs
        update_steps = self.epoch_per_update * config["minibatch_per_epoch"]
        bar = tqdm(
            desc="Updating",
            total=update_steps,
            position=1,
            leave=False,
        )
        batch_size = batch_turns.shape[0]
        for _ in range(self.epoch_per_update):  # ALG STEP 6 & 7
            b_idx = torch.randperm(batch_size)
            # need to skip remainder
            b_idx = b_idx[: config["step_per_collect"]]
            for start_i in range(0, b_idx.shape[0], self.minibatch_size):
                start_time = time()
                mb_idx = b_idx[start_i : start_i + self.minibatch_size]
                # Calculate V_phi and pi_theta(a_t | s_t)
                v, curr_log_probs, entropy = self.evaluate(
                    batch_usr_embeds[mb_idx],
                    batch_sys_embeds[mb_idx],
                    batch_belief_embeds[mb_idx],
                    batch_db_embeds[mb_idx],
                    batch_turns[mb_idx],
                    batch_response_ids[mb_idx],
                    batch_response_mask[mb_idx],
                )
                ratios = torch.exp(
                    curr_log_probs.to(DEVICE)
                    - old_log_probs[mb_idx].to(DEVICE)
                )
                self.clip_fracs.append(
                    ((ratios - 1.0).abs() > self.clip_range)
                    .float()
                    .mean()
                    .item()
                )

                # Calculate surrogate losses.
                surr1 = ratios * adv[mb_idx]
                surr2 = (
                    torch.clamp(
                        ratios, 1 - self.clip_range, 1 + self.clip_range
                    )
                    * adv[mb_idx]
                )

                # Calculate actor and critic losses.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = ((v - rets[mb_idx]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    - self.et_coef * entropy_loss
                )
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.context_model.parameters(), config["grad_clip"]
                )
                nn.utils.clip_grad_norm_(
                    self.response_model.parameters(), config["grad_clip"]
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), config["grad_clip"]
                )
                self.actor_optim.step()
                self.critic_optim.step()
                self.u_so_far += 1
                update_time = time() - start_time
                logger.debug(f"Update duration: {update_time:.5f}")
                bar.update()
                bar.set_postfix(
                    {
                        "loss": f"{loss.item():.2f}",
                        "a_loss": f"{actor_loss.item():.2f}",
                        "c_loss": f"{critic_loss.item():.2f}",
                        "e_loss": f"{-entropy_loss.item():.2f}",
                    }
                )

                # Log actor loss
                self.actor_losses.append(actor_loss.item())
                self.actor_max_grad.append(
                    max(
                        [
                            p.grad.abs().max().item()
                            for n, p in self.response_model.named_parameters()
                            if p.requires_grad
                            and "bias" not in n
                            and p.grad is not None
                        ]
                    )
                )
                self.critic_losses.append(critic_loss.item())
                self.critic_max_grad.append(
                    max(
                        [
                            p.grad.abs().max().item()
                            for n, p in self.critic.named_parameters()
                            if p.requires_grad
                            and "bias" not in n
                            and p.grad is not None
                        ]
                    )
                )
                self.entropy_losses.append(-entropy_loss.item())
                self.losses.append(loss.item())
                if config["recompute_adv"]:
                    v = v.detach()
                    if self.norm_value:
                        # denormalize
                        v = v * self.ret_stat.std() + self.ret_stat.mean()
                    adv[mb_idx] = self.compute_gae(
                        batch_reward[mb_idx],
                        v,
                        batch_done[mb_idx],
                    )
                    rets[mb_idx] = adv[mb_idx] + v.to(DEVICE)
                    if self.norm_adv:
                        adv[mb_idx] = (adv[mb_idx] - adv[mb_idx].mean()) / (
                            adv[mb_idx].std() + 1e-10
                        )
                    if self.norm_value:
                        self.ret_stat.update(rets[mb_idx])
                        rets[mb_idx] = (
                            rets[mb_idx] - self.ret_stat.mean()
                        ) / (self.ret_stat.std() + 1e-10)
                if self.u_so_far % self.update_per_test == 0:
                    self.test()
                if self.u_so_far % self.update_per_log == 0:
                    self.log_outputs()

    def log_outputs(self):
        log_dict = {
            "trainer/env_step": self.t_so_far,
            "trainer/global_step": self.u_so_far,
            "train/actor_loss": np.mean(self.actor_losses).item(),
            "train/actor_max_grad": np.mean(self.actor_max_grad).item(),
            "train/critic_loss": np.mean(self.critic_losses).item(),
            "train/critic_max_grad": np.mean(self.critic_max_grad).item(),
            "train/entropy_loss": np.mean(self.entropy_losses).item(),
            "train/loss": np.mean(self.losses).item(),
            "train/clip_frac": np.mean(self.clip_fracs).item(),
        }
        self.actor_losses = []
        self.actor_max_grad = []
        self.critic_losses = []
        self.critic_max_grad = []
        self.entropy_losses = []
        self.losses = []
        self.clip_fracs = []
        self.global_outputs.append(log_dict)
        if config["use_wandb"]:
            wandb.log(log_dict)

    def train(self, total_timesteps: int, n_warmup_timesteps: int):
        """
        Train the actor and critic networks. Here is where the main PPO algorithm resides.

        Args:
            n_train_timesteps (int): The total number of timesteps to train for.

        """
        self.t_so_far = 0  # Timesteps simulated so far
        self.i_so_far = 0  # Iterations ran so far
        self.u_so_far = 0  # Update ran so far
        self.best_avg_succ = 0
        self.best_avg_len = 0
        self.best_avg_rew = 0
        self.prev_avg_succ = 0  # prev test succ for tqdm bar
        self.best_path = ""
        self.global_outputs = []

        # These are temperory buffers that refreshed when log.
        self.actor_losses = []
        self.actor_max_grad = []
        self.critic_losses = []
        self.critic_max_grad = []
        self.entropy_losses = []
        self.losses = []
        self.clip_fracs = []

        self.context_model.train()
        self.response_model.train()
        self.critic.train()
        if config["test_before_train"]:
            test_outputs: Dict[str, Any] = self.rollout(
                episode_per_collect=config["episode_per_test"],
                is_train=False,
            )
            avg_succ = np.mean([i["success"] for i in test_outputs["ep_info"]])
            avg_len = test_outputs["ep_len"].mean().item()
            avg_rew = test_outputs["ep_rew"].mean().item()
            logger.info(
                f"Before train avg_succ: {avg_succ:.3f}, "
                f"avg_rew: {avg_rew:.3f} "
                f"avg_len: {avg_len:.3f}"
            )
            if config["use_wandb"]:
                wandb.run.summary["before_train_avg_succ"] = avg_succ
                wandb.run.summary["before_train_avg_rew"] = avg_rew
                wandb.run.summary["before_train_avg_len"] = avg_len

        bar = tqdm(
            desc="Training",
            total=n_warmup_timesteps,
            unit="frames",
        )

        bar = tqdm(
            desc="Training",
            initial=self.t_so_far,
            total=total_timesteps,
            unit="frames",
            position=0,
        )
        while self.t_so_far < total_timesteps:  # ALG STEP 2
            (
                batch_usr_embeds,
                batch_sys_embeds,
                batch_belief_embeds,
                batch_db_embeds,
                batch_turns,
                batch_response_ids,
                batch_response_mask,
                batch_log_probs,
                batch_values,
                batch_reward,
                batch_done,
                ep_len,
                ep_rew,
                ep_info,
                ep_history,
                ep_goal,
            ) = self.rollout(
                step_per_collect=config["step_per_collect"], is_train=True
            )  # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            self.t_so_far += ep_len.sum().item()

            # Increment the number of iterations
            self.i_so_far += 1

            # Logging timesteps so far and iterations so far
            avg_succ = np.mean([i["success"] for i in ep_info]).item()
            avg_len = ep_len.mean().item()
            avg_rew = ep_rew.mean().item()
            log_dict = {
                "train/avg_succ": avg_succ,
                "train/avg_rew": avg_rew,
                "train/avg_len": avg_len,
                "trainer/env_step": self.t_so_far,
                "trainer/global_step": self.u_so_far,
                "memory(MB)": torch.cuda.memory_allocated() / 1024 / 1024,
            }
            self.global_outputs.append(log_dict)
            bar.update(int(ep_len.sum().item()))
            bar.set_postfix(
                {
                    "train_succ": f"{avg_succ:.2f}",
                    "test_succ": f"{self.prev_avg_succ:.2f}",
                    "train_rew": f"{avg_rew:.2f}",
                    "train_len": f"{avg_len:.2f}",
                }
            )
            self.update(
                batch_usr_embeds.to(DEVICE),
                batch_sys_embeds.to(DEVICE),
                batch_belief_embeds.to(DEVICE),
                batch_db_embeds.to(DEVICE),
                batch_turns.to(DEVICE),
                batch_response_ids.to(DEVICE),
                batch_response_mask.to(DEVICE),
                batch_log_probs.to(DEVICE),
                batch_values.to(DEVICE),
                batch_reward.to(DEVICE),
                batch_done.to(DEVICE),
            )
            if (
                config["use_wandb"]
                and self.u_so_far - self.last_log_step >= self.update_per_log
            ):
                wandb.log(log_dict)
                self.last_log_step = self.u_so_far

        df = pd.DataFrame(self.global_outputs)
        df.to_csv(os.path.join(self.best_path, "outputs.csv"), index=False)
        lineplot(
            df[["trainer/env_step", "train/avg_succ"]].dropna(),
            x="trainer/env_step",
            y="train/avg_succ",
            title="Train success rate",
            xlabel="frames",
            ylabel="success rate",
            out_filename=os.path.join(
                self.best_path, "train_learning_curve.png"
            ),
            show=False,
        )
        lineplot(
            df[["trainer/env_step", "test/avg_succ"]].dropna(),
            x="trainer/env_step",
            y="test/avg_succ",
            title="Test success rate",
            xlabel="frames",
            out_filename=os.path.join(
                self.best_path, "test_learning_curve.png"
            ),
            show=False,
        )
        print(
            f"The best model's avg_succ={self.best_avg_succ:.3f} "
            f"avg_rew={self.best_avg_rew:.3f} "
            f"avg_len={self.best_avg_len:.3f}"
        )

    def warmup_rollout(
        self,
        step_per_collect: int = None,
        episode_per_collect: int = None,
    ) -> Dict[str, Union[torch.Tensor, List[Dict[str, Any]]]]:
        """
        Note that each turn brings us one sample. So one episode can have
        many samples.

        Return:
            batch_context_ids: The input ids of the history dialogue acts.
                shape: (n_sample, context_len).
                torch.LongTensor.
            batch_context_mask: 0 indicates mask.
                shape: (n_sample, context_len).
                torch.LongTensor.
            batch_response_ids: The input ids of the response dialogue acts.
                shape: (n_sample, response_len).
                torch.LongTensor.
            batch_response_mask: 0 indicates mask.
                shape: (n_sample, response_len)
                torch.LongTensor.
            batch_label_ids: The label ids, including domain, intent and slot.
                shape: (n_sample, response_len)
                torch.LongTensor.
            batch_cat_ids: The category ids.
                shape: (n_sample, response_len)
                torch.LongTensor.
            batch_reward: The rewards. A naive way is to just copy the end
                turn rewards to every turn, or simply give the reward to the
                last timestep
                shape: (total response timesteps).
                torch.FloatTensor.
            batch_done: 1 indicates episode end.
                Shape: (total response timesteps)
                torch.LongTensor.
            ep_len: The number of turns of each episode this batch.
                Shape: (n_epi)
                torch.LongTensor.
            ep_info: The info given by the env at the last timestep of
                every epoch.
                Shape: (n_epi)
                List of Dict.
            ep_rew: The episodic reward.
                torch.FLoatTensor.
            ep_history: The dialog history of every episode.
                List of List of Dict(diaacts)
            ep_goal: The goal of every episode.
                List of Dict.

        All the returned tensors should be cpu tensor.

        """
        if step_per_collect is None and episode_per_collect is None:
            raise ValueError(
                "Provide either `step_per_collect` or `episode_per_collect`."
            )
        if step_per_collect is None:
            step_per_collect = float("inf")
            unit = "ep"
            bar = tqdm(
                desc="Sampling for warmup",
                total=episode_per_collect,
                unit=unit,
            )
        if episode_per_collect is None:
            episode_per_collect = float("inf")
            unit = "frame"
            bar = tqdm(
                desc="Sampling for warmup",
                total=step_per_collect,
                unit=unit,
            )

        env = self.train_env
        batch_context_ids = []  # Tensor of shape (context_len_i)
        batch_context_mask = []  # Tensor of all one of shape (context_len_i)
        batch_response_ids = []
        batch_response_mask = []
        batch_label_ids = []
        batch_cat_ids = []
        batch_reward = []
        batch_done = []
        ep_len = []
        ep_info = []
        ep_rew = []
        ep_history = []
        ep_goal = []
        t_env = 0
        n_ep = 0
        warmup_policy = RulePolicy(character="sys")
        while t_env < step_per_collect and n_ep < episode_per_collect:
            # Before start of episode
            cum_rew = []  # rewards collected per episode
            context_ids, context_type_ids, context_db_vectors = env.reset()
            context_ids = context_ids[-config["max_seq_len"] :]
            context_type_ids = context_type_ids[-config["max_seq_len"] :]
            context_db_vectors = context_db_vectors[-config["max_seq_len"] :]
            done = False
            for t in range(config["max_turn"]):
                t_env += 1  # Increment timesteps ran this batch so far
                diaacts: Dict[str, List[List[str]]] = warmup_policy.predict(
                    env.sys_dst.state
                )

                # env.step requires Dict of d-i -> [[s,v]]
                # action.decoder.decode will do that
                obs, rew, done, info = env.step(diaacts)

                # TODO continue below
                response_ids, label_ids, cat_ids = str_to_ids(
                    flatten_diaact_str=diaact_to_str(diaacts)[0],
                    tokenizer=self.tokenizer,
                    domains=domains,
                    intents=intents,
                    slots=slots,
                )
                response_ids = torch.tensor(response_ids, dtype=torch.long)
                label_ids = torch.tensor(label_ids, dtype=torch.long)
                cat_ids = torch.tensor(cat_ids, dtype=torch.long)

                exceed = (
                    context_ids.shape[0]
                    + response_ids.shape[0]
                    - config["max_seq_len"]
                )
                if exceed > 0:
                    context_ids = context_ids[exceed:]
                batch_context_ids.append(context_ids)
                batch_context_mask.append(
                    torch.ones_like(context_ids, dtype=torch.long)
                )
                batch_response_ids.append(response_ids)
                batch_response_mask.append(torch.ones_like(response_ids))
                batch_label_ids.append(label_ids)
                batch_cat_ids.append(cat_ids)
                cum_rew.append(rew)
                batch_reward.append(rew)
                batch_done.append(done)
                # debug
                # env.sys_dst.state['history]
                if done:
                    break
                context_ids = torch.tensor(
                    obs[-config["max_seq_len"] :],
                    dtype=torch.long,
                )

            # Track episodic lengths and rewards
            ep_info.append(info)
            ep_len.append(t + 1)
            ep_rew.append(sum(cum_rew))
            ep_history.append(env.sys_dst.state["history"])
            ep_goal.append(env.evaluator.goal)
            n_ep += 1
            if unit == "ep":
                bar.update(1)
            else:
                bar.update(t + 1)

        avg_len = np.mean(ep_len).item()
        avg_rew = np.mean(ep_rew).item()
        avg_succ = np.mean([o["success"] for o in ep_info]).item()
        bar.set_postfix(
            {
                f"warmup/avg_len": f"{avg_len:.3f}",
                f"warmup/avg_rew": f"{avg_rew:.3f}",
                f"warmup/avg_succ": f"{avg_succ:.3f}",
            }
        )
        # Reshape data as tensors in the shape specified in function description, before returning
        batch_context_ids = pad_sequence(batch_context_ids, batch_first=True)
        batch_context_mask = pad_sequence(batch_context_mask, batch_first=True)
        batch_response_ids = pad_sequence(batch_response_ids, batch_first=True)
        batch_response_mask = pad_sequence(
            batch_response_mask, batch_first=True
        )
        batch_label_ids = pad_sequence(
            batch_label_ids, batch_first=True, padding_value=-100
        )
        batch_cat_ids = pad_sequence(
            batch_cat_ids, batch_first=True, padding_value=-100
        )
        batch_reward = torch.tensor(batch_reward, dtype=torch.float)
        batch_done = torch.tensor(batch_done, dtype=torch.bool)
        ep_len = torch.tensor(ep_len, dtype=torch.float)

        assert batch_context_ids.shape == batch_context_mask.shape
        assert (
            batch_response_ids.shape
            == batch_response_mask.shape
            == batch_label_ids.shape
            == batch_cat_ids.shape
        )
        assert batch_reward.shape == batch_done.shape
        assert len(ep_info) == len(ep_len)

        return {
            "batch_context_ids": batch_context_ids,
            "batch_context_mask": batch_context_mask,
            "batch_response_ids": batch_response_ids,
            "batch_response_mask": batch_response_mask,
            "batch_label_ids": batch_label_ids,
            "batch_cat_ids": batch_cat_ids,
            "batch_reward": batch_reward,
            "batch_done": batch_done,
            "ep_len": ep_len,
            "ep_rew": torch.tensor(ep_rew, dtype=torch.float),
            "ep_info": ep_info,
            "ep_history": ep_history,
            "ep_goal": ep_goal,
        }

    def rollout(
        self,
        step_per_collect: int = None,
        episode_per_collect: int = None,
        is_train=True,
    ) -> Dict[str, Union[torch.Tensor, List[Dict[str, Any]]]]:
        """
        Note that each turn brings us one sample. So one episode can have
        many samples.

        Return:
            batch_usr_embeds:  The last usr diaact embeds
                shape: (n_sample, E).
            batch_sys_embeds: The last sys diaact embeds.
                shape: (n_sample, E).
            batch_belief_embeds: The belief embeds.
                shape: (n_sample, E).
            batch_db_embeds: The db embeds.
                shape: (n_sample, E).
            batch_turns: The turn numbers.
                shape: (n_samples).
            batch_response_ids: The response token ids.
                shape: (n_samples, T)
            batch_response_mask: The response mask.
                shape: (n_samples, T)
            batch_log_probs: The log probs of the responses, which are the sum
                of log probs of all the response tokens.
                Shape: (n_samples)
            batch_values: The critic values
                Shape: (n_samples)
            batch_reward: The rewards.
                Shape: (n_samples)
            batch_done: .
                Shape: (n_samples)
            ep_len: The episodic length.
                torch.FLoatTensor.
            ep_rew: The episodic reward.
                torch.FLoatTensor.
            ep_history: The dialog history of every episode.
                List of List of Dict(diaacts)
            ep_goal: The goal of every episode.
                List of Dict.

        All the returned tensors should be cpu tensor.

        """
        if step_per_collect is None and episode_per_collect is None:
            raise ValueError(
                "Provide either `step_per_collect` or `episode_per_collect`."
            )
        if is_train and self.critic is None:
            raise ValueError("Critic must be provided for training.")
        if step_per_collect is None:
            step_per_collect = float("inf")
            unit = "ep"
            bar = tqdm(
                desc="Train Sampling" if is_train else "Sampling for test",
                total=episode_per_collect,
                unit=unit,
                position=1,
                leave=False,
            )
        if episode_per_collect is None:
            episode_per_collect = float("inf")
            unit = "frame"
            bar = tqdm(
                desc="Train Sampling" if is_train else "Sampling for test",
                total=step_per_collect,
                unit=unit,
                position=1,
                leave=False,
            )

        env = self.train_env if is_train else self.test_env

        # usr last diaact embs , Tensor of shape (E)
        batch_usr_embeds = []
        # sys last diaact embs, Tensor of shape (E)
        batch_sys_embeds = []
        # belief embds, Tensor of shape (E)
        batch_belief_embeds = []
        # db embd, Tensor of shape (E)
        batch_db_embeds = []
        # the turn numbers, Tensor of shape ()
        batch_turns = []

        # Tensor of shape (T)
        batch_response_ids = []
        # Tensor of all one of shape (T)
        batch_response_mask = []

        # Tensor of shape ()
        batch_log_probs = []
        # Tensor of shape ()
        batch_values = []
        # List of floats
        batch_reward = []
        # List of bools
        batch_done = []
        ep_len = []
        ep_info = []
        ep_rew = []
        ep_history = []
        ep_goal = []
        t_env = 0
        n_ep = 0
        while t_env < step_per_collect and n_ep < episode_per_collect:
            # Before start of episode
            cum_rew = []  # rewards collected per episode
            (
                last_usr_diaact_ids,  # shape (T1)
                last_sys_diaact_ids,  # shape (T2)
                belief_ids,  # shape (T3)
                db_ids,  # shape (T3)
                turn,  # shape ()
            ) = env.reset()
            done = False
            start_time = time()
            for t in range(config["max_turn"]):
                last_usr_diaact_ids = last_usr_diaact_ids.unsqueeze(dim=0).to(
                    DEVICE
                )
                last_sys_diaact_ids = last_sys_diaact_ids.unsqueeze(dim=0).to(
                    DEVICE
                )
                belief_ids = belief_ids.unsqueeze(dim=0).to(DEVICE)
                db_ids = db_ids.unsqueeze(dim=0).to(DEVICE)
                # shape (1)
                turn = turn.unsqueeze(dim=0).to(DEVICE)
                with torch.no_grad():
                    t_env += 1  # Increment timesteps ran this batch so far
                    start_time = time()
                    # shape (1, E)
                    last_usr_diaact_embeds = self.context_model(
                        input_ids=last_usr_diaact_ids
                    )[1]
                    # shape (1, E)
                    last_sys_diaact_embeds = self.context_model(
                        input_ids=last_sys_diaact_ids
                    )[1]
                    # shape (1, E)
                    belief_embeds = self.context_model(input_ids=belief_ids)[1]
                    # shape (1, E)
                    db_embeds = self.context_model(input_ids=db_ids)[1]
                    duration = time() - start_time
                    start_time = time()
                    # shape (1,1)
                    output_ids = torch.tensor(
                        self.response_tokenizer.encode(
                            config["decoder_prefix"]
                        ),
                        dtype=torch.long,
                        device=DEVICE,
                    ).unsqueeze(dim=0)
                    # shape (1, T)
                    output_ids: torch.Tensor = self.response_model.generate(
                        tgt_ids=output_ids,
                        last_usr_diaact_embeds=last_usr_diaact_embeds,
                        last_sys_diaact_embeds=last_sys_diaact_embeds,
                        belief_embeds=belief_embeds,
                        db_embeds=db_embeds,
                        turns=turn
                        if config["add_encoder_turn_embedding"]
                        else None,
                        eos_token_id=self.response_tokenizer.convert_tokens_to_ids(
                            config["end_token"]
                        ),
                        do_sample=is_train,
                    )
                    outputs = self.response_model(
                        tgt_ids=output_ids,
                        last_usr_diaact_embeds=last_usr_diaact_embeds,
                        last_sys_diaact_embeds=last_sys_diaact_embeds,
                        belief_embeds=belief_embeds,
                        db_embeds=db_embeds,
                        turns=turn.to(DEVICE)
                        if config["add_encoder_turn_embedding"]
                        else None,
                    )
                    duration = time() - start_time
                    # shape (1, T-1, V)
                    dist = Categorical(logits=outputs["logits"][:, :-1])
                    # shape (1, T-1)
                    log_probs = dist.log_prob(output_ids[:, 1:])
                    # shape (1, E)
                    last_hidden_state = outputs["hidden_states"][:1, -1]
                    if self.critic is not None:
                        if config["use_critic_transformer"]:
                            # shape (1, 1)
                            v = self.critic(
                                last_usr_diaact_embeds=last_usr_diaact_embeds,
                                last_sys_diaact_embeds=last_sys_diaact_embeds,
                                belief_embeds=belief_embeds,
                                db_embeds=db_embeds,
                                turns=turn.to(DEVICE)
                                if config["add_encoder_turn_embedding"]
                                else None,
                            )
                        else:
                            # shape (1, 1)
                            v = self.critic(last_hidden_state)
                        v = v.squeeze(dim=0)

                # env.step requires Dict of d-i -> [[s,v]]
                # action_decoder.decode will do that
                obs, rew, done, info = env.step(output_ids[0])

                batch_usr_embeds.append(
                    last_usr_diaact_embeds[0].cpu().detach()
                )
                batch_sys_embeds.append(
                    last_sys_diaact_embeds[0].cpu().detach()
                )
                batch_belief_embeds.append(belief_embeds[0].cpu().detach())
                batch_db_embeds.append(db_embeds[0].cpu().detach())
                batch_turns.append(turn[0].cpu().detach())
                batch_response_ids.append(output_ids[0].cpu().detach())
                batch_response_mask.append(
                    torch.ones(len(output_ids[0]), dtype=torch.long)
                )
                batch_log_probs.append(log_probs[0].sum(dim=0).cpu().detach())
                if is_train:
                    batch_values.append(v[0].cpu().detach())
                cum_rew.append(rew)
                batch_reward.append(rew)
                batch_done.append(done)

                if done:
                    break

                (
                    last_usr_diaact_ids,
                    last_sys_diaact_ids,
                    belief_ids,
                    db_ids,
                    turn,
                ) = obs
            duration = time() - start_time
            time_per_step = duration / (t + 1)
            logger.debug(f"Rollout duration per step: {time_per_step:.4f}")

            # Track episodic lengths and rewards
            ep_info.append(info)
            ep_len.append(t + 1)
            ep_rew.append(sum(cum_rew))
            ep_history.append(env.sys_dst.state["history"])
            if env.evaluator:
                ep_goal.append(env.evaluator.goal)
            else:
                p = env.usr.policy
                if hasattr(p, "policy"):
                    p = p.policy
                ep_goal.append(p.goal.domain_goals)
            n_ep += 1
            if unit == "ep":
                bar.update(1)
            else:
                bar.update(t + 1)

            avg_len = np.mean(ep_len).item()
            avg_rew = np.mean(ep_rew).item()
            avg_succ = np.mean([o["success"] for o in ep_info]).item()
            bar.set_postfix(
                {
                    "len": f"{avg_len:.3f}",
                    "rew": f"{avg_rew:.3f}",
                    "suc": f"{avg_succ:.3f}",
                }
            )
        # Reshape data as tensors in the shape specified in function description, before returning
        batch_usr_embeds = rearrange(batch_usr_embeds, "B E -> B E")
        batch_sys_embeds = rearrange(batch_sys_embeds, "B E -> B E")
        batch_belief_embeds = rearrange(batch_belief_embeds, "B E -> B E")
        batch_db_embeds = rearrange(batch_db_embeds, "B E -> B E")
        batch_turns = rearrange(batch_turns, "B -> B")

        batch_response_ids = pad_sequence(batch_response_ids, batch_first=True)
        batch_response_mask = pad_sequence(
            batch_response_mask, batch_first=True
        )

        batch_log_probs = rearrange(batch_log_probs, "B -> B")
        if batch_values:
            batch_values = rearrange(batch_values, "B -> B")
        batch_reward = torch.tensor(batch_reward, dtype=torch.float)
        batch_done = torch.tensor(batch_done, dtype=torch.bool)
        ep_len = torch.tensor(ep_len, dtype=torch.float)
        ep_rew = torch.tensor(ep_rew, dtype=torch.float)

        assert len(ep_info) == len(ep_len)

        return (
            batch_usr_embeds,
            batch_sys_embeds,
            batch_belief_embeds,
            batch_db_embeds,
            batch_turns,
            batch_response_ids,
            batch_response_mask,
            batch_log_probs,
            batch_values,
            batch_reward,
            batch_done,
            ep_len,
            ep_rew,
            ep_info,
            ep_history,
            ep_goal,
        )

    def compute_gae(self, batch_rews, batch_values, batch_done):
        """
        Compute the generalized advantage estimation.

        Parameters:
            batch_rews - the rewards in a batch,
                Shape: (n_timesteps)
            batch_values -
                Shape: (n_timesteps)
            batch_done: 1 indicates the episode terminates.
                Shape: (n_timesteps)

        Return:
            advantages - Shape: (n_timesteps)
        """
        batch_rews = batch_rews.to(DEVICE)
        batch_done = batch_done.bool().to(DEVICE)
        advantages = torch.zeros_like(batch_rews)
        last_gaelam = 0
        ts = advantages.shape[0]

        for t in reversed(range(ts)):
            if t == ts - 1:
                next_value = 0
                next_done = torch.tensor(True, dtype=torch.bool, device=DEVICE)
            else:
                next_value = batch_values[t + 1]
                next_done = batch_done[t + 1]
            delta = (
                batch_rews[t]
                + self.gamma * next_value * (~next_done)
                - batch_values[t]
            )
            last_gaelam = delta + self.gamma * self.lamb * last_gaelam * (
                ~next_done
            )
            advantages[t] = last_gaelam

        return advantages

    def evaluate(
        self,
        batch_usr_embeds,
        batch_sys_embeds,
        batch_belief_embeds,
        batch_db_embeds,
        batch_turns,
        batch_response_ids,
        batch_response_mask,
    ):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from train.

        Args:
        Return:
            v - The critic value of each sample. The last hidden state
                each of example is used as input to the critic network.
                shape: (B)
            log_probs - The sum of log probabilities of each sample.
                shape: (B)
            entropy - The sum ofdistribution entropy of each sample.
                shape: (B)

        """
        outputs = self.response_model(
            tgt_ids=batch_response_ids,
            tgt_key_padding_mask=batch_response_mask,
            last_usr_diaact_embeds=batch_usr_embeds,
            last_sys_diaact_embeds=batch_sys_embeds,
            belief_embeds=batch_belief_embeds,
            db_embeds=batch_db_embeds,
            turns=batch_turns
            if config["add_encoder_turn_embedding"]
            else None,
        )
        # shape (B, T-1)
        mask = batch_response_mask.clone().bool()
        # Ignore the last token.
        mask[:, 0] = 0
        mask = mask.roll(shifts=-1, dims=1)
        dist = Categorical(logits=outputs["logits"][mask])
        # shape (B, T)
        log_probs = torch.zeros_like(batch_response_ids, dtype=torch.float)
        log_probs[mask] = dist.log_prob(
            batch_response_ids[mask.roll(shifts=1, dims=1)]
        )
        # shape (B)
        log_probs = log_probs.sum(dim=1)
        # shape (B, T)
        entropy = torch.zeros_like(batch_response_ids, dtype=torch.float)
        entropy[mask] = dist.entropy()
        entropy = entropy.sum(dim=1)

        # shape (B)
        last_pos = batch_response_mask.sort(dim=1, stable=True)[1][:, -1]
        # shape (B, E)
        last_hidden_state = outputs["hidden_states"][
            torch.arange(last_pos.shape[0]), last_pos
        ]
        if config["use_critic_transformer"]:
            # shape (B, 1)
            v = self.critic(
                last_usr_diaact_embeds=batch_usr_embeds,
                last_sys_diaact_embeds=batch_sys_embeds,
                belief_embeds=batch_belief_embeds,
                db_embeds=batch_db_embeds,
                turns=batch_turns
                if config["add_encoder_turn_embedding"]
                else None,
            )
        else:
            # shape (B, 1)
            v = self.critic(last_hidden_state)
        # shape (B)
        v = v.squeeze(dim=1)
        return v, log_probs, entropy


def get_env(
    config: Dict[str, Any], context_tokenizer, response_tokenizer, is_train
):
    seed = config["train_env_seed"] if is_train else config["test_env_seed"]
    seeder = get_seeder(seed)
    action_encoder = SGDActionEncoder(
        tokenizer=response_tokenizer,
        output_act_prefix=config["output_act_prefix"],
        decoder_prefix=config["decoder_prefix"],
        end_token=config["end_token"],
        lower_case=True,
    )
    state_encoder = SGDStateEncoder(
        tokenizer=context_tokenizer,
        usr_diaact_prefix=config["usr_diaact_prefix"],
        sys_diaact_prefix=config["sys_diaact_prefix"],
        belief_prefix=config["belief_prefix"],
        db_prefix=config["db_prefix"],
        remove_belief_value=config.get("remove_belief_value", False),
        add_repeat_act_num=config.get("add_repeat_act_num", False),
    )
    if config["use_gentus"]:
        policy_usr = UserPolicy(
            SGD_CKPT,
            mode="semantic",
            dataset="sgd",
            data_split="train" if is_train else "validate",
            seeder=seeder,
            output_dict=True,
        )
    else:
        policy_usr = UserPolicyAgendaSGD(
            max_turn=config["max_turn"],
            seeder=seeder,
            lower_case=True,
        )

    simulator = PipelineAgent(
        nlu=None,
        dst=None,
        policy=policy_usr,
        nlg=None,
        name="usr",
    )
    dst = RuleDST(lower_case=True)
    # evaluator = MultiwozEvaluator()
    if config["rew_type"] == "normal":
        env = Environment(
            sys_nlg=None,
            usr=simulator,
            sys_nlu=None,
            sys_dst=dst,
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            evaluator=None,
            encode_state=True,
        )
    elif config["rew_type"] == "aggressive":
        env = AggressiveRewardEnv(
            sys_nlg=None,
            usr=simulator,
            sys_nlu=None,
            sys_dst=dst,
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            evaluator=None,
            encode_state=True,
            # inform_intents=["inform", "inform", "offer"],
            inform_intents=["inform"],
            request_intents=["request"],
        )
    elif config["rew_type"] == "conservative":
        env = ConservativeRewardEnv(
            sys_nlg=None,
            usr=simulator,
            sys_nlu=None,
            sys_dst=dst,
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            evaluator=None,
            encode_state=True,
            inform_intents=["inform"],
            request_intents=["request"],
        )
    elif config["rew_type"] == "actionnum":
        env = ActionNumEnvironment(
            sys_nlg=None,
            usr=simulator,
            sys_nlu=None,
            sys_dst=dst,
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            evaluator=None,
            encode_state=True,
        )
    return env


def test(config):
    config_saved = json.load(
        open(os.path.join(config["checkpoint"], "config.json"), "r")
    )
    torch_saved = torch.load(
        os.path.join(config["checkpoint"], "ckpt.pth"), map_location=DEVICE
    )
    # solve a bug here
    if "prefinetuned_path" in config_saved:
        config_saved2 = json.load(
            open(
                os.path.join(config_saved["prefinetuned_path"], "config.json"),
                "r",
            )
        )
        for k, v in config_saved2.items():
            if k not in config_saved:
                config_saved[k] = v

    torch_saved["context_model"] = {
        k.replace("module.", ""): v
        for k, v in torch_saved["context_model"].items()
    }
    if "from_scratch" not in config_saved:
        config_saved["from_scratch"] = False
    context_model, _ = get_context_model(
        config_saved, remove_dropout=config["remove_dropout"]
    )
    # end comment
    context_model = context_model.to(DEVICE)
    context_model.load_state_dict(torch_saved["context_model"])
    context_tokenizer = torch_saved["context_tokenizer"]
    response_tokenizer = torch_saved["response_tokenizer"]
    response_model = get_response_model(
        config_saved,
        vocab_size=len(response_tokenizer),
        context_hidden_size=context_model.config.hidden_size,
        dropout_p=0.0 if config["remove_dropout"] else 0.1,
    ).to(DEVICE)
    response_model.load_state_dict(torch_saved["response_model"])
    config["output_act_prefix"] = config_saved["output_act_prefix"]
    config["decoder_prefix"] = config_saved["decoder_prefix"]
    config["end_token"] = config_saved["end_token"]
    config["usr_diaact_prefix"] = config_saved["usr_diaact_prefix"]
    config["sys_diaact_prefix"] = config_saved["sys_diaact_prefix"]
    config["belief_prefix"] = config_saved["belief_prefix"]
    config["db_prefix"] = config_saved["db_prefix"]
    config["add_encoder_turn_embedding"] = config_saved[
        "add_encoder_turn_embedding"
    ]
    config["remove_belief_value"] = config_saved.get(
        "remove_belief_value", False
    )
    config["add_repeat_act_num"] = config_saved.get(
        "add_repeat_act_num", False
    )
    test_env = get_env(
        config=config,
        context_tokenizer=context_tokenizer,
        response_tokenizer=response_tokenizer,
        is_train=False,
    )

    trainer = PPOTrainer(
        context_model=context_model,
        response_model=response_model,
        critic=None,
        context_tokenizer=context_tokenizer,
        response_tokenizer=response_tokenizer,
        train_env=None,
        test_env=test_env,
        actor_lr=1,
        critic_lr=1,
        epoch_per_update=1,
        minibatch_per_epoch=1,
        step_per_collect=1,
        update_per_test=1,
        episode_per_test=1,
        update_per_log=1,
        norm_adv=1,
        norm_value=1,
        gamma=1,
        lamb=1,
        clip_range=1,
        vf_coef=1,
        et_coef=1,
    )
    (
        batch_usr_embeds,
        batch_sys_embeds,
        batch_belief_embeds,
        batch_db_embeds,
        batch_turns,
        batch_response_ids,
        batch_response_mask,
        batch_log_probs,
        batch_values,
        batch_reward,
        batch_done,
        ep_len,
        ep_rew,
        ep_info,
        ep_history,
        ep_goal,
    ) = trainer.rollout(
        episode_per_collect=config["episode_per_test"], is_train=False
    )  # ALG STEP 3
    avg_succ = np.mean([i["success"] for i in ep_info])
    avg_len = ep_len.mean().item()
    avg_rew = ep_rew.mean().item()
    extract_rollout_output(
        ep_history,
        ep_info,
        ep_rew,
        ep_goal,
        ep_len,
        os.path.join(config["checkpoint"], "test_outputs.txt"),
    )
    print(f"Test results for {config['checkpoint']}.")
    print(f"succ: {avg_succ:.3f}, len: {avg_len:.3f}, rew: {avg_rew:.3f}")
    return avg_succ, avg_len, avg_rew


def extract_rollout_output(
    ep_history, ep_info, ep_rew, ep_goal, ep_len, output_path=None
):
    print_outputs = []
    for i, (
        dh,
        info,
        rew,
        goal,
        l,
    ) in enumerate(zip(ep_history, ep_info, ep_rew, ep_goal, ep_len)):
        print_output = (
            f"example idx: {i}\n"
            f"success: {info['success']}\n"
            f"ep reward: {rew.item()}\n"
            f"len: {l.item()}\n"
            f"user_goal: {json.dumps(goal,indent=2)}\n"
            "dialog_history:\n"
        )
        for i, turns in enumerate(dh):
            if i > 0:
                print_output += f"sys: {str(turns[0])}\n"
            print_output += f"usr: {str(turns[1])}\n"
        print_output += "\n"
        print_outputs.append(print_output)
    if output_path is not None:
        with open(
            output_path,
            "w",
        ) as f:
            f.write("\n".join(print_outputs))
    return print_outputs


def train(config):
    pprint(config)
    if not config["prefinetuned_path"]:
        raise ValueError(
            "Currently only support training with prefinetuned model."
        )
    config_saved = json.load(
        open(os.path.join(config["prefinetuned_path"], "config.json"), "r")
    )
    torch_saved = torch.load(
        os.path.join(config["prefinetuned_path"], "ckpt.pth"),
        map_location=DEVICE,
    )
    context_model, _ = get_context_model(
        config_saved, remove_dropout=config["remove_dropout"]
    )
    context_model = context_model.to(DEVICE)
    context_model.load_state_dict(torch_saved["context_model"])
    if config_saved["freeze_context_model"]:
        freeze_model(context_model)
        logger.info("Freezed context model.")
    context_tokenizer = torch_saved["context_tokenizer"]
    response_tokenizer = torch_saved["response_tokenizer"]
    response_model = get_response_model(
        config_saved,
        vocab_size=len(response_tokenizer),
        context_hidden_size=context_model.config.hidden_size,
        dropout_p=0.0 if config["remove_dropout"] else 0.1,
    ).to(DEVICE)
    response_model.load_state_dict(torch_saved["response_model"])
    context_model = torch.nn.DataParallel(
        context_model, device_ids=list(range(torch.cuda.device_count()))
    )

    # context model parameters
    config["context_model_name"] = config_saved["context_model_name"]
    config["from_scratch"] = config_saved["from_scratch"]

    # response model parameters
    config["hidden_size"] = config_saved["hidden_size"]
    config["n_heads"] = config_saved["n_heads"]
    config["n_encoder_layers"] = config_saved["n_encoder_layers"]
    config["n_decoder_layers"] = config_saved["n_decoder_layers"]
    config["add_encoder_turn_embedding"] = config_saved[
        "add_encoder_turn_embedding"
    ]
    config["add_encoder_type_embedding"] = config_saved[
        "add_encoder_type_embedding"
    ]
    config["add_decoder_pos_embedding"] = config_saved[
        "add_decoder_pos_embedding"
    ]
    config["add_decoder_type_embedding"] = config_saved[
        "add_decoder_type_embedding"
    ]
    config["tie_weights"] = config_saved["tie_weights"]

    config["output_act_prefix"] = config_saved["output_act_prefix"]
    config["decoder_prefix"] = config_saved["decoder_prefix"]
    config["end_token"] = config_saved["end_token"]
    config["usr_diaact_prefix"] = config_saved["usr_diaact_prefix"]
    config["sys_diaact_prefix"] = config_saved["sys_diaact_prefix"]
    config["belief_prefix"] = config_saved["belief_prefix"]
    config["db_prefix"] = config_saved["db_prefix"]
    config["add_encoder_turn_embedding"] = config_saved[
        "add_encoder_turn_embedding"
    ]
    if "remove_belief_value" in config_saved:
        config["remove_belief_value"] = config_saved["remove_belief_value"]
    else:
        logger.info(
            "remove_belief_value not found in saved config. Probably an old"
            " version."
        )
        config["remove_belief_value"] = False
    if "add_repeat_act_num" in config_saved:
        config["add_repeat_act_num"] = config_saved["add_repeat_act_num"]
    else:
        logger.info(
            "add_repeat_act_num not found in saved config. Probably an old"
            " version."
        )
        config["add_repeat_act_num"] = False

    # can be figured to be more complicated
    if config["use_critic_transformer"]:
        critic = CriticTransformer(
            context_hidden_size=context_model.module.config.hidden_size
            if hasattr(context_model, "module")
            else context_model.config.hidden_size,
            hidden_size=config_saved["hidden_size"],  # same as response model
            n_heads=1,
            n_layers=1,
            dropout_p=0,
            add_turn_embedding=config_saved["add_encoder_turn_embedding"],
            pool_hidden_sizes=config["critic_hidden_sizes"],
        ).to(DEVICE)
        logger.info("Using transformer critic.")
    else:
        critic = MLP(
            input_dim=(
                response_model.module.hidden_size
                if hasattr(response_model, "module")
                else response_model.hidden_size
            ),
            hidden_sizes=config["critic_hidden_sizes"],
            output_dim=1,
            activation=nn.ReLU,
        ).to(DEVICE)
        logger.info("Using MLP critic.")

    train_env = get_env(
        config=config,
        context_tokenizer=context_tokenizer,
        response_tokenizer=response_tokenizer,
        is_train=True,
    )
    test_env = get_env(
        config=config,
        context_tokenizer=context_tokenizer,
        response_tokenizer=response_tokenizer,
        is_train=False,
    )
    trainer = PPOTrainer(
        context_model=context_model,
        response_model=response_model,
        critic=critic,
        context_tokenizer=context_tokenizer,
        response_tokenizer=response_tokenizer,
        train_env=train_env,
        test_env=test_env,
        actor_lr=config["actor_learning_rate"],
        critic_lr=config["critic_learning_rate"],
        epoch_per_update=config["epoch_per_update"],
        minibatch_per_epoch=config["minibatch_per_epoch"],
        step_per_collect=config["step_per_collect"],
        update_per_test=config["update_per_test"],
        episode_per_test=config["episode_per_test"],
        update_per_log=config["update_per_log"],
        norm_adv=config["norm_adv"],
        norm_value=config["norm_value"],
        gamma=config["gamma"],
        lamb=config["lamb"],
        clip_range=config["clip_range"],
        vf_coef=config["vf_coef"],
        et_coef=config["et_coef"],
    )
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
        wandb.watch(critic, log="all")
    trainer.train(config["n_train_steps"], config["n_warmup_steps"])


def get_random_params(config):
    seeder = random.Random()
    config["norm_adv"] = seeder.choice([True, False])
    config["norm_value"] = seeder.choice([True, False])
    config["recompute_adv"] = seeder.choice([True, False])
    config["gamma"] = seeder.choice([0.9, 0.93, 0.95, 0.97, 0.99])
    config["critic_learning_rate"] = seeder.choice([3e-4, 1e-4, 5e-5, 5e-6])
    config["actor_learning_rate"] = seeder.choice([1e-5, 5e-6, 5e-7])
    minibatch_size = seeder.choice([64, 128, 256])
    config["step_per_collect"] = seeder.choice([512, 1024, 2048])
    config["minibatch_per_epoch"] = int(
        config["step_per_collect"] / minibatch_size
    )
    logger.info(
        f"Random sampled parameters: norm_adv={config['norm_adv']},"
        f" norm_value={config['norm_value']},"
        f" recompute_adv={config['recompute_adv']},"
        f" gamma={config['gamma']},"
        f" critic_learning_rate={config['critic_learning_rate']},"
        f" actor_learning_rate={config['actor_learning_rate']},"
        f" step_per_collect={config['step_per_collect']},"
        f" minibatch_per_epoch={config['minibatch_per_epoch']},"
    )


if __name__ == "__main__":
    config = vars(get_args())
    set_seed(config["seed"])
    parser = argparse.ArgumentParser()
    if importlib.util.find_spec("wandb") is None:
        config["use_wandb"] = False
    else:
        import wandb

    logger = get_logger(
        logger_level="debug",
        console_level="info",
    )
    # random param tune:
    # seeder = random.Random()
    # bz = seeder.choice([32, 64])
    # step_per_collect = seeder.choice([256, 512, 768, 1024])
    # epoch_per_update = seeder.choice([2, 4, 6])
    # vf_coef = seeder.choice([0.1, 0.5])
    # critic_hidden_sizes = seeder.choice([[64], [64, 64], [128], [128, 128]])
    # config["step_per_collect"] = step_per_collect
    # config["minibatch_per_epoch"] = int(step_per_collect / bz)
    # config["epoch_per_update"] = epoch_per_update
    # config["vf_coef"] = vf_coef
    # config["critic_hidden_sizes"] = critic_hidden_sizes
    # norm_conf = seeder.choice([0, 1, 2, 3])
    # if norm_conf == 0:
    #     pass
    # elif norm_conf == 1:
    #     config["norm_adv"] = True
    # elif norm_conf == 2:
    #     config["norm_value"] = True
    # elif norm_conf == 3:
    #     config["norm_adv"] = True
    #     config["norm_value"] = True
    if config["random_params"]:
        get_random_params(config)
    if config["do_train"]:
        train(config)
    if config["do_test"]:
        test(config)
