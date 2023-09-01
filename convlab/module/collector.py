import warnings
import time
from typing import Any, Dict, Optional, Callable, Tuple, List
from copy import deepcopy

import numpy as np
import torch
import json

from tianshou.data import (
    Batch,
    to_numpy,
    Collector as C,
)


class Collector(C):
    """A Customizable collector.
    It contains 7 customizable methods:
        * reset_env() -> None
        * reset_buffer() -> nd.array
        * policy_act() -> None
            Get next action. Do update on self.data.
            Mainly self.data.act.
        * env_step(ready_env_ids: np.array) -> (obs, r, done, info)
            Interacts with env.
        * buffer_add(ready_env_ids:np.array) -> (ptr, ep_rew, ep_len, ep_idx)
            Add samples to buffer
        * collect_statistic(ep_len, ep_rew, ep_idx, info: np.array)
            -> Dict[str, Any]
            Collect episodic statistics from info. Maintain new one in
            each collect.
        * process_statistic(Dict[str, List[Any]]) -> Dict[str, Any]
            Process those statistic before returning.
    Attr:
        no_grad will no longer be used. DIY in act method.
    """

    def __init__(self, will_reset: bool = True, *args, **kwargs):
        self.will_reset = will_reset
        super().__init__(*args, **kwargs)

    def reset_env(self, env_ids=None) -> np.ndarray:
        obs = self.env.reset(env_ids)
        if self.preprocess_fn:
            obs = self.preprocess_fn(
                obs=obs, env_id=env_ids if env_ids else np.arange(self.env_num)
            ).get("obs", obs)
        return obs

    def reset_buffer(self, keep_statistics=False) -> None:
        """Use default reset"""
        if self.will_reset:
            super().reset_buffer(keep_statistics)

    def policy_act(self) -> None:
        """Generate policy action"""
        last_state = self.data.policy.pop("hidden_state", None)
        with torch.no_grad():  # faster than retain_grad version
            # self.data.obs will be used by agent to get result
            result = self.policy(self.data, last_state)

        # update state / act / policy into self.data
        policy = result.get("policy", Batch())
        assert isinstance(policy, Batch)
        state = result.get("state", None)
        if state is not None:
            policy.hidden_state = state  # save state into buffer
        act = to_numpy(result.act)
        if self.exploration_noise:
            act = self.policy.exploration_noise(act, self.data)
        self.data.update(policy=policy, act=act)

    def env_step(self, ready_env_ids: np.ndarray) -> None:
        """Interact with env.

        Save interactions to self.data
        """
        # get bounded and remapped actions first (not saved into buffer)
        action_remap = self.policy.map_action(self.data.act)
        # step in env
        obs_next, rew, done, info = self.env.step(action_remap, ready_env_ids)  # type: ignore
        self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)

    def buffer_add(
        self, ready_env_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
            self.data, buffer_ids=ready_env_ids
        )
        return ep_rew, ep_len, ep_idx

    def reset_state(self, local_idx: np.ndarray):
        """Reset some state variables when some episodes finished.
            Typically, we update the next_state variables. The assignment
            of next to present is done in increment_state.

            Again, this method should be used in rare situation.

        Args:
            local_idx (np.ndarray): The idx of finished episodes.
        """
        pass

    def increment_state(self, mask: np.ndarray):
        """Increment present state to next state.
            self.data have been handled already.
            This method should only be used in very rare situation, where I
            have some variables that are incompatable with Batch.

        Args:
            mask: 0 for env that will be shut down. Should apply mask first
            before incrementing.

        Example:
            self.data = self.data[mask]
            self.data.obs = self.data.obs_next
        """
        pass

    def collect_statistic(
        self,
        ep_len: np.ndarray,
        ep_rew: np.ndarray,
        ep_idx: np.ndarray,
        info: np.ndarray,
    ) -> Dict[str, Any]:
        """episode_count and step_count will be collected outside.

        Args:
            ep_len (np.ndarray): array of int.
            ep_rew (np.ndarray): array of floats.
            ep_idx (np.ndarray): array of int.
            info (np.ndarray): array of dicts.
            length of them will be num_episodes finished.

        Returns:
            Dict[str, Any]: Those values will be appended to a list
                corresponding to their keys. These key names will be returned
                in the collect method.


        Example:
            suc_rate should be collected here.
        """
        return {"rews": ep_rew, "lens": ep_len, "idxs": ep_idx}

    def process_statistic(self, stats: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Args:
            stats (Dict[str, List[Any]]): The aggregated stats from
                `collect_statistic` method.

        Returns:
            Dict[str, Any]

        """

        stats = deepcopy(stats)
        # episode count might be 0
        if "rews" not in stats:
            stats["rews"] = np.array([])
            stats["rew"] = 0
            stats["rew_std"] = 0
            stats["lens"] = np.array([], int)
            stats["len"] = 0
            stats["len_std"] = 0
            stats["idxs"] = np.array([], int)
        else:
            rews = np.concatenate(stats["rews"])
            stats["rews"] = rews
            stats["rew"] = rews.mean()
            stats["rew_std"] = rews.std()
            lens = np.concatenate(stats["lens"])
            stats["lens"] = lens
            stats["len"] = lens.mean()
            stats["len_std"] = lens.std()
            stats["idxs"] = np.concatenate(stats["idxs"])
        return stats

    def reset(self) -> None:
        """Reset all related variables in the collector."""
        # use empty Batch for "state" so that self.data supports slicing
        # convert empty Batch to None when passing data to policy
        self.data = Batch(
            obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={}
        )
        obs = self.reset_env()
        self.data.obs = obs
        self.reset_buffer()
        self.reset_stat()

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
    ) -> Dict[str, Any]:
        assert (
            not self.env.is_async
        ), "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                "Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env"
                    f" ({self.env_num}), which may cause extra transitions"
                    " collected into the buffer."
                )
            self.ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            self.ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[: min(self.env_num, n_episode)]
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
            )

        start_time = time.time()
        step_count = 0
        episode_count = 0
        stats = {}

        while True:
            assert len(self.data) == len(self.ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                self.data.update(
                    act=[
                        self._action_space[i].sample()
                        for i in self.ready_env_ids
                    ]
                )
            else:
                self.policy_act()

            self.env_step(self.ready_env_ids)
            done = self.data.done
            info = self.data.info
            # self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)
            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=self.ready_env_ids,
                    )
                )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            ep_rew, ep_len, ep_idx = self.buffer_add(self.ready_env_ids)

            # collect statistics
            step_count += len(self.ready_env_ids)

            mask = np.ones_like(self.ready_env_ids, dtype=bool)
            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = self.ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_stats = self.collect_statistic(
                    ep_len[env_ind_local],
                    ep_rew[env_ind_local],
                    ep_idx[env_ind_local],
                    info[env_ind_local],
                )
                for k, v in episode_stats.items():
                    stats.setdefault(k, []).append(v)
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                obs_reset = self.reset_env(env_ind_global)
                self.data.obs_next[env_ind_local] = obs_reset
                self.reset_state(env_ind_local)
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(self.ready_env_ids) - (
                        n_episode - episode_count
                    )
                    if surplus_env_num > 0:
                        mask[env_ind_local[:surplus_env_num]] = False
                        self.ready_env_ids = self.ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next
            self.increment_state(mask)

            if (n_step and step_count >= n_step) or (
                n_episode and episode_count >= n_episode
            ):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={},
                act={},
                rew={},
                done={},
                obs_next={},
                info={},
                policy={},
            )
            self.data.obs = self.reset_env()

        processed_stats = self.process_statistic(stats)
        processed_stats.update({"n/ep": episode_count, "n/st": step_count})
        return processed_stats


class MultiwozCollector(Collector):
    """
    Track suc_rate.
    """

    def __init__(
        self,
        track_suc_rate: bool = False,
        track_history=False,
        *args,
        **kwargs,
    ):
        self.track_suc_rate = track_suc_rate
        self.track_history = track_history
        super().__init__(*args, **kwargs)

    def collect_statistic(
        self,
        ep_len: np.ndarray,
        ep_rew: np.ndarray,
        ep_idx: np.ndarray,
        info: np.ndarray,
    ) -> Dict[str, Any]:
        d = super().collect_statistic(ep_len, ep_rew, ep_idx, info)
        if self.track_suc_rate:
            d["success_rates"] = []
            for i in info:
                d["success_rates"].append(i["success"])
        if self.track_history:
            d["history"] = [json.loads(i["history"]) for i in info]
        return d

    def process_statistic(self, stats: Dict[str, List[Any]]) -> Dict[str, Any]:
        d = super().process_statistic(stats)
        if self.track_suc_rate:
            d["success_rates"] = np.concatenate(d["success_rates"])
            d["success_rate"] = np.mean(d["success_rates"])
        if self.track_history:
            d["histories"] = [
                l for sublist in d.pop("history") for l in sublist
            ]
        return d


class WarmupCollector(MultiwozCollector):
    def reset_env(self, env_ids=None) -> np.ndarray:
        obs = self.env.reset(env_ids)
        if self.preprocess_fn:
            obs = self.preprocess_fn(
                obs=obs, env_id=env_ids if env_ids else np.arange(self.env_num)
            ).get("obs", obs)
        # some epi finished.
        if env_ids is not None:
            self._next_state_dicts = np.array(obs)
        else:
            self.state_dicts = np.array(obs)
        obs = np.stack([self.policy.state_encoder.encode(s) for s in obs])
        return obs

    def policy_act(self):
        with torch.no_grad():  # faster than retain_grad version
            result = self.policy(self.data, self.state_dicts)
        act = result.act
        if self.exploration_noise:
            act = self.policy.exploration_noise(act, self.data)
        self.data.update(policy=Batch(), act=act)

    def env_step(self, ready_env_ids: np.ndarray) -> None:
        """Interact with env.

        Returns:
            ob_next, rew, done, info. All np array with legnth num_env.

        """
        # get bounded and remapped actions first (not saved into buffer)
        action_remap = self.policy.map_action(self.data.act)
        # step in env
        result = self.env.step(action_remap, ready_env_ids)  # type: ignore
        obs_next, rew, done, info = result
        # my code
        self.next_state_dicts = np.array(obs_next)
        obs_next = np.stack(
            [self.policy.state_encoder.encode(s) for s in obs_next]
        )
        self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)

    def reset_state(self, local_idx):
        self.next_state_dicts[local_idx] = self._next_state_dicts

    def increment_state(self, mask):
        self.next_state_dicts = self.next_state_dicts[mask]
        self.state_dicts = self.next_state_dicts
