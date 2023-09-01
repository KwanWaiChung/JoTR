import time
import numpy as np
import tqdm

from collections import defaultdict
from typing import Callable, Dict, Optional, Union, Any
from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer import gather_info
from convlab.module.test import test_episode
from tianshou.utils import BaseLogger, LazyLogger, MovAvg, tqdm_config


def offpolicy_trainer(
    policy: BasePolicy,
    train_collector: Collector,
    test_collector: Collector,
    max_epoch: int,
    step_per_epoch: int,
    episode_per_test: int,
    batch_size: int,
    step_per_collect: Optional[int] = None,
    episode_per_collect: Optional[int] = None,
    update_per_step: Union[int, float] = 1,
    train_fn: Optional[Callable[[int, int], None]] = None,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    stop_fn: Optional[Callable[[float], bool]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    save_checkpoint_fn: Optional[
        Callable[[int, int, int], Dict[str, Any]]
    ] = None,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    logger: BaseLogger = LazyLogger(),
    verbose: bool = True,
    test_in_train: bool = True,
    start_epoch: int = None,
    env_step: int = None,
    gradient_step: int = None,
    track_success_rate: bool = False,
) -> Dict[str, Union[float, str]]:
    """My modified version of offpolicy trainer.

    Modifications:
        1. Allow input start_epoch, env_step, gradient_step. ok
            Compatable.
        2. Add gradient step and success rate to return dict. ok
            Compatable.
        3. Track success rate from test. add parameter `track_success_rate`. ok
            Add success_rate to info dict only when episode done.
            Compatable.
        4. Only do save_fn when best success rate is achieved. ok
            Change save_fn signature to ``f(epoch: int, env_step: int,
            gradient_step: int, success_rate: float) -> str ``
        5. Change save_checkpoint_fn signature to ``f(int,int,int) -> Dict``. 
            It return the dict needs to be saved. Also removed resume_from_log.
        6. Allow episode_per_collect.
            Compatable.
    """
    start_epoch = start_epoch or 0
    env_step = env_step or 0
    gradient_step = gradient_step or 0
    last_rew, last_len = 0.0, 0
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    train_collector.reset_stat()
    test_collector.reset_stat()
    test_in_train = test_in_train and train_collector.policy == policy
    test_result = test_episode(
        policy,
        test_collector,
        test_fn,
        start_epoch,
        episode_per_test,
        logger,
        env_step,
        reward_metric,
    )
    best_epoch = start_epoch
    best_reward = test_result["rew"]
    best_reward_std = test_result["rew_std"]
    if track_success_rate:
        best_suc = test_result["success_rate"]

    for epoch in range(1 + start_epoch, 1 + max_epoch):
        # train
        policy.train()
        with tqdm.tqdm(
            total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                if train_fn:
                    train_fn(epoch, env_step)
                result = train_collector.collect(
                    n_step=step_per_collect, n_episode=episode_per_collect
                )
                if result["n/ep"] > 0 and reward_metric:
                    rew = reward_metric(result["rews"])
                    result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
                env_step += int(result["n/st"])
                t.update(result["n/st"])
                logger.log_train_data(result, env_step)
                last_rew = result["rew"] if result["n/ep"] > 0 else last_rew
                last_len = result["len"] if result["n/ep"] > 0 else last_len
                data = {
                    "env_step": str(env_step),
                    "rew": f"{last_rew:.2f}",
                    "len": f"{last_len:.2f}",
                    "n/ep": str(int(result["n/ep"])),
                    "n/st": str(int(result["n/st"])),
                    "eps": f"{policy.eps:.3f}",
                }
                if result["n/ep"] > 0:
                    if test_in_train and stop_fn and stop_fn(result["rew"]):
                        test_result = test_episode(
                            policy,
                            test_collector,
                            test_fn,
                            epoch,
                            episode_per_test,
                            logger,
                            env_step,
                        )
                        if stop_fn(test_result["rew"]):
                            if save_fn:
                                save_fn(policy)
                            logger.save_data(
                                epoch,
                                env_step,
                                gradient_step,
                                save_checkpoint_fn,
                            )
                            t.set_postfix(**data)
                            return gather_info(
                                start_time,
                                train_collector,
                                test_collector,
                                test_result["rew"],
                                test_result["rew_std"],
                            )
                        else:
                            policy.train()
                for _ in range(round(update_per_step * result["n/st"])):
                    gradient_step += 1
                    losses = policy.update(batch_size, train_collector.buffer)
                    for k in losses.keys():
                        # give exception to lr
                        if "lr" not in k:
                            stat[k].add(losses[k])
                            losses[k] = stat[k].get()
                            data[k] = f"{losses[k]:.3f}"
                        else:
                            data[k] = f"{losses[k]:e}"
                    logger.log_update_data(losses, gradient_step)
                    t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # test
        test_result = test_episode(
            policy,
            test_collector,
            test_fn,
            epoch,
            episode_per_test,
            logger,
            env_step,
            reward_metric,
        )
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        suc = -1
        if "success_rate" in test_result:
            suc = test_result["success_rate"]

        if best_epoch < 0 or best_reward < rew:
            best_reward, best_reward_std = rew, rew_std
            if save_fn and not track_success_rate:
                if hasattr(logger, "save_best"):
                    logger.save_best(
                        epoch, env_step, gradient_step, suc, save_fn
                    )
                else:
                    save_fn(policy)
        if best_epoch < 0 or (best_suc < suc and track_success_rate):
            best_epoch, best_suc = epoch, suc
            if save_fn:
                if hasattr(logger, "save_best"):
                    logger.save_best(
                        epoch, env_step, gradient_step, suc, save_fn
                    )
                else:
                    save_fn(epoch, env_step, gradient_step, suc)

        logger.save_data(epoch, env_step, gradient_step, save_checkpoint_fn)
        if verbose and not track_success_rate:
            print(
                f"Epoch #{epoch}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
                f"ard: {best_reward:.6f} ± {best_reward_std:.6f} in #{best_epoch}"
            )
        if verbose and track_success_rate:
            print(
                f"Epoch #{epoch}: test_reward: {rew:.6f} ± {rew_std:.6f}, "
                f"test_success_rate: {suc}, best_success_rate: {best_suc} in "
                f"#{best_epoch}"
            )

        if stop_fn and stop_fn(best_reward):
            break

    return_dict = gather_info(
        start_time,
        train_collector,
        test_collector,
        best_reward,
        best_reward_std,
    )
    return_dict.update({"gradient_step": gradient_step})
    if track_success_rate:
        return_dict.update({"best_success_rate": f"{best_suc:.2f}"})
    return return_dict
