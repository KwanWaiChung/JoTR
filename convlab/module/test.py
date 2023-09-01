import time
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils import BaseLogger


def test_episode(
    policy: BasePolicy,
    collector: Collector,
    test_fn: Optional[Callable[[int, Optional[int]], None]],
    epoch: int,
    n_episode: int,
    logger: Optional[BaseLogger] = None,
    global_step: Optional[int] = None,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """A simple wrapper of testing policy in collector."""
    collector.data.obs = collector.reset_env()
    collector.reset_buffer()
    policy.eval()
    if test_fn:
        test_fn(epoch, global_step)
    result = collector.collect(n_episode=n_episode)
    if reward_metric:
        rew = reward_metric(result["rews"])
        result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
    if logger and global_step is not None:
        logger.log_test_data(result, global_step)
    return result
