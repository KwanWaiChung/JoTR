from matplotlib.pyplot import get
from tianshou.policy import DQNPolicy as P
from tianshou.data import ReplayBuffer
from pytorch_template.utils import getlogger
from typing import Any, Dict, Optional
import torch


class DQNPolicy(P):
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        lr_scheduler=None,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            optim,
            discount_factor,
            estimation_step,
            target_update_freq,
            reward_normalization,
            is_double,
            **kwargs,
        )
        self.lr_scheduler = lr_scheduler

    def learn(self, batch) -> Dict[str, float]:
        d = super().learn(batch)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            d["lr"] = self.lr_scheduler.get_last_lr()[0]
        return d


class WDQNPolicy(DQNPolicy):
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        warmup_buffer: ReplayBuffer = None,
        lr_scheduler=None,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            optim,
            lr_scheduler,
            discount_factor,
            estimation_step,
            target_update_freq,
            reward_normalization,
            is_double,
            **kwargs,
        )
        self.logger = getlogger(__name__)
        if warmup_buffer is None:
            self.logger.info(
                f"Didn't received a warmup buffer argument. "
                "Please double check if it's intended."
            )
        self.warmup_sample = True
        self.warmup_buffer = warmup_buffer

    def update(
        self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any
    ) -> Dict[str, Any]:
        if self.warmup_buffer and (
            self.warmup_sample or len(buffer) < sample_size
        ):
            res = super().update(sample_size, self.warmup_buffer, **kwargs)
        else:
            res = super().update(sample_size, buffer, **kwargs)
        self.warmup_sample = not self.warmup_sample
        return res
