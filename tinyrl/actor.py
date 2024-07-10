from typing import Generic

import torch

from tinyrl.types import T_Action


class BaseActor(Generic[T_Action]):
    def index(self, action: T_Action) -> int:
        raise NotImplementedError

    def sample(self, probs: torch.Tensor) -> tuple[T_Action, float]:
        raise NotImplementedError

    def select(self, probs: torch.Tensor, action: T_Action) -> torch.Tensor:
        return probs[self.index(action)]
