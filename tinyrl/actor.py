from typing import Generic, TypeVar

import torch

_T_Action = TypeVar("_T_Action")


class BaseActor(Generic[_T_Action]):
    def select(self, probs: torch.Tensor, action: _T_Action) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, probs: torch.Tensor) -> tuple[_T_Action, float]:
        raise NotImplementedError
