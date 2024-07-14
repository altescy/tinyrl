from typing import Generic

import torch

from tinyrl.types import T_State


class BasePolicyNetwork(Generic[T_State], torch.nn.Module):
    def forward(self, state: T_State) -> torch.Tensor:
        raise NotImplementedError


class BaseValueNetwork(Generic[T_State], torch.nn.Module):
    def forward(self, state: T_State) -> torch.Tensor:
        raise NotImplementedError
