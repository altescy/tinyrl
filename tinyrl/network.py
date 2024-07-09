from typing import Generic, TypeVar

import torch

_T_State = TypeVar("_T_State")


class BasePolicyNetwork(Generic[_T_State], torch.nn.Module):
    def forward(self, state: _T_State) -> torch.Tensor:
        raise NotImplementedError
