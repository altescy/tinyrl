from typing import Generic

import torch

from tinyrl.distribution import BaseDistribution
from tinyrl.types import T_Action, T_Scalar, T_State


class BaseAgent(Generic[T_State, T_Action, T_Scalar]):
    def act(self, state: T_State) -> T_Action:
        action, _ = self.sample(state)
        return action

    def prob(self, state: T_State, action: T_Action) -> T_Scalar:
        dist = self.dist(state)
        return dist.prob(action)

    def sample(self, state: T_State) -> tuple[T_Action, T_Scalar]:
        dist = self.dist(state)
        action = dist.sample()
        return action, dist.prob(action)

    def dist(self, state: T_State) -> BaseDistribution[T_Action, T_Scalar]:
        raise NotImplementedError


class BaseTorchAgent(
    torch.nn.Module,
    BaseAgent[T_State, T_Action, torch.Tensor],
    Generic[T_State, T_Action],
):
    pass
