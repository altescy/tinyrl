from typing import Generic

import numpy
import torch

from tinyrl.distribution import BaseDistribution, CategoricalDistribution
from tinyrl.types import T_Action, T_Scalar, T_State
from tinyrl.utils import softmax


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


class QLearningAgent(BaseAgent[T_State, T_Action, float]):
    def __init__(
        self,
        actions: set[T_Action],
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._temperature = temperature
        self._actions = actions
        self._q_values: dict[tuple[T_State, T_Action], float] = {}

    def dist(
        self,
        state: T_State,
    ) -> CategoricalDistribution[T_Action]:
        actions = list(self._actions)
        if self._q_values:
            action_probs = softmax([self._q_values.get((state, action), 0.0) / self._temperature for action in actions])
        else:
            action_probs = numpy.ones(len(actions)) / len(actions)
        return CategoricalDistribution(action_probs, actions)

    def update(
        self,
        state: T_State,
        action: T_Action,
        reward: float,
        next_state: T_State,
    ) -> None:
        current_q = self._q_values.get((state, action), 0)
        next_q_values = [self._q_values.get((next_state, next_action), 0) for next_action in self._actions]
        max_next_q = max(next_q_values) if next_q_values else 0
        new_q = current_q + self._alpha * (reward + self._gamma * max_next_q - current_q)
        self._q_values[(state, action)] = new_q
