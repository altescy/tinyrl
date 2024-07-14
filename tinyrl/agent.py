from typing import Generic

import torch

from tinyrl.constraints import BaseActionConstraint
from tinyrl.distribution import BaseDistribution, CategoricalDistribution
from tinyrl.environment import BaseEnvironment
from tinyrl.types import T_Action, T_Scalar, T_State
from tinyrl.utils import softmax


class BaseAgent(Generic[T_State, T_Action, T_Scalar]):
    def __init__(
        self,
        constraint: BaseActionConstraint[T_State, T_Action, T_Scalar] | None = None,
    ) -> None:
        self._constraint: BaseActionConstraint | None = constraint

    def act(self, env: BaseEnvironment[T_State, T_Action], state: T_State) -> T_Action:
        action, _ = self.sample(env, state)
        return action

    def prob(self, env: BaseEnvironment[T_State, T_Action], state: T_State, action: T_Action) -> T_Scalar:
        dist = self.dist(state)
        if self._constraint is not None:
            dist = self._constraint.constrain(env, state, dist)
        return dist.prob(action)

    def sample(self, env: BaseEnvironment[T_State, T_Action], state: T_State) -> tuple[T_Action, T_Scalar]:
        dist = self.dist(state)
        if self._constraint is not None:
            dist = self._constraint.constrain(env, state, dist)
        action = dist.sample()
        return action, dist.prob(action)

    def dist(self, state: T_State) -> BaseDistribution[T_Action, T_Scalar]:
        raise NotImplementedError


class BaseTorchAgent(
    torch.nn.Module,
    BaseAgent[T_State, T_Action, torch.Tensor],
    Generic[T_State, T_Action],
):
    def __init__(
        self,
        constraint: BaseActionConstraint[T_State, T_Action, torch.Tensor] | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        BaseAgent.__init__(self, constraint)


class QLearningAgent(BaseAgent[T_State, T_Action, float]):
    def __init__(
        self,
        actions: set[T_Action],
        constraint: BaseActionConstraint[T_State, T_Action, float] | None = None,
    ) -> None:
        super().__init__(constraint)

        self._actions = actions
        self._q_values: dict[tuple[T_State, T_Action], float] = {}

    def dist(
        self,
        state: T_State,
    ) -> CategoricalDistribution[T_Action]:
        actions = list(self._actions)
        action_probs = softmax([self._q_values.get((state, action), 0.0) for action in actions])
        return CategoricalDistribution(action_probs, actions)

    def get_q_value(self, state: T_State, action: T_Action) -> float:
        return self._q_values.get((state, action), 0.0)

    def set_q_value(self, state: T_State, action: T_Action, value: float) -> None:
        self._q_values[(state, action)] = value

    @property
    def actions(self) -> set[T_Action]:
        return self._actions
