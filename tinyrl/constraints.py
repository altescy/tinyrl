from typing import Generic

from tinyrl.distributions import BaseDistribution, CategoricalDistribution, TorchCategoricalDistribution
from tinyrl.environments import BaseEnvironment, ICategoricalActionEnvironment
from tinyrl.types import T_Action, T_Scalar, T_State


class BaseActionConstraint(Generic[T_State, T_Action, T_Scalar]):
    def constrain(
        self,
        env: BaseEnvironment[T_State, T_Action],
        state: T_State,
        dist: BaseDistribution[T_Action, T_Scalar],
    ) -> BaseDistribution[T_Action, T_Scalar]:
        raise NotImplementedError


class CategoricalActionConstraint(BaseActionConstraint[T_State, T_Action, T_Scalar]):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self._epsilon = epsilon

    def constrain(
        self,
        env: BaseEnvironment[T_State, T_Action],
        state: T_State,
        dist: BaseDistribution[T_Action, T_Scalar],
    ) -> BaseDistribution[T_Action, T_Scalar]:
        if not isinstance(env, ICategoricalActionEnvironment):
            raise ValueError(f"Unsupported environment type: {type(env)}")
        if not isinstance(dist, (CategoricalDistribution, TorchCategoricalDistribution)):
            raise ValueError(f"Unsupported distribution type: {type(dist)}")
        dist.mask(env.available_actions(state))
        return dist
