from __future__ import annotations

from typing import Generic

from tinyrl.types import T_Action, T_State


class BaseEnvironment(Generic[T_State, T_Action]):
    def reset(self) -> T_State:
        raise NotImplementedError

    def step(self, action: T_Action) -> tuple[T_State, float, bool]:
        raise NotImplementedError
