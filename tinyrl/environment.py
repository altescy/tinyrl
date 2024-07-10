from __future__ import annotations

from typing import Generic, TypeVar

_T_State = TypeVar("_T_State")
_T_Action = TypeVar("_T_Action")


class BaseEnvironment(Generic[_T_State, _T_Action]):
    def reset(self) -> _T_State:
        raise NotImplementedError

    def step(self, action: _T_Action) -> tuple[_T_State, float, bool]:
        raise NotImplementedError
