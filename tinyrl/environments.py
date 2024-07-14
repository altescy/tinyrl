from __future__ import annotations

from typing import Generic, Protocol, TypeVar, runtime_checkable

from tinyrl.types import T_Action, T_State


class BaseEnvironment(Generic[T_State, T_Action]):
    def reset(self) -> T_State:
        raise NotImplementedError

    def step(self, action: T_Action) -> tuple[T_State, float, bool]:
        raise NotImplementedError


_T_State_contra = TypeVar("_T_State_contra", contravariant=True)


@runtime_checkable
class ICategoricalActionEnvironment(Protocol[_T_State_contra, T_Action]):
    def available_actions(self, state: _T_State_contra) -> set[T_Action]:
        ...
