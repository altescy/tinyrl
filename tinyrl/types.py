from collections.abc import Hashable
from typing import TypeVar

T_State = TypeVar("T_State", bound=Hashable)
T_Action = TypeVar("T_Action", bound=Hashable)
