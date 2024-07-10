from collections.abc import Hashable
from typing import TypeVar

import torch

T_Scalar = TypeVar("T_Scalar", float, torch.Tensor)
T_State = TypeVar("T_State", bound=Hashable)
T_Action = TypeVar("T_Action", bound=Hashable)
