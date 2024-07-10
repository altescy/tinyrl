from collections.abc import Hashable
from typing import Generic, TypeVar

import torch

from tinyrl.types import T_Scalar

_T = TypeVar("_T", bound=Hashable)


class BaseDistribution(Generic[_T, T_Scalar]):
    def sample(self) -> _T:
        raise NotImplementedError

    def prob(self, value: _T) -> T_Scalar:
        raise NotImplementedError

    def log_prob(self, value: _T) -> T_Scalar:
        raise NotImplementedError

    def entropy(self) -> T_Scalar:
        raise NotImplementedError


class BaseTorchDistribution(BaseDistribution[_T, torch.Tensor], Generic[_T]):
    def log_prob(self, value: _T) -> torch.Tensor:
        return self.prob(value).log()


class TorchCategoricalDistribution(BaseTorchDistribution[_T]):
    def __init__(self, probs: torch.Tensor, values: list[_T]):
        self.probs = probs
        self.values = values

    def sample(self) -> _T:
        index = int(torch.multinomial(self.probs, 1).item())
        return self.values[index]

    def prob(self, value: _T) -> torch.Tensor:
        index = self.values.index(value)
        return self.probs[index]

    def entropy(self) -> torch.Tensor:
        return -torch.sum(self.probs * torch.log(self.probs))
