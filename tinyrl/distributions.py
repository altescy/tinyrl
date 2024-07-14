import math
from collections.abc import Hashable
from typing import Generic, TypeVar

import numpy
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


class CategoricalDistribution(BaseDistribution[_T, float]):
    def __init__(
        self,
        probs: numpy.ndarray | list[float],
        values: list[_T],
    ):
        self._probs = numpy.asarray(probs)
        self._values = values
        self._indices = numpy.arange(len(self._values))

    @property
    def probs(self) -> numpy.ndarray:
        return self._probs

    @property
    def values(self) -> list[_T]:
        return self._values

    def sample(self) -> _T:
        index = int(numpy.random.choice(self._indices, p=self._probs))
        return self._values[index]

    def prob(self, value: _T) -> float:
        return float(self._probs[self._values.index(value)])

    def log_prob(self, value: _T) -> float:
        return math.log(self.prob(value))

    def entropy(self) -> float:
        return float(-numpy.sum(self._probs * numpy.log(self._probs)))

    def mask(self, values: set[_T]) -> "CategoricalDistribution[_T]":
        indices = [self.values.index(value) for value in values]
        mask = numpy.zeros(len(self.probs))
        mask[indices] = 1.0
        probs = self._probs * mask
        probs /= self._probs.sum()
        return CategoricalDistribution(probs, self.values)


class BaseTorchDistribution(BaseDistribution[_T, torch.Tensor], Generic[_T]):
    def log_prob(self, value: _T) -> torch.Tensor:
        return self.prob(value).log()


class TorchCategoricalDistribution(BaseTorchDistribution[_T]):
    def __init__(
        self,
        probs: torch.Tensor,
        values: list[_T],
    ) -> None:
        self._probs = probs
        self._values = values

    @property
    def probs(self) -> torch.Tensor:
        return self._probs

    @property
    def values(self) -> list[_T]:
        return self._values

    def sample(self) -> _T:
        index = int(torch.multinomial(self.probs, 1).item())
        return self.values[index]

    def prob(self, value: _T) -> torch.Tensor:
        index = self.values.index(value)
        return self.probs[index]

    def entropy(self) -> torch.Tensor:
        return -torch.sum(self.probs * torch.log(self.probs))

    def mask(self, values: set[_T]) -> "TorchCategoricalDistribution[_T]":
        indices = [self.values.index(value) for value in values]
        mask = torch.zeros_like(self.probs)
        mask[indices] += 1.0
        probs = self._probs * mask
        probs = probs / probs.sum()
        return TorchCategoricalDistribution(probs, self.values)
