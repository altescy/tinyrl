from collections.abc import Sequence
from typing import cast

import numpy


def softmax(x: numpy.ndarray | Sequence[float]) -> numpy.ndarray:
    x = numpy.asarray(x)
    if x.size == 0:
        return numpy.array([], dtype=x.dtype)
    exp_x = numpy.exp(x - numpy.max(x))
    return cast(numpy.ndarray, exp_x / exp_x.sum())
