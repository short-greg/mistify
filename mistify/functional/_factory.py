from enum import Enum
from abc import abstractmethod, ABC
import torch
from functools import partial
import typing

from ._functional import (
    prod, bounded_min_on, prob_sum, prod_on,
    min_on, smooth_min_on, adamin_on,
    bounded_max_on, max_on, smooth_max_on, adamax_on,
    bounded_min, smooth_min, adamin, bounded_max,
    smooth_max, adamax

)
from ._ste import (
    max_on_ste, max_ste, min_ste, min_on_ste
)


ON_F = typing.Callable[[torch.Tensor, int, bool], torch.Tensor]
BETWEEN_F = typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

class InterOn(Enum):

    prod_on = prod_on
    bounded_min_on = bounded_min_on
    min_on = min_on
    smooth_min_on = smooth_min_on
    adamin_on = adamin_on
    min_on_ste = min_on_ste

    def f(self, **kwargs):
        return partial(self.value, **kwargs)


class UnionOn(Enum):

    bounded_max_on = bounded_max_on
    max_on = max_on
    smooth_max_on = smooth_max_on
    adamax_on = adamax_on
    max_on_ste = max_on_ste

    def f(self, **kwargs):
        return partial(self.value, **kwargs)


class Inter(Enum):

    bounded_min = bounded_min
    min = min
    smooth_min = smooth_min
    adamin = adamin
    prod = prod
    min_ste = min_ste

    def f(self, **kwargs):
        return partial(self.value, **kwargs)


class Union(Enum):

    bounded_max = bounded_max
    max = max
    smooth_max = smooth_max
    adamax = adamax
    prob_sum = prob_sum
    max_ste = max_ste

    def f(self, **kwargs):
        return partial(self.value, **kwargs)


class LogicalF(ABC):
    """A function for executing 
    """

    @abstractmethod
    def __call__(self, x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
        pass


class AndF(LogicalF):

    def __init__(self, union: BETWEEN_F, inter_on: ON_F):
        """Create a Functor for performing an And operation between
        a tensor and a weight

        Args:
            union (BETWEEN_F): The inner operation
            inter_on (ON_F): The aggregate operation
        """
        if isinstance(union, str):
            union = Union[union].value
        if isinstance(inter_on, str):
            inter_on = InterOn[inter_on].value

        self.union = union
        self.inter_on = inter_on

    def __call__(self, x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:

        return self.inter_on(
            self.union_between(x.unsqueeze(-1), w[None]), dim=dim
        )


class OrF(LogicalF):

    def __init__(self, inter: BETWEEN_F, union_on: ON_F):
        """Create a Functor for performing an Or operation between
        a tensor and a weight

        Args:
            inter (BETWEEN_F): The inner operation
            union_on (ON_F): The aggregate operation
        """
        if isinstance(inter, str):
            inter = Inter[inter].value
        if isinstance(union_on, str):
            union_on = UnionOn[union_on].value

        self.union_on = union_on
        self.inter = inter

    def __call__(self, x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:

        return self.union_on(
            self.inter(x.unsqueeze(-1), w[None]), dim=dim
        )
