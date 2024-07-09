from enum import Enum
from abc import abstractmethod, ABC
import torch
from functools import partial
import typing

from ._join import (
    prob_inter, bounded_inter_on, prob_union, prob_inter_on,
    inter_on, smooth_inter_on, ada_inter_on,
    bounded_union_on, union_on, smooth_union_on, ada_union_on,
    bounded_inter, smooth_inter, ada_inter, bounded_union,
    smooth_union, ada_union, union, inter, prob_union_on

)

ON_F = typing.Callable[[torch.Tensor, int, bool], torch.Tensor]
BETWEEN_F = typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class InterOn(Enum):

    prob = (prob_inter_on,)
    bounded = bounded_inter_on,
    std = inter_on,
    smooth = smooth_inter_on,
    daa = ada_inter_on,

    def f(self, dim: int=None, keepdim: bool=None, **kwargs):
        if dim is not None:
            kwargs['dim'] = dim
        if keepdim is not None:
            kwargs['keepdim'] = keepdim
        return partial(self.value[0], **kwargs)

    def __call__(self, x: torch.Tensor, dim: int=-1, keepdim: bool=False, **kwargs):

        return self.value[0](x, dim, keepdim, **kwargs)


class UnionOn(Enum):

    bounded = bounded_union_on,
    std = union_on,
    smooth = smooth_union_on,
    ada = ada_union_on,
    prob = prob_union_on,

    def f(self, dim: int=None, keepdim: bool=None, **kwargs):
        if dim is not None:
            kwargs['dim'] = dim
        if keepdim is not None:
            kwargs['keepdim'] = keepdim
        return partial(self.value[0], **kwargs)

    def __call__(self, x: torch.Tensor, dim: int=-1, keepdim: bool=False, **kwargs):

        return self.value[0](x, dim, keepdim, **kwargs)


class Inter(Enum):

    bounded = bounded_inter,
    std = inter,
    smooth = smooth_inter,
    ada = ada_inter,
    prob = prob_inter,

    def f(self, **kwargs):
        return partial(self.value[0], **kwargs)

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):

        return self.value[0](x1, x2, **kwargs)


class Union(Enum):

    bounded = bounded_union,
    std = union,
    smooth = smooth_union,
    ada = ada_union,
    prob = prob_union,

    def f(self, **kwargs):
        return partial(self.value[0], **kwargs)

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):

        return self.value[0](x1, x2, **kwargs)


class LogicalF(ABC):
    """A function for executing 
    """

    @abstractmethod
    def __call__(self, x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
        pass


class AndF(LogicalF):

    def __init__(self, union: BETWEEN_F, inter_on: ON_F, pop: bool=False):
        """Create a Functor for performing an And operation between
        a tensor and a weight

        Args:
            union (BETWEEN_F): The inner operation
            inter_on (ON_F): The aggregate operation
        """
        if isinstance(union, str):
            union = Union[union]
        if isinstance(inter_on, str):
            inter_on = InterOn[inter_on]

        self.pop = pop
        self.union = union
        self.inter_on = inter_on

    def __call__(self, x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:

        if self.pop:
            w = w[:,None]
        else:
            w = w[None]
        return self.inter_on(
            self.union(x.unsqueeze(-1), w), dim=dim
        )

    def inner(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        
        if self.pop:
            x2 = x2[:,None]
        else:
            x2 = x2[None]
        return self.union(x.unsqueeze(-1), x2)


class OrF(LogicalF):

    def __init__(self, inter: BETWEEN_F, union_on: ON_F, pop: bool=False):
        """Create a Functor for performing an Or operation between
        a tensor and a weight

        Args:
            inter (BETWEEN_F): The inner operation
            union_on (ON_F): The aggregate operation
        """
        if isinstance(inter, str):
            inter = Inter[inter]
        if isinstance(union_on, str):
            union_on = UnionOn[union_on]

        self.pop = pop
        self.union_on = union_on
        self.inter = inter

    def __call__(self, x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:

        if self.pop:
            w = w[:,None]
        else:
            w = w[None]
        return self.union_on(
            self.inter(x.unsqueeze(-1), w), dim=dim
        )

    def inner(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        
        if self.pop:
            x2 = x2[:,None]
        else:
            x2 = x2[None]
        return self.inter(x.unsqueeze(-1), x2)
