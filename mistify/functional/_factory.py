from enum import Enum
from abc import abstractmethod, ABC
import torch
from functools import partial
import typing

# from ._join import (
#     prob_inter, bounded_inter_on, prob_union, prob_inter_on,
#     inter_on, smooth_inter_on, ada_inter_on,
#     bounded_union_on, union_on, smooth_union_on, ada_union_on,
#     bounded_inter, smooth_inter, ada_inter, bounded_union,
#     smooth_union, ada_union, union, inter

# )

# ON_F = typing.Callable[[torch.Tensor, int, bool], torch.Tensor]
# BETWEEN_F = typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# class InterOn(Enum):

#     prod_on = prob_inter_on
#     bounded_min_on = bounded_inter_on
#     min_on = inter_on
#     smooth_min_on = smooth_inter_on
#     adamin_on = ada_inter_on
#     min_on_ste = min_on

#     def f(self, **kwargs):
#         return partial(self.value, **kwargs)


# class UnionOn(Enum):

#     bounded_max_on = bounded_union_on
#     max_on = union_on
#     smooth_max_on = smooth_union_on
#     adamax_on = ada_union_on
#     max_on_ste = max_on

#     def f(self, **kwargs):
#         return partial(self.value, **kwargs)


# class Inter(Enum):

#     bounded_min = bounded_inter
#     min = min
#     smooth_min = smooth_inter
#     adamin = ada_inter
#     prod = prob_inter
#     min_ste = min

#     def f(self, **kwargs):
#         return partial(self.value, **kwargs)


# class Union(Enum):

#     bounded_max = bounded_union
#     max = max
#     smooth_max = smooth_union
#     adamax = ada_union
#     prob_sum = prob_union
#     max_ste = max

#     def f(self, **kwargs):
#         return partial(self.value, **kwargs)


# class LogicalF(ABC):
#     """A function for executing 
#     """

#     @abstractmethod
#     def __call__(self, x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
#         pass


# class AndF(LogicalF):

#     def __init__(self, union: BETWEEN_F, inter_on: ON_F):
#         """Create a Functor for performing an And operation between
#         a tensor and a weight

#         Args:
#             union (BETWEEN_F): The inner operation
#             inter_on (ON_F): The aggregate operation
#         """
#         if isinstance(union, str):
#             union = Union[union].value
#         if isinstance(inter_on, str):
#             inter_on = InterOn[inter_on].value

#         self.union = union
#         self.inter_on = inter_on

#     def __call__(self, x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:

#         return self.inter_on(
#             self.union_between(x.unsqueeze(-1), w[None]), dim=dim
#         )


# class OrF(LogicalF):

#     def __init__(self, inter: BETWEEN_F, union_on: ON_F):
#         """Create a Functor for performing an Or operation between
#         a tensor and a weight

#         Args:
#             inter (BETWEEN_F): The inner operation
#             union_on (ON_F): The aggregate operation
#         """
#         if isinstance(inter, str):
#             inter = Inter[inter].value
#         if isinstance(union_on, str):
#             union_on = UnionOn[union_on].value

#         self.union_on = union_on
#         self.inter = inter

#     def __call__(self, x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:

#         return self.union_on(
#             self.inter(x.unsqueeze(-1), w[None]), dim=dim
#         )
