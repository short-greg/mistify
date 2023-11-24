# # 1st party
# import typing

# # 3rd party
# import torch
# import torch.nn as nn

# # local
# from .._base import (
#     UnionOn, IntersectionOn, Complement
# )
# from .. import _base as base
# from ... import functional
# from . import _generate
# from ...utils import weight_func, EnumFactory


# class FuzzyComplement(Complement):

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         return 1 - m


# class FuzzyIntersectionOn(IntersectionOn):
#     """Intersect sets that comprise a fuzzy set on a dimension
#     """

#     def __init__(self, f: str='min', dim: int=-1, keepdim: bool=False):
#         """Intersect sets that comprise a fuzzy set on a specified dimension

#         Args:
#             f (str, optional): The function to use for intersection. Defaults to 'min'.
#             dim (int, optional): Dimension to intersect on. Defaults to -1.
#             keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

#         Raises:
#             ValueError: If the intersection function is invalid
#         """
#         super().__init__()
#         self._f = IntersectionOnEnum.factory(f)
#         self.dim = dim
#         self.keepdim = keepdim

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         return self._f(m, dim=self.dim, keepdim=self.keepdim)


# class FuzzyUnionOn(UnionOn):
#     """Union on a specific dimension
#     """

#     def __init__(self, f: str='max', dim: int=-1, keepdim: bool=False):
#         """

#         Args:
#             f (str, optional): The function to use for dimension. Defaults to 'max'.
#             dim (int, optional): The dimension to union on. Defaults to -1.
#             keepdim (bool, optional): Whether to keep the dimension. Defaults to False.
#         """
#         super().__init__()
#         self._f = UnionOnEnum.factory(f)
#         self.dim = dim
#         self.keepdim = keepdim

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         return self._f(m, dim=self.dim, keepdim=self.keepdim)


# class FuzzyOr(base.Or):
#     """
#     """

#     def __init__(
#         self, in_features: int, out_features: int, n_terms: int=None, 
#         f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="max_min",
#         wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp"
#     ):
#         """Create an or neuron for calculating selecting values and calculating the or of them

#         Args:
#             in_features (int): the number of in features
#             out_features (int): the number of out features
#             n_terms (int, optional): the number of terms. Defaults to None.
#             f (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The function for computing or. Defaults to "max_min".
#             wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The function to preprocess the weights with. Defaults to "clamp".
#         """
#         super().__init__()
#         if n_terms is not None:
#             shape = (n_terms, in_features, out_features)
#         else:
#             shape = (in_features,  out_features)
#         self.weight = nn.parameter.Parameter(_generate.positives(*shape))
#         self._f = OrEnum.factory(f)
#         self._wf = weight_func(wf)
#         self._n_terms = n_terms
#         self._in_features = in_features
#         self._out_features = out_features

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
        
#         weight = self._wf(self.weight)
#         return self._f(m, weight)


# class FuzzyAnd(base.Or):

#     def __init__(
#         self, in_features: int, out_features: int, n_terms: int=None, 
#         f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="minmax",
#         wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp"
#     ):
#         """Create an And neuron for calculating selecting values and calculating the "and" of them

#         Args:
#             in_features (int): The number of in features
#             out_features (int): The number of out features
#             n_terms (int, optional): The number of terms. Defaults to None.
#             f (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The and function. Defaults to "minmax".
#             wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The function to preprocess the weights with. Defaults to "clamp".
#         """
#         super().__init__()
#         if n_terms is not None:
#             shape = (n_terms, in_features, out_features)
#         else:
#             shape = (in_features,  out_features)
#         self.weight = nn.parameter.Parameter(_generate.negatives(*shape))
#         self._wf = weight_func(wf)
#         self._n_terms = n_terms
#         self._in_features = in_features
#         self._out_features = out_features
#         self._f = AndEnum.factory(f)
    
#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         """

#         Args:
#             m (torch.Tensor): 

#         Returns:
#             torch.Tensor: 
#         """
#         weight = self._wf(self.weight)
#         return self._f(m, weight)


# class FuzzyElse(base.Else):

#     def __init__(self, dim=-1, keepdim: bool = False):
#         """Calculate else along a certain dimension It calculates the sum of all the membership values along the dimension

#         Args:
#             dim (int, optional): _description_. Defaults to -1.
#             keepdim (bool, optional): _description_. Defaults to False.
#         """
#         super().__init__(dim, keepdim)

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         """Calculate the else for the fuzzy set

#         Args:
#             m (torch.Tensor): the membership value

#         Returns:
#             torch.Tensor: the else of the fuzzy set
#         """
#         return torch.clamp(1 - m.sum(self.dim, keepdim=self.keepdim), 0, 1)
