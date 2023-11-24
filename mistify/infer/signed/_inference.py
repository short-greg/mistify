# """
# For crisp ternary sets where -1 is False, 0 is unknown, and 1 is true
# """

# # 1st party
# import typing

# # 3rd party
# import torch
# import torch.nn as nn

# # local
# from .._base import (
#     get_comp_weight_size, Or, And, Complement, Else,
#     UnionOn, IntersectionOn
# )
# from ...functional import maxmin, minmax
# from . import _functional as signed_func
# from ...utils import weight_func


# class SignedOr(Or):
#     """Calculate the relation between two ternary sets
#     """

#     def __init__(self, in_features: int, out_features: int, n_terms: int=None, 
#         wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp"):
        
#         if n_terms is not None:
#             shape = (n_terms, in_features, out_features)
#         else:
#             shape = (in_features,  out_features)
#         self._weight = nn.parameter.Parameter(signed_func.positives(*shape))
#         self._wf = weight_func(wf)

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         """ 
#         Args:
#             m (torch.Tensor): 

#         Returns:
#             torch.Tensor: Relationship between ternary set and the weights
#         """
#         weight = self._wf(self._weight)

#         return maxmin(m, weight[None])


# class SignedAnd(And):
#     """Calculate the relation between two ternary sets
#     """

#     def __init__(self, in_features: int, out_features: int, n_terms: int=None, 
#         wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp"):
        
#         if n_terms is not None:
#             shape = (n_terms, in_features, out_features)
#         else:
#             shape = (in_features,  out_features)
#         self._weight = nn.parameter.Parameter(signed_func.positives(*shape))
#         self._wf = weight_func(wf)

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         """ 
#         Args:
#             m (torch.Tensor): 

#         Returns:
#             torch.Tensor: Relationship between ternary set and the weights
#         """
#         weight = self._wf(self._weight)

#         return minmax(m, weight[None])


# class SignedComplement(Complement):

#     def complement(self, m: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             m (torch.Tensor): The membership tensor

#         Returns:
#             torch.Tensor: The complement of the ternary set
#         """
#         return -m


# class SignedElse(Else):

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
        
#         return -m.max(self.dim, keepdim=self.keepdim)[0]


# class SignedIntersectionOn(IntersectionOn):

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         return torch.min(m, dim=self.dim, keepdim=self.keepdim)[0]


# class SignedUnionOn(UnionOn):

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         return torch.max(m, dim=self.dim, keepdim=self.keepdim)
