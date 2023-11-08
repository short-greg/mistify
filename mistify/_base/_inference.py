from abc import abstractmethod

from torch import nn
import torch
from .functional import join


class Or(nn.Module):

    @abstractmethod
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class And(nn.Module):

    @abstractmethod
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class JunctionOn(nn.Module):

    def __init__(self, dim: int=-1, keepdim: bool=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    @abstractmethod
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UnionOn(JunctionOn):

    @abstractmethod
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IntersectionOn(JunctionOn):

    @abstractmethod
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Exclusion(nn.Module):

    @abstractmethod
    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Inclusion(nn.Module):

    @abstractmethod
    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Complement(nn.Module):

    @abstractmethod
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Join(nn.Module):

    def __init__(self, nn_module: nn.Module, dim: int=-1, unsqueeze_dim: int=None):
        super().__init__()
        self.nn_module = nn_module
        self.dim = dim
        self.unsqueeze_dim = unsqueeze_dim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        
        return join(m, self.nn_module, self.dim, self.unsqueeze_dim)


class Else(nn.Module):
    
    def __init__(self, dim=-1, keepdim: bool=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    @abstractmethod
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def get_comp_weight_size(in_features: int, out_features: int, in_variables: int=None):

    if in_variables is None or in_variables == 0:
        return torch.Size([in_features, out_features])
    return torch.Size([in_variables, in_features, out_features])


# class AndAgg(nn.Module):

#     def __init__(self, dim: int=-1):
#         super().__init__()
#         self.dim = dim

#     @abstractmethod
#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError


# class OrAgg(nn.Module):

#     def __init__(self, dim: int=-1):
#         super().__init__()
#         self.dim = dim

#     @abstractmethod
#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError



# class ComplementBase(nn.Module):
#     """Base complement class for calculating complement of a set
#     """
    
#     @abstractmethod
#     def complement(self, m: torch.Tensor) -> torch.Tensor:
#         """Take complemento f tensor

#         Args:
#             m (torch.Tensor): Tensor to take complement of

#         Returns:
#             torch.Tensor: Complemented tensor
#         """
#         raise NotImplementedError

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         """Take complement of tesor

#         Args:
#             m (torch.Tensor): 

#         Returns:
#             torch.Tensor: 
#         """
#         return self.complement(m)


# class CompositionBase(nn.Module):

#     def __init__(
#         self, in_features: int, out_features: int, in_variables: int=None
#     ):
#         """Base class for taking relations between two tensor

#         Args:
#             in_features (int): Number of input features (i.e. terms)
#             out_features (int): Number of outputs features (i.e. terms)
#             in_variables (int, optional): Number of linguistic variables in. Defaults to None.
#         """
#         super().__init__()
#         self._in_features = in_features
#         self._out_features = out_features
#         self._multiple_variables = in_variables is not None
#         self.weight = torch.nn.parameter.Parameter(
#             self.init_weight(in_features, out_features, in_variables)
#         )

#     @abstractmethod
#     def clamp_weights(self):
#         pass
    
#     @abstractmethod
#     def init_weight(self, in_features: int, out_features: int, in_variables: int=None) -> torch.Tensor:
#         pass


# class CatOp(nn.Module):
#     """Concatenate the output of an operation with the input
#     """

#     def __init__(self, operation: nn.Module, dim: int=-1):
#         """Concatenate the output of operation with the input

#         Args:
#             operation (nn.Module): the operation to concatenate with
#             dim (int, optional): the axis to concatenate on. Defaults to -1.
#         """
#         super().__init__()
#         self.operation = operation
#         self.dim = dim

#     def forward(self, m: torch.Tensor) -> torch.Tensor:

#         return torch.cat([m, self.operation(m)], dim=self.dim)