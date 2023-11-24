from abc import abstractmethod

from torch import nn
import torch
from ..utils import join


class Exclusion(nn.Module):

    @abstractmethod
    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Inclusion(nn.Module):

    @abstractmethod
    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Join(nn.Module):

    def __init__(self, nn_module: nn.Module, dim: int=-1, unsqueeze_dim: int=None):
        super().__init__()
        self.nn_module = nn_module
        self.dim = dim
        self.unsqueeze_dim = unsqueeze_dim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        
        return join(m, self.nn_module, self.dim, self.unsqueeze_dim)


def get_comp_weight_size(in_features: int, out_features: int, in_variables: int=None):

    if in_variables is None or in_variables == 0:
        return torch.Size([in_features, out_features])
    return torch.Size([in_variables, in_features, out_features])


# class IntersectionOnEnum(EnumFactory):

#     min = functional.min_on
#     min_ada = functional.smooth_min_on
#     prod = functional.prod_on


# class UnionOnEnum(EnumFactory):

#     max = functional.min_on
#     max_ada = functional.smooth_max_on


# class AndEnum(EnumFactory):

#     min_max = functional.minmax
#     min_max_ada = functional.ada_minmax


# class OrEnum(EnumFactory):

#     max_min = functional.maxmin
#     maxmin_ada = functional.ada_minmax
#     max_prod = functional.maxprod



# class Or(nn.Module):

#     @abstractmethod
#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError


# class And(nn.Module):

#     def __init__(
#         self, in_features: int, out_features: int, n_terms: int=None, 
#         f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="minmax",
#         wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="binary"
#     ):
#         """ a BinaryAnd

#         Args:
#             in_features (int): _description_
#             out_features (int): _description_
#             n_terms (int, optional): _description_. Defaults to None.
#             f (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): _description_. Defaults to "minmax".
#             wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): _description_. Defaults to "binary".
#         """
#         super().__init__()
#         if n_terms is not None:
#             shape = (n_terms, in_features, out_features)
#         else:
#             shape = (in_features,  out_features)
#         self.weight = nn.parameter.Parameter(positives(*shape))
#         self._wf = weight_func(wf)
#         self._n_terms = n_terms
#         self._in_features = in_features
#         self._out_features = out_features
    
#         if f == "minmax":
#             self._f = functional.minmax
#         else:
#             self._f = f

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
        
#         weight = self._wf(self.weight)
#         return self._f(m, weight)

# class JunctionOn(nn.Module):

#     def __init__(self, dim: int=-1, keepdim: bool=False):
#         super().__init__()
#         self.dim = dim
#         self.keepdim = keepdim

#     @abstractmethod
#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError


# class UnionOn(JunctionOn):

#     def __init__(self, f: str='max', dim: int=-1, keepdim: bool=False):
#         """_summary_

#         Args:
#             f (str, optional): _description_. Defaults to 'max'.
#             dim (int, optional): _description_. Defaults to -1.
#             keepdim (bool, optional): _description_. Defaults to False.

#         Raises:
#             ValueError: _description_
#         """
#         super().__init__()
#         self.dim = dim
#         self.keepdim = keepdim
#         self._f = UnionOnEnum.factory(f)

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         return self._f(m, dim=self.dim, keepdim=self.keepdim)


# class IntersectionOn(JunctionOn):

#     def __init__(self, f: str='min', dim: int=-1, keepdim: bool=False):
#         """_summary_

#         Args:
#             f (str, optional): _description_. Defaults to 'min'.
#             dim (int, optional): _description_. Defaults to -1.
#             keepdim (bool, optional): _description_. Defaults to False.

#         Raises:
#             ValueError: _description_
#         """
#         super().__init__(dim, keepdim)
#         self._f = IntersectionOnEnum.factory(f)

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         return self._f(m, dim=self.dim, keepdim=self.keepdim)


# class Complement(nn.Module):

#     @abstractmethod
#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError

# class Else(nn.Module):
    
#     def __init__(self, dim=-1, keepdim: bool=False):
#         super().__init__()
#         self.dim = dim
#         self.keepdim = keepdim

#     @abstractmethod
#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError