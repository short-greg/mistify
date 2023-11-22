from abc import abstractmethod

from torch import nn
import torch
from ..utils import join


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
