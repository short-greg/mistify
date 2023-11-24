# 1st party
import typing
from enum import Enum
from abc import abstractmethod

# 3rd party
import torch
import torch.nn as nn

# local
from .. import functional
from ..utils import weight_func, EnumFactory
from . import signed
from . import fuzzy
from . import boolean



# def get_comp_weight_size(in_features: int, out_features: int, in_variables: int=None):

#     if in_variables is None or in_variables == 0:
#         return torch.Size([in_features, out_features])
#     return torch.Size([in_variables, in_features, out_features])


class IntersectionOn(nn.Module):
    """Intersect sets that comprise a fuzzy set on a dimension
    """
    F = EnumFactory(
        min=functional.min_on,
        min_ada=functional.smooth_min_on,
        prod=functional.prod_on
    )

    def __init__(self, f: str='min', dim: int=-1, keepdim: bool=False):
        """Intersect sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for intersection. Defaults to 'min'.
            dim (int, optional): Dimension to intersect on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the intersection function is invalid
        """
        super().__init__()
        self._f = self.F.factory(f)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class UnionOn(nn.Module):
    """Union on a specific dimension
    """
    F = EnumFactory(
        max = functional.min_on,
        max_ada = functional.smooth_max_on
    )

    def __init__(self, f: str='max', dim: int=-1, keepdim: bool=False):
        """

        Args:
            f (str, optional): The function to use for dimension. Defaults to 'max'.
            dim (int, optional): The dimension to union on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dimension. Defaults to False.
        """
        super().__init__()
        self._f = self.F.factory(f)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class Or(nn.Module):
    """
    """

    F = EnumFactory(
        max_min=functional.maxmin,
        maxmin_ada=functional.ada_minmax,
        max_prod=functional.maxprod
    )

    def __init__(
        self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="max_min",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp"
    ):
        """Create an or neuron for calculating selecting values and calculating the or of them

        Args:
            in_features (int): the number of in features
            out_features (int): the number of out features
            n_terms (int, optional): the number of terms. Defaults to None.
            f (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The function for computing or. Defaults to "max_min".
            wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The function to preprocess the weights with. Defaults to "clamp".
        """
        super().__init__()
        if n_terms is not None:
            shape = (n_terms, in_features, out_features)
        else:
            shape = (in_features,  out_features)
        self.weight = nn.parameter.Parameter(torch.ones(*shape))
        self._f = self.F.factory(f)
        self._wf = weight_func(wf)
        self._n_terms = n_terms
        self._in_features = in_features
        self._out_features = out_features

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): 

        Returns:
            torch.Tensor: 
        """
        
        weight = self._wf(self.weight)
        return self._f(m, weight)


class And(nn.Module):

    F = EnumFactory(
        min_max = functional.minmax,
        min_max_ada = functional.ada_minmax
    )

    def __init__(
        self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="min_max",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp"
    ):
        """Create an And neuron for calculating selecting values and calculating the "and" of them

        Args:
            in_features (int): The number of in features
            out_features (int): The number of out features
            n_terms (int, optional): The number of terms. Defaults to None.
            f (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The and function. Defaults to "minmax".
            wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The function to preprocess the weights with. Defaults to "clamp".
        """
        super().__init__()
        if n_terms is not None:
            shape = (n_terms, in_features, out_features)
        else:
            shape = (in_features,  out_features)
        self.weight = nn.parameter.Parameter(torch.zeros(*shape))
        self._wf = weight_func(wf)
        self._n_terms = n_terms
        self._in_features = in_features
        self._out_features = out_features
        print(list(self.F.keys()))
        self._f = self.F.factory(f)
    
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): 

        Returns:
            torch.Tensor: 
        """
        weight = self._wf(self.weight)
        return self._f(m, weight)


class Else(nn.Module):

    F = EnumFactory(
        signed= signed.else_,
        fuzzy= fuzzy.else_,
        boolean= boolean.else_
    )
    def __init__(self, f: typing.Callable, dim=-1, keepdim: bool = False):
        """Calculate else along a certain dimension It calculates the sum of all the membership values along the dimension

        Args:
            dim (int, optional): _description_. Defaults to -1.
            keepdim (bool, optional): _description_. Defaults to False.
        """
        super().__init__(dim, keepdim)
        self._f = self.F.factory(f)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Calculate the else for the fuzzy set

        Args:
            m (torch.Tensor): the membership value

        Returns:
            torch.Tensor: the else of the fuzzy set
        """
        return self._f(m)


class Complement(nn.Module):

    F = EnumFactory(
        signed = signed.complement,
        boolean = boolean.complement,
        fuzzy= fuzzy.complement
    )

    def __init__(self, f: typing.Callable='boolean'):
        """Calculate else along a certain dimension It calculates the sum of all the membership values along the dimension

        Args:
            dim (int, optional): _description_. Defaults to -1.
            keepdim (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self._f = self.F.factory(f)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Calculate the complement for the set

        Args:
            m (torch.Tensor): the membership value

        Returns:
            torch.Tensor: the complement of the set
        """
        return self._f(m)


class Exclusion(nn.Module):

    @abstractmethod
    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Inclusion(nn.Module):

    @abstractmethod
    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
