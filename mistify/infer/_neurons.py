# 1st party
import typing
from abc import abstractmethod
from functools import partial

# 3rd party
import torch
import torch.nn as nn

# local
from .. import functional
from ..utils import EnumFactory
from . import signed
from . import fuzzy
from . import boolean


WEIGHT_FACTORY = EnumFactory(
    sigmoid=torch.sigmoid,
    clamp=partial(torch.clamp, min=0, max=1),
    sign=torch.sign,
    boolean=lambda x: torch.clamp(torch.round(x), 0, 1)
)

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
        self._wf = WEIGHT_FACTORY.factory(wf)
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
        self._wf = WEIGHT_FACTORY.factory(wf)
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
