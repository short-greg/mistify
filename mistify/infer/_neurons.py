# 1st party
import typing
from abc import ABC
from functools import partial

# 3rd party
import torch
import torch.nn as nn

# local
from .. import functional
from ..utils import EnumFactory


WEIGHT_FACTORY = EnumFactory(
    sigmoid=torch.sigmoid,
    clamp=partial(torch.clamp, min=0, max=1),
    sign=torch.sign,
    boolean=lambda x: torch.clamp(torch.round(x), 0, 1)
)


class LogicalNeuron(nn.Module):

    F = EnumFactory()

    def __init__(self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="max_min",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp") -> None:
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

    def init_weight(self):

        self.weight.fill_(1.0)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): 

        Returns:
            torch.Tensor: 
        """
        weight = self._wf(self.weight)
        return self._f(m, weight)
    


class Or(LogicalNeuron):
    """
    """
    F = EnumFactory(
        max_min=functional.maxmin,
        maxmin_ada=functional.ada_maxmin,
        max_prod=functional.maxprod
    )

    def init_weight(self):

        self.weight.fill_(1.0)


class And(LogicalNeuron):

    F = EnumFactory(
        min_max = functional.minmax,
        min_max_ada = functional.ada_minmax
    )

    def init_weight(self):

        self.weight.fill_(0.0)
