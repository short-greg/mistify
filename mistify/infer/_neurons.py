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
WEIGHT_FACTORY[None] = (lambda x: x)


class LogicalNeuron(nn.Module):
    """A Logical neuron implements a logical function such as And or Or
    """

    F = EnumFactory()

    def __init__(self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="max_min",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp"
    ) -> None:
        """Create an Logical neuron

        Args:
            in_features (int): The number of features into the neuron
            out_features (int): The number of features out of the neuron.
            n_terms (int, optional): The number of terms for the neuron. Defaults to None.
            f (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): _description_. Defaults to "min_max".
            wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): _description_. Defaults to "clamp".
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
        self.init_weight()

    def init_weight(self, f: typing.Callable[[torch.Tensor], torch.Tensor]=None):

        if f is None:
            f = torch.rand_like
        self.weight.data = f(self.weight.data)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): 

        Returns:
            torch.Tensor: 
        """
        greater = (self.weight > 1.0).float().sum()
        lesser = (self.weight < 0.0).float().sum()
        if greater > 0 or lesser > 0:
            print(f'Neuron: Outside bounds counts {greater} {lesser}')
        weight = self._wf(self.weight)
        return self._f(m, weight)


class Or(LogicalNeuron):
    """An Or neuron implements an or function where the input is intersected with the 
    weights and the union is used for the aggregation of those outputs.
    """

    F = EnumFactory(
        max_min=functional.maxmin,
        max_min_ada=functional.ada_maxmin,
        max_prod=functional.maxprod
    )


class And(LogicalNeuron):
    """An And neuron implements an and function where the input is unioned with the 
    weights and the intersection is used for the aggregation of those outputs.
    """

    F = EnumFactory(
        min_max = functional.minmax,
        min_max_ada = functional.ada_minmax
    )

    def __init__(self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="min_max",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp",
        sub1: bool=True

    ) -> None:
        """Create an And neuron

        Args:
            in_features (int): The number of features into the neuron
            out_features (int): The number of features out of the neuron.
            n_terms (int, optional): The number of terms for the neuron. Defaults to None.
            f (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): _description_. Defaults to "min_max".
            wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): _description_. Defaults to "clamp".
        """
        wf = WEIGHT_FACTORY.factory(wf)
        if sub1 and wf is not None:
            wf_ = lambda w: wf(1 - w)
        elif sub1:
            wf_ = lambda w: 1 - w
        else:
            wf_ = wf

        super().__init__(
            in_features=in_features, out_features=out_features, n_terms=n_terms,
            f=f, wf=wf_
        )
