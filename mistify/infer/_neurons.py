# 1st party
import typing
from functools import partial

# 3rd party
import torch
import torch.nn as nn

# local
from .. import _functional as functional
from .._functional import G
from ..utils import EnumFactory
from abc import abstractmethod
from ._ops import (
    UnionBase, UnionOnBase, InterBase, 
    InterOnBase, Union, Inter, InterOn, UnionOn,
    ProbInter, ProbUnion
)
from .._functional._factory import OrF, AndF


WEIGHT_FACTORY = EnumFactory(
    sigmoid=torch.sigmoid,
    clamp=partial(torch.clamp, min=0, max=1),
    sign=torch.sign,
    boolean=lambda x: torch.clamp(torch.round(x), 0, 1),
    none=lambda x: x
)
WEIGHT_FACTORY[None] = (lambda x: x)


def validate_weight_range(w: torch.Tensor, min_value: float, max_value: float) -> typing.Tuple[int, int]:
    """Calculate the number of weights outside the valid range

    Args:
        w (torch.Tensor): the weights
        min_value (float): Min value for weights
        max_value (float): Max value for weights

    Returns:
        typing.Tuple[int, int]: [lesser count, greater count]
    """
    greater = (w > max_value).float().sum()
    lesser = (w < min_value).float().sum()
    return lesser.item(), greater.item()


def validate_binary_weight(w: torch.Tensor, neg_value: float=0.0, pos_value: float=1.0) -> int:
    """_summary_

    Args:
        w (torch.Tensor): _description_
        neg_value (float, optional): _description_. Defaults to 0.0.
        pos_value (float, optional): _description_. Defaults to 1.0.

    Returns:
        int: _description_
    """
    return ((w != neg_value) & (w != pos_value)).float().sum().item()


class Or(nn.Module):
    """
    """

    def __init__(self, in_features: int, out_features: int, n_terms: int=None, 
        f: OrF=None, wf: 'WeightF'=None
    ) -> None:
        """Create an Logical neuron

        Args:
            in_features (int): The number of features into the neuron
            out_features (int): The number of features out of the neuron.
            n_terms (int, optional): The number of terms for the neuron. Defaults to None.
            f (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The function to use for the neuron. Defaults to "min_max".
            wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The weight function to use for the neuron. Defaults to "clamp".
        """
        super().__init__()
        
        if n_terms is not None:
            shape = (n_terms, in_features, out_features)
        else:
            shape = (in_features,  out_features)
        self.weight_base = nn.parameter.Parameter(torch.ones(*shape))
        self.wf = wf or NullWeightF()
        if isinstance(f, typing.Tuple):
            f = OrF(f[0], f[1])
        self.f = f or OrF('std', 'std')

        self._n_terms = n_terms
        self._in_features = in_features
        self._out_features = out_features
        self.init_weight()

    def w(self) -> torch.Tensor:
        return self.wf(self.weight_base)

    def init_weight(self, f: typing.Callable[[torch.Tensor], torch.Tensor]=None):

        if f is None:
            f = torch.rand_like
        self.weight_base.data = f(self.weight_base.data)
    
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership to get the output for

        Returns:
            torch.Tensor: The output of the membership
        """
        w = self.w()
        return self.f(m, w)


class And(nn.Module):
    """
    """

    def __init__(self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=None, wf: 'WeightF'=None,
        sub1: bool=False
    ) -> None:
        """Create an Logical neuron

        Args:
            in_features (int): The number of features into the neuron
            out_features (int): The number of features out of the neuron.
            n_terms (int, optional): The number of terms for the neuron. Defaults to None.
            f (typing.Union[str, typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional): The function to use for the neuron. Defaults to "min_max".
            wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The weight function to use for the neuron. Defaults to "clamp".
        """
        super().__init__()
        
        if n_terms is not None:
            shape = (n_terms, in_features, out_features)
        else:
            shape = (in_features,  out_features)
        self.weight_base = nn.parameter.Parameter(torch.ones(*shape))
        if isinstance(f, typing.Tuple):
            f = AndF(f[0], f[1])
        self.f = f or AndF('std', 'std')
        self.wf = wf or NullWeightF()
        self.sub1 = sub1

        self._n_terms = n_terms
        self._in_features = in_features
        self._out_features = out_features
        self.init_weight()

    def w(self) -> torch.Tensor:
        w = self.wf(self.weight_base)
        if self.sub1:
            return 1 - w
        return w

    def init_weight(self, f: typing.Callable[[torch.Tensor], torch.Tensor]=None):

        if f is None:
            f = torch.rand_like
        self.weight_base.data = f(self.weight_base.data)
    
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership to get the output for

        Returns:
            torch.Tensor: The output of the membership
        """
        w = self.w()
        return self.f(m, w)


class MaxMin(Or):

    def __init__(
        self, in_features: int, out_features: int, 
        n_terms: int = None, wf: 'WeightF'=None,
        g: G=None
    ) -> None:
        super().__init__(in_features, out_features, n_terms, (Inter(g=g), UnionOn(dim=-2, g=g)), wf)


class MaxProd(Or):

    def __init__(
        self, in_features: int, out_features: int, 
        n_terms: int = None, wf: 'WeightF'=None,
        g: G=None
    ) -> None:
        super().__init__(in_features, out_features, n_terms, (ProbInter(), UnionOn(dim=-2, g=g)), wf)


class MinMax(And):

    def __init__(
        self, in_features: int, out_features: int, 
        n_terms: int = None, wf: 'WeightF'=None,
        g: G=None
    ) -> None:
        super().__init__(in_features, out_features, n_terms, (Union(g=g), InterOn(dim=-2, g=g)), wf)


class MinSum(And):

    def __init__(
        self, in_features: int, out_features: int, 
        n_terms: int = None, wf: 'WeightF'=None,
        g: G=None
    ) -> None:
        super().__init__(in_features, out_features, n_terms, (ProbUnion(), InterOn(dim=-2, g=g)), wf)


class WeightF(nn.Module):

    @abstractmethod
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        pass


class ClampWeightF(WeightF):

    def __init__(self, min_val: float=0.0, max_val: float=1.0, g: G=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.g = g

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return functional.clamp(w, self.min_val, self.max_val, self.g)


class Sub1WeightF(WeightF):

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return (1 - w)


class BooleanWeightF(WeightF):

    def __init__(self, g: G=None):
        super().__init__()
        self.g = g

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return functional.binarize(w, self.g)


class SignWeightF(WeightF):

    def __init__(self, g: G=None):
        super().__init__()
        self.g = g

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return functional.signify(w, self.g)


class NullWeightF(WeightF):

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w


class Sub1WeightF(WeightF):

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return (1 - w)


class SigmoidWeightF(nn.Module):

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(w)
