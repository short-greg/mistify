# 1st party
import typing
from abc import ABC
from functools import partial

# 3rd party
import torch
import torch.nn as nn

# local
from .. import _functional as functional
from .._functional import G
from ..utils import EnumFactory

from typing_extensions import Self


WEIGHT_FACTORY = EnumFactory(
    sigmoid=torch.sigmoid,
    clamp=partial(torch.clamp, min=0, max=1),
    sign=torch.sign,
    boolean=lambda x: torch.clamp(torch.round(x), 0, 1),
    none=lambda x: x
)
WEIGHT_FACTORY[None] = (lambda x: x)


class LogicalNeuron(nn.Module):
    """A Logical neuron implements a logical function such as And or Or
    """

    F = EnumFactory()

    def __init__(self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]=None,
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp"
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
        self.f = self.F.f(f)
        self.wf = WEIGHT_FACTORY.f(wf)
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
        # greater = (w > 1.0).float().sum()
        # lesser = (w < 0.0).float().sum()
        # if greater > 0 or lesser > 0:
        #     print(f'Neuron: Outside bounds counts {greater} {lesser}')
        return self.f(m, w)


class Or(LogicalNeuron):
    """An Or neuron implements an or function where the input is intersected with the 
    weights and the union is used for the aggregation of those outputs.
    """

    F = EnumFactory(
        max_min=functional.max_min,
        ada_max_min=functional.ada_max_min,
        max_prod=functional.max_prod,

    )

    def __init__(self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="max_min",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]=None,

    ) -> None:
        """Create an And neuron

        Args:
            in_features (int): The number of features into the neuron
            out_features (int): The number of features out of the neuron.
            n_terms (int, optional): The number of terms for the neuron. Defaults to None.
            f (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The function to use for the neuron. Defaults to "min_max".
            wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The function to preprocess the weights with. Defaults to "clamp".
        """
        wf = WEIGHT_FACTORY.f(wf)

        super().__init__(
            in_features=in_features, out_features=out_features, n_terms=n_terms,
            f=functional.max_min if f is None else f, wf=wf
        )


class And(LogicalNeuron):
    """An And neuron implements an and function where the input is unioned with the 
    weights and the intersection is used for the aggregation of those outputs.
    """
    F = EnumFactory(
        min_sum=functional.min_sum,
        ada_min_max=functional.ada_min_max,
        min_max=functional.min_max,
        
    )

    def __init__(self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]=None,
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]=None,
        sub1: bool=True

    ) -> None:
        """Create an And neuron

        Args:
            in_features (int): The number of features into the neuron
            out_features (int): The number of features out of the neuron.
            n_terms (int, optional): The number of terms for the neuron. Defaults to None.
            f (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The function to use for the neuron. Defaults to "min_max".
            wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The function to preprocess the weights with. Defaults to "clamp".
        """
        wf = WEIGHT_FACTORY.f(wf)
        if sub1 and wf is not None:
            wf_ = lambda w: wf(1 - w)
        elif sub1:
            wf_ = lambda w: 1 - w
        else:
            wf_ = wf

        super().__init__(
            in_features=in_features, out_features=out_features, n_terms=n_terms,
            f=functional.min_max if f is None else f, wf=wf_
        )


class MaxMin(Or):

    def __init__(self, in_features: int, out_features: int, n_terms: int = None, wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        super().__init__(in_features, out_features, n_terms, 'max_min', wf)


class MaxProd(Or):

    def __init__(self, in_features: int, out_features: int, n_terms: int = None, wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        super().__init__(in_features, out_features, n_terms, 'max_prod', wf)


class MinMax(And):

    def __init__(self, in_features: int, out_features: int, n_terms: int = None, wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        super().__init__(in_features, out_features, n_terms, 'min_max', wf)


class MinSum(And):

    def __init__(self, in_features: int, out_features: int, n_terms: int = None, wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        super().__init__(in_features, out_features, n_terms, 'min_sum', wf)


class BuildLogical(object):

    def sigmoid_wf(self) -> Self:

        self.wf = torch.sigmoid
        return self
    
    def sign_wf(self, g: G=None) -> Self:

        self.wf = partial(functional.signify, g=g)
        return self
    
    def no_wf(self) -> Self:

        self.wf = lambda x: x
        return self

    def boolean_wf(self, g: G=None) -> Self:

        self.wf = partial(functional.binarize, g=g)
        return self

    def clamp_wf(self, min_val: float=0.0, max_val: float=1.0, g: G=None) -> Self:

        self.wf = partial(functional.clamp, min_val=min_val, max_val=max_val, g=g)
        return self


class BuildOr(BuildLogical):

    def __init__(self) -> None:
        super().__init__()
        self.inter()
        self.union_on()

    def sub1(self, sub1: bool=True) -> Self:
        self.sub1 = sub1
        return self
    
    def inter(self, g: G=None) -> Self:

        self.interf = partial(functional.inter, g=g)
        return self

    def prob_inter(self) -> Self:

        self.interf = functional.prob_inter
        return self

    def smooth_inter(self, a: float=10.0) -> Self:

        self.interf = partial(functional.smooth_inter, a=a)
        return self

    def ada_inter(self) -> Self:

        self.interf = functional.ada_inter
        return self

    def bounded_inter(self, g: G=None) -> Self:

        self.interf = partial(functional.bounded_inter, g=g)
        return self

    def union_on(self, g: G=None) -> Self:

        self.union_onf = partial(functional.union_on, g=g)
        return self

    def prob_union_on(self) -> Self:

        self.union_onf = functional.prob_union_on
        return self

    def smooth_union_on(self, a: float=10.0) -> Self:

        self.union_onf = partial(functional.smooth_union_on, a=a)
        return self

    def ada_union_on(self) -> Self:

        self.union_onf = functional.ada_union_on
        return self

    def bounded_union_on(self, g: G=None) -> Self:

        self.union_onf = partial(functional.bounded_union_on, g=g)
        return self

    def lambda_union_on(self, union_on_f) -> Self:
        self.union_onf = union_on_f
        return self

    def lambda_inter(self, inter_f) -> Self:
        self.interf = inter_f
        return self

    def build(self, in_features: int, out_features: int, n_terms: int=None) -> 'Or':

        return Or(
            in_features, out_features, n_terms, functional.OrF(self.interf, self.union_onf), self.wf
        )


class BuildAnd(BuildLogical):

    def __init__(self) -> None:
        """Build an and function
        """
        super().__init__()
        self.union()
        self.inter_on()

    def sub1(self, sub1: bool=True) -> Self:
        self.sub1 = sub1
        return self
    
    def union(self, g: G=None) -> Self:

        self.unionf = partial(functional.union, g=g)
        return self

    def prob_union(self) -> Self:

        self.unionf = functional.prob_union
        return self

    def smooth_union(self, a: float=10.0) -> Self:

        self.unionf = partial(functional.smooth_union, a=a)
        return self

    def ada_union(self) -> Self:

        self.unionf = functional.ada_union
        return self

    def bounded_union(self) -> Self:

        self.unionf = partial(functional.bounded_union, g=G)
        return self

    def inter_on(self, g: G=None) -> Self:

        self.inter_onf = partial(functional.inter_on, g=g)
        return self

    def prob_inter_on(self) -> Self:

        self.inter_onf = functional.prob_inter_on
        return self

    def smooth_inter_on(self, a: float=10.0) -> Self:

        self.inter_onf = partial(functional.smooth_inter_on, a=a)
        return self

    def ada_inter_on(self) -> Self:

        self.inter_onf = functional.ada_inter_on
        return self

    def bounded_inter_on(self, g: G=None) -> Self:

        self.inter_onf = partial(functional.bounded_inter_on, g=g)
        return self

    def lambda_inter_on(self, inter_on_f) -> Self:
        self.inter_onf = inter_on_f

    def lambda_union(self, union_f) -> Self:
        self.unionf = union_f

    def build(self, in_features: int, out_features: int, n_terms: int=None) -> 'And':

        return And(
            in_features, out_features, n_terms, functional.AndF(self.unionf, self.inter_onf), self.wf
        )


# class AndF(object):

#     def __init__(self, wf, union, inter_on, sub1=True):

#         self.wf = wf
#         self.union = union
#         self.inter_on = inter_on
#         self.sub1 = sub1

#     def __call__(self, in_features: int, out_features: int, n_terms: int=None) -> 'And':

#         return And(
#             in_features, out_features, n_terms, self.union, self.inter_on, self.sub1
#         )


# class OrF(object):

#     def __init__(self, wf, inter, union_on):

#         self.wf = wf
#         self.inter = inter
#         self.union_on = union_on

#     def __call__(self, in_features: int, out_features: int, n_terms: int=None) -> 'Or':

#         return Or(
#             in_features, out_features, n_terms, self.inter, self.union_on
#         )
