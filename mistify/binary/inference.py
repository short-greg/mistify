"""
Functionality for crisp binary sets where 1 is True and 0 is False

"""
import typing
import torch
from torch import nn

from .._base import UnionOn, Else, IntersectionOn, Or, Complement
from .. import functional
from .._base.utils import weight_func
from .generate import positives
from . import functional as binary_func


class BinaryComplement(Complement):

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return 1 - m


class BinaryIntersectionOn(IntersectionOn):

    def __init__(self, f: str='min', dim: int=-1, keepdim: bool=False):
        super().__init__()
        if f == 'min':
            self._f = functional.min_on
        elif isinstance(f, typing.Callable):
            self._f = f
        else:
            raise ValueError(f'Invalid intersection')
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class BinaryUnionOn(UnionOn):

    def __init__(self, f: str='max', dim: int=-1, keepdim: bool=False):
        super().__init__()
        if f == 'max':
            self._f = functional.max_on
        elif isinstance(f, typing.Callable):
            self._f = f
        else:
            raise ValueError(f'Invalid union')
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class BinaryAnd(Or):

    def __init__(
        self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="minmax",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="binary"
    ):
        """ a BinaryAnd

        Args:
            in_features (int): _description_
            out_features (int): _description_
            n_terms (int, optional): _description_. Defaults to None.
            f (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): _description_. Defaults to "minmax".
            wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): _description_. Defaults to "binary".
        """
        super().__init__()
        if n_terms is not None:
            shape = (n_terms, in_features, out_features)
        else:
            shape = (in_features,  out_features)
        self.weight = nn.parameter.Parameter(positives(*shape))
        self._wf = weight_func(wf)
        self._n_terms = n_terms
        self._in_features = in_features
        self._out_features = out_features
    
        if f == "minmax":
            self._f = functional.minmax
        else:
            self._f = f

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        
        weight = self._wf(self.weight)
        return self._f(m, weight)


class BinaryOr(Or):

    def __init__(
        self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="maxmin",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp"
    ):
        super().__init__()
        if n_terms is not None:
            shape = (n_terms, in_features, out_features)
        else:
            shape = (in_features,  out_features)
        self.weight = nn.parameter.Parameter(positives(*shape))
        self._wf = weight_func(wf)
        self._n_terms = n_terms
        self._in_features = in_features
        self._out_features = out_features
    
        if f == "maxmin":
            self._f = functional.maxmin
        else:
            self._f = f

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        
        weight = self._wf(self.weight)
        return self._f(m, weight)


class BinaryElse(Else):

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = x.max(dim=self.dim, keepdim=self.keepdim)[0]
        return (1 - y)


# class BinaryComposition(CompositionBase):

#     def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
#         return positives(get_comp_weight_size(in_features, out_features, in_variables))

#     def forward(self, m: torch.Tensor):
#         return maxmin(m, self.weight).round()

#     def clamp_weights(self):
#         self.weight.data = self.weight.data.clamp(0, 1)
