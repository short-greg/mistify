"""
Functionality for crisp binary sets where 1 is True and 0 is False

"""
import typing
import torch
from torch import nn

from .._base import UnionOn, Else, IntersectionOn, Or, Complement, And
from .. import _functional
from .._base._utils import weight_func
from ._generate import positives
from . import _functional as binary_func


class BooleanComplement(Complement):

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return 1 - m


class BooleanIntersectionOn(IntersectionOn):

    def __init__(self, f: str='min', dim: int=-1, keepdim: bool=False):
        super().__init__()
        if f == 'min':
            self._f = _functional.min_on
        elif isinstance(f, typing.Callable):
            self._f = f
        else:
            raise ValueError(f'Invalid intersection')
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class BooleanUnionOn(UnionOn):

    def __init__(self, f: str='max', dim: int=-1, keepdim: bool=False):
        super().__init__()
        if f == 'max':
            self._f = _functional.max_on
        elif isinstance(f, typing.Callable):
            self._f = f
        else:
            raise ValueError(f'Invalid union')
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class BooleanAnd(And):

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
            self._f = _functional.minmax
        else:
            self._f = f

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        
        weight = self._wf(self.weight)
        return self._f(m, weight)


class BooleanOr(Or):

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
            self._f = _functional.maxmin
        else:
            self._f = f

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        
        weight = self._wf(self.weight)
        return self._f(m, weight)


class BooleanElse(Else):

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = x.max(dim=self.dim, keepdim=self.keepdim)[0]
        return (1 - y)
