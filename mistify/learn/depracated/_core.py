from abc import abstractmethod, abstractclassmethod

from torch import nn
import torch
import typing
from functools import partial
from enum import Enum
from zenkai import XCriterion, Reduction, IO


class ToOptim(Enum):
    """
    Specify whehther to optimize x, theta or both
    """

    X = 'x'
    THETA = 'theta'
    BOTH = 'both'

    def x(self) -> bool:
        return self in (ToOptim.X, ToOptim.BOTH)

    def theta(self) -> bool:
        return self in (ToOptim.THETA, ToOptim.BOTH)


class MistifyLoss(XCriterion):
    """Loss to use in modules for Mistify
    """

    def reduce(self, y: torch.Tensor, reduction_override: str=None):

        reduction = reduction_override or self.reduction
        return Reduction[reduction].reduce(y)

        # if self.reduction == 'mean':
        #     return y.mean()
        # elif self.reduction == 'sum':
        #     return y.sum()
        # elif self.reduction == 'batchmean':
        #     return y.sum() / len(y)
        # elif self.reduction == 'none':
        #     return y
        
    @abstractmethod
    def forward(self, x: IO, y: IO, t: IO, reduction_override: float=None) -> torch.Tensor:
        raise NotImplementedError
    
    @classmethod
    def factory(cls, *args, **kwargs) -> 'MistifyLossFactory':
        return MistifyLossFactory(cls, *args, **kwargs)


class MistifyLossFactory(object):

    def __init__(self, module_cls: typing.Type[MistifyLoss], *args, **kwargs):

        self.factory = partial(module_cls, *args, **kwargs)

    def __call__(self, *args, **kwargs) -> 'MistifyLoss':

        return self.factory(*args, **kwargs)
