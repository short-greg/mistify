from abc import abstractmethod, abstractclassmethod

from torch import nn
import torch
import typing
from functools import partial
from enum import Enum


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


class MistifyLoss(nn.Module):
    """Loss to use in modules for Mistify
    """

    def __init__(self, reduction: str='mean'):
        super().__init__()
        self.reduction = reduction
        if reduction not in ('mean', 'sum', 'batchmean', 'none'):
            raise ValueError(f"Reduction {reduction} is not a valid reduction")

    def reduce(self, y: torch.Tensor):

        if self.reduction == 'mean':
            return y.mean()
        elif self.reduction == 'sum':
            return y.sum()
        elif self.reduction == 'batchmean':
            return y.sum() / len(y)
        elif self.reduction == 'none':
            return y
        
    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @classmethod
    def factory(cls, *args, **kwargs) -> 'MistifyLossFactory':
        return MistifyLossFactory(cls, *args, **kwargs)


class MistifyLossFactory(object):

    def __init__(self, module_cls: typing.Type[MistifyLoss], *args, **kwargs):

        self.factory = partial(module_cls, *args, **kwargs)

    def __call__(self, *args, **kwargs) -> 'MistifyLoss':

        return self.factory(*args, **kwargs)
