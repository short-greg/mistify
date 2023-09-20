from abc import abstractmethod

from torch import nn
import torch
import typing


class MistifyLoss(nn.Module):
    """Loss to use in modules for Mistify
    """

    def __init__(self, module: nn.Module, reduction: str='mean'):
        super().__init__()
        self.reduction = reduction
        self._module = module
        if reduction not in ('mean', 'sum', 'batchmean', 'none'):
            raise ValueError(f"Reduction {reduction} is not a valid reduction")

    @property
    def module(self) -> nn.Module:
        return self._module 

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
    
    @abstractmethod
    def factory(cls, *args, **kwargs) -> typing.Callable[[nn.Module], 'MistifyLoss']:
        pass
