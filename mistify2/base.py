import torch
import typing
import torch.nn as nn


class Set(object):
    
    def __init__(self, data: torch.Tensor, is_batch: bool=None):

        if is_batch is None:
            is_batch = False if data.dim() <= 1 else True

        self._data = data
        if is_batch and data.dim() == 1:
            raise ValueError(f'Is batch cannot be set to true if data dimensionality is 1')
        
        if is_batch: 
            self._value_size = None if data.dim() == 2 else data.shape[1:-1]
        else:
            self._value_size = None if data.dim() == 1 else data.shape[1:-1]

        self._is_batch = is_batch
        self._n_values = data.shape[-1]
    
    @property
    def data(self) -> torch.Tensor:
        return self._data
    
    @property
    def is_batch(self) -> bool:
        return self._is_batch

    def dim(self) -> int:
        return self.data.dim()

    @property
    def n_samples(self) -> int:
        if self._is_batch:
            return self.data.size(0)
        return None
    # TODO: Consider whether to move some of these methods out of here


class SetParam(nn.Module):

    def __init__(self, set_: Set, requires_grad: bool=True):

        super().__init__()
        self._set = set_
        self._data = nn.parameter.Parameter(
            set_.data, requires_grad=requires_grad
        )

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, data: torch.Tensor):        
        self._set.data = data
        self._data = nn.parameter.Parameter(
            self._set.data.data, 
            requires_grad=self._data.requires_grad
        )

    @property
    def set(self) -> Set:
        return self._set
    
    @set.setter
    def set(self, set_: 'Set'):
        self.data = set_.data
    
    def __getitem__(self, idx) -> torch.Tensor:
        return self.data[idx]
