import torch
import torch.nn as nn
from abc import abstractmethod


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
        self._param = nn.parameter.Parameter(
            set_.data, requires_grad=requires_grad
        )

    @property
    def data(self) -> torch.Tensor:
        return self._param.data

    @data.setter
    def data(self, data: torch.Tensor):        
        self._set.data = data
        self._param.data = self._set.data

    @property
    def param(self) -> nn.parameter.Parameter:
        return self._param

    @property
    def set(self) -> Set:
        return self._set
    
    @set.setter
    def set(self, set_: 'Set'):
        self.data = set_.data
    
    def __getitem__(self, idx) -> torch.Tensor:
        return self.data[idx]

    def size(self, dim=None):
        if dim is None: self._set.data.size()
        return self._set.data.size(dim)

class CompositionBase(nn.Module):

    def __init__(
        self, in_features: int, out_features: int, 
        complement_inputs: bool=False, in_variables: int=None
    ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._complement_inputs = complement_inputs
        if complement_inputs:
            in_features *= 2
        self._multiple_variables = in_variables is not None
        self.weight = self.init_weight(in_features, out_features, in_variables)
    
    @abstractmethod
    def init_weight(self, in_features: int, out_features: int, in_variables: int=None) -> SetParam:
        pass

    def prepare_inputs(self, m: Set) -> torch.Tensor:
        if self._complement_inputs:
            return torch.cat([m.data, 1 - m.data], dim=-1)
        return m.data
    
    @property
    def to_complement(self) -> bool:
        return self._complement_inputs


class MistifyLoss(nn.Module):

    def __init__(self, reduction: str='mean'):
        super().__init__()
        self._reduction = reduction
        if reduction not in ('mean', 'sum', 'batchmean', 'none'):
            raise ValueError(f"Reduction {reduction} is not a valid reduction")        

    def reduce(self, y: torch.Tensor):

        if self._reduction == 'mean':
            return y.mean()
        elif self._reduction == 'sum':
            return y.sum()
        elif self._reduction == 'batchmean':
            return y.view(y.size(0), -1).mean(dim=1)
        elif self._reduction == 'none':
            return y
