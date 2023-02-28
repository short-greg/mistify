import torch
import torch.nn as nn
from abc import abstractmethod
from enum import Enum


class ToOptim(Enum):

    X = 'x'
    THETA = 'theta'
    BOTH = 'both'

    def x(self) -> bool:
        return self in (ToOptim.X, ToOptim.BOTH)

    def theta(self) -> bool:
        return self in (ToOptim.THETA, ToOptim.BOTH)


class MistifyLoss(nn.Module):

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


def get_comp_weight_size(in_features: int, out_features: int, in_variables: int=None):

    if in_variables is None or in_variables == 0:
        return torch.Size([in_features, out_features])
    return torch.Size([in_variables, in_features, out_features])


def smooth_max(x: torch.Tensor, x2: torch.Tensor, a: float) -> torch.Tensor:
    z1 = ((x + 1) ** a).detach()
    z2 = ((x2 + 1) ** a).detach()
    return (x * z1 + x2 * z2) / (z1 + z2)

def smooth_max_on(x: torch.Tensor, dim: int, a: float) -> torch.Tensor:
    z = ((x + 1) ** a).detach()
    return (x * z).sum(dim=dim) / z.sum(dim=dim)


def smooth_min(x: torch.Tensor, x2: torch.Tensor, a: float) -> torch.Tensor:
    return smooth_max(x, x2, -a)


def smooth_min_on(x: torch.Tensor, dim: int, a: float) -> torch.Tensor:
    return smooth_max_on(x, dim, -a)


def adamax(x: torch.Tensor, x2: torch.Tensor):
    q = torch.clamp(-69 / torch.log(torch.max(x, x2)), max=1000, min=-1000).detach()  
    return ((x ** q + x2 ** q) / 2) ** (1 / q)


def adamin(x: torch.Tensor, x2: torch.Tensor):
    q = torch.clamp(69 / torch.log(torch.min(x, x2)).detach(), max=1000, min=-1000)
    result = ((x ** q + x2 ** q) / 2) ** (1 / q)
    return result


def adamax_on(x: torch.Tensor, dim: int):

    q = torch.clamp(-69 / torch.log(torch.max(x, dim=dim)[0]).detach(), max=1000, min=-1000)
    return (torch.sum(x ** q.unsqueeze(dim), dim=dim) / x.size(dim)) ** (1 / q)


def adamin_on(x: torch.Tensor, dim: int):

    q = torch.clamp(69 / torch.log(torch.min(x, dim=dim)[0]).detach(), max=1000, min=-1000)
    return (torch.sum(x ** q.unsqueeze(dim), dim=dim) / x.size(dim)) ** (1 / q)


def maxmin(x: torch.Tensor, w: torch.Tensor, dim=-2):
    return torch.max(torch.min(x.unsqueeze(-1), w[None]), dim=dim)[0]


def minmax(x: torch.Tensor, w: torch.Tensor, dim=-2):
    return torch.min(torch.max(x.unsqueeze(-1), w[None]), dim=dim)[0]


def maxprod(x: torch.Tensor, w: torch.Tensor, dim=-2):
    return torch.max(x.unsqueeze(-1) * w[None], dim=dim)[0]


class ComplementBase(nn.Module):

    def __init__(self, concatenate_dim: int=None):
        super().__init__()
        self.concatenate_dim = concatenate_dim

    def postprocess(self, m: torch.Tensor, m_out: torch.Tensor):
        if self.concatenate_dim is None:
            return
        
        return torch.cat(
            [m, m_out], dim=self.concatenate_dim
        )
    
    @abstractmethod
    def complement(self, m: torch.Tensor):
        raise NotImplementedError

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.postprocess(m, self.complement(m))


class CompositionBase(nn.Module):

    def __init__(
        self, in_features: int, out_features: int, in_variables: int=None
    ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        # self._complement_inputs = complement_inputs
        # if complement_inputs:
        #     in_features = in_features * 2
        self._multiple_variables = in_variables is not None
        self.weight = torch.nn.parameter.Parameter(
            self.init_weight(in_features, out_features, in_variables)
        )
    
    @abstractmethod
    def init_weight(self, in_features: int, out_features: int, in_variables: int=None) -> torch.Tensor:
        pass

    # def prepare_inputs(self, m: torch.Tensor) -> torch.Tensor:
    #     if self._complement_inputs:
    #         return torch.cat([m, 1 - m], dim=-1)
        
    #     return m
    
    # @property
    # def to_complement(self) -> bool:
    #     return self._complement_inputs



# class Set(object):
    
#     def __init__(self, data: torch.Tensor, is_batch: bool=None):

#         if is_batch is None:
#             is_batch = False if data.dim() <= 1 else True

#         self._data = data
#         if is_batch and data.dim() == 1:
#             raise ValueError(f'Is batch cannot be set to true if data dimensionality is 1')
        
#         if is_batch: 
#             self._value_size = None if data.dim() == 2 else data.shape[1:-1]
#         else:
#             self._value_size = None if data.dim() == 1 else data.shape[1:-1]

#         self._is_batch = is_batch
#         self._n_values = data.shape[-1]
    
#     @property
#     def data(self) -> torch.Tensor:
#         return self._data
    
#     @property
#     def is_batch(self) -> bool:
#         return self._is_batch

#     def dim(self) -> int:
#         return self.data.dim()

#     @property
#     def n_samples(self) -> int:
#         if self._is_batch:
#             return self.data.size(0)
#         return None
#     # TODO: Consider whether to move some of these methods out of here

# class SetParam(nn.Module):

#     def __init__(self, set_: Set, requires_grad: bool=True):

#         super().__init__()
#         self._set = set_
#         self._param = nn.parameter.Parameter(
#             set_.data, requires_grad=requires_grad
#         )

#     @property
#     def data(self) -> torch.Tensor:
#         return self._param.data

#     @data.setter
#     def data(self, data: torch.Tensor):        
#         self._set.data = data
#         self._param.data = self._set.data

#     @property
#     def param(self) -> nn.parameter.Parameter:
#         return self._param

#     @property
#     def set(self) -> Set:
#         return self._set
    
#     @set.setter
#     def set(self, set_: 'Set'):
#         self.data = set_.data
    
#     def __getitem__(self, idx) -> torch.Tensor:
#         return self.data[idx]

#     def size(self, dim=None):
#         if dim is None: self._set.data.size()
#         return self._set.data.size(dim)



