# 1st party
import typing
from abc import abstractmethod
import math
from functools import partial


# 3rd party
import torch
import torch.nn as nn
from torch.nn import functional as nn_func

# local
from ._core import CompositionBase, ComplementBase, maxmin, minmax, maxprod, MistifyLoss, ToOptim, get_comp_weight_size

def unify(m: torch.Tensor, m2: torch.Tensor):
    return torch.max(m, m2)


def differ(m: torch.Tensor, m2: torch.Tensor):
    return (m - m2).clamp(0.0, 1.0)


def positives(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    return torch.ones(*size, dtype=dtype, device=device)


def negatives(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    return torch.zeros(*size, dtype=dtype, device=device)


def intersect(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    return torch.min(m1, m2)

def intersect_on(m: torch.Tensor, dim: int=-1) -> torch.Tensor:
    return torch.min(m, dim=dim)[0]

def unify_on(m: torch.Tensor, dim: int=-1):
    return torch.max(m, dim=dim)[0]


def inclusion(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return (1 - m2) + torch.min(m1, m2)

def exclusion(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return (1 - m1) + torch.min(m1, m2)


def rand(*size: int,  dtype=torch.float32, device='cpu'):

    return (torch.rand(*size, device=device, dtype=dtype))


class FuzzyComposition(CompositionBase):

    @abstractmethod
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def clamp(self):
        self.weight.data = torch.clamp(self.weight.data, 0, 1)


class MaxMin(FuzzyComposition):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
        return positives(get_comp_weight_size(in_features, out_features, in_variables))
    
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        # assume inputs are binary
        # binarize the weights
        return maxmin(m, self.weight)


class MaxProd(FuzzyComposition):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
        return positives(get_comp_weight_size(in_features, out_features, in_variables))

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        # assume inputs are binary
        # binarize the weights
        return maxprod(m, self.weight)


class MinMax(FuzzyComposition):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
        return negatives(get_comp_weight_size(in_features, out_features, in_variables))
    
    def forward(self, m: torch.Tensor):
        # assume inputs are binary
        # binarize the weights
        return minmax(m, self.weight)


class Inner(nn.Module):

    def __init__(self, f: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__()
        self._f = f
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._f(x.unsqueeze(-1), y[None])


class Outer(nn.Module):
        
    def __init__(self, f: typing.Callable[[torch.Tensor], torch.Tensor], agg_dim: int=-2, idx: int=None):
        super().__init__()
        if idx is None:
            self._f =  partial(f, dim=agg_dim)
        if idx is not None:
            self._f = lambda x: f(x, dim=agg_dim)[idx]
        self._idx = idx
        # change from being manually defined
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._f(x)


class FuzzyComplement(ComplementBase):

    def complement(self, m: torch.Tensor):
        return 1 - m


class FuzzyRelation(FuzzyComposition):

    def __init__(
        self, in_features: int, out_features: int, 
        in_variables: int=None, 
        inner: typing.Union[Inner, typing.Callable]=None, 
        outer: typing.Union[Outer, typing.Callable]=None
    ):
        super().__init__(in_features, out_features, in_variables)
        inner = inner or Inner(torch.min)
        outer = outer or Outer(torch.max, idx=0)
        if not isinstance(inner, Inner):
            inner = Inner(inner)
        if not isinstance(outer, Outer):
            outer = Outer(outer)
        self.inner = inner
        self.outer = outer

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
        return positives(get_comp_weight_size(in_features, out_features, in_variables))
                         
    def forward(self, m: torch.Tensor):
        return self.outer(self.inner(m, self.weight))


class IntersectOn(nn.Module):

    def __init__(self, dim: int, keepdim: bool=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return torch.min(m, dim=self.dim, keepdim=self.keepdim)


class UnionOn(nn.Module):

    def __init__(self, dim: int, keepdim: bool=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return torch.max(m, dim=self.dim, keepdim=self.keepdim)


class MaxMinAgg(nn.Module):

    def __init__(self, in_variables: int, in_features: int, out_features: int, agg_features: int):
        super().__init__()
        self._max_min = MaxMin(in_features, out_features * agg_features, in_variables)
        self._agg_features = agg_features
    
    def forward(self, m: torch.Tensor):
        data = self._max_min.forward(m)
        return data.view(*data.shape[:-1], -1, self._agg_features).max(dim=-1)[0]


class FuzzyElse(nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, m: torch.Tensor):

        return torch.clamp(1 - m.sum(self.dim, keepdim=True), 0, 1)


class WithFuzzyElse(nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        self.else_ = FuzzyElse(dim)
    
    @property
    def dim(self) -> int:
        return self.else_.dim

    @dim.setter
    def dim(self, dim: int) -> None:
        self.else_.dim = dim

    def forward(self, m: torch.Tensor):

        else_ = self.else_.forward(m)
        return torch.cat([m, else_], dim=self.else_.dim)


class IntersectOnLoss(MistifyLoss):

    def __init__(self, intersect: IntersectOn, reduction: str='mean', not_chosen_weight: float=1.0):
        super().__init__(intersect, reduction)
        self.intersect = intersect
        self.not_chosen_weight = not_chosen_weight
        self._mse = nn.MSELoss(reduction='none')

    def calc_loss(self, x: torch.Tensor, t: torch.Tensor, mask: torch.BoolTensor=None, weight: torch.Tensor=None):
        
        result = self._mse.forward(x, t)
        if mask is not None:
            result = result * mask.float()
        if weight is not None:
            result = result * weight
        return 0.5 * self.reduce(result)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.intersect.keepdim:
            y = y.unsqueeze(self.intersect.dim)
            t = t.unsqueeze(self.intersect.dim)

        with torch.no_grad():
            x_less_than = (x < t)
            x_t = x - torch.relu(y - t)
            chosen = x == y

        return (
            self.calc_loss(x, t.detach(), x_less_than)
            + self.calc_loss(x, x_t.detach(), chosen)
            + self.calc_loss(x, x_t.detach(), ~chosen, self.not_chosen_weight)
        )


class UnionOnLoss(MistifyLoss):

    def __init__(self, union: UnionOn, reduction: str='mean', not_chosen_weight: float=1.0):
        super().__init__(union, reduction)
        self.union = union
        self.not_chosen_weight = not_chosen_weight
        self._mse = nn.MSELoss(reduction='none')

    def calc_loss(self, x: torch.Tensor, t: torch.Tensor, mask: torch.BoolTensor=None, weight: torch.Tensor=None):
        
        result = self._mse.forward(x, t)
        if mask is not None:
            result = result * mask.float()
        if weight is not None:
            result = result * weight
        return 0.5 * self.reduce(result)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.intersect.keepdim:
            y = y.unsqueeze(self.intersect.dim)
            t = t.unsqueeze(self.intersect.dim)

        with torch.no_grad():
            x_greater_than = (x > t)
            x_t = x + torch.relu(t - y)
            chosen = x == y

        return (
            self.calc_loss(x, t.detach(), x_greater_than)
            + self.calc_loss(x, x_t.detach(), chosen)
            + self.calc_loss(x, x_t.detach(), ~chosen, self.not_chosen_weight)
        )


class MaxMinLoss(MistifyLoss):

    def __init__(
        self, maxmin: MaxMin, reduction='batchmean', not_chosen_x_weight: float=1.0, not_chosen_theta_weight: float=1.0, 
        default_optim: ToOptim=ToOptim.BOTH
    ):
        super().__init__(maxmin, reduction)
        self._maxmin = maxmin
        self._default_optim = default_optim
        self._mse = nn.MSELoss(reduction='none')
        self.not_chosen_theta_weight = not_chosen_theta_weight
        self.not_chosen_x_weight = not_chosen_x_weight

    def calc_loss(self, x: torch.Tensor, t: torch.Tensor, mask: torch.BoolTensor=None, weight: torch.Tensor=None):
        result = 0.5 * self._mse.forward(x, t)
        if mask is not None:
            result = result * mask.float()
        if weight is not None:
            result = result * weight
        return self.reduce(result)
    
    def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
        return torch.min(x, w)

    def set_chosen(self, inner_values: torch.Tensor):

        val, idx = torch.max(inner_values, dim=-2, keepdim=True)
        return inner_values == val
        # chosen = torch.zeros(inner_values.size(), dtype=torch.bool, device=inner_values.device)
        # chosen.scatter_(1, idx,  1.0)
        # return chosen
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        x = x.unsqueeze(x.dim())
        y = y[:,None]
        t = t[:,None]
        w = self._maxmin.weight[None]
        inner_values = self.calc_inner_values(x, w)
        chosen = self.set_chosen(inner_values)
        with torch.no_grad():
            dy = nn_func.relu(t - y)
            d_inner = nn_func.relu(inner_values - t)

        loss = None
        if self._default_optim.theta():
            with torch.no_grad():

                # value will not exceed the x
                w_target = torch.max(torch.min(w + dy, x), w)
                w_target_2 = w - d_inner

            loss = (
                self.calc_loss(w, w_target_2.detach()) +
                # self.calc_loss(w, t.detach(), inner_values > t) +
                self.calc_loss(w, w_target.detach(), chosen) +
                self.calc_loss(w, w_target.detach(), ~chosen, self.not_chosen_theta_weight)
            )
        if self._default_optim.x():
            with torch.no_grad():
                # x_target = torch.max(torch.min(x + dy, w), x)
                # this is wrong.. y can end up targetting a value greater than
                # one because of this...
                # x=0.95, w=0.1 -> y=0.75 t=0.8 ... x will end up targetting 1.0
                # this is also a conundrum because i may want to reduce the value of
                # x.. But the value is so high it does not get reduced

                # value will not exceed the target if smaller than target
                # if larger than target will not change
                x_target = torch.max(x, torch.min(x + dy, t))
                w_target_2 = x - d_inner

            cur_loss = (
                self.calc_loss(x, w_target_2.detach()) +
                self.calc_loss(x, x_target.detach(), chosen) +
                self.calc_loss(x, x_target.detach(), ~chosen, self.not_chosen_x_weight) 
            )
            loss = cur_loss if loss is None else loss + cur_loss

        return loss


class MinMaxLoss(MistifyLoss):

    def __init__(self, minmax: MinMax, reduction='batchmean', not_chosen_weight: float=None, default_optim: ToOptim=ToOptim.BOTH):
        super().__init__(minmax, reduction)
        self._minmax = minmax
        self._default_optim = default_optim
        self._mse = nn.MSELoss(reduction='none')
        self.not_chosen_weight = not_chosen_weight or 1.0

    def calc_loss(self, x: torch.Tensor, t: torch.Tensor, mask: torch.BoolTensor=None, weight: torch.Tensor=None):
        result = 0.5 * self._mse.forward(x, t)
        if mask is not None:
            result = result * mask.float()
        if weight is not None:
            result = result * weight
        return self.reduce(result)
    
    def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
        return torch.min(x, w)

    def set_chosen(self, inner_values: torch.Tensor):

        val, idx = torch.min(inner_values, dim=-2, keepdim=True)
        return inner_values == val
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        # TODO: Update so that the number of dimensions is more flexible
        x = x.unsqueeze(x.dim())
        y = y[:,None]
        t = t[:,None]
        w = self._minmax.weight[None]
        inner_values = self.calc_inner_values(x, w)
        chosen = self.set_chosen(inner_values)
        with torch.no_grad():
            dy = nn_func.relu(y - t)

        if self._default_optim.theta():
            with torch.no_grad():
                w_target = torch.min(torch.max(w - dy, x), w)

            loss = (
                self.calc_loss(w, t.detach(), inner_values < t) +
                self.calc_loss(w, w_target.detach(), chosen) +
                self.calc_loss(w, w_target.detach(), ~chosen, self.not_chosen_weight)
            )
        if self._default_optim.x():
            with torch.no_grad():
                x_target = torch.min(torch.max(x - dy, w), x)

            cur_loss = (
                self.calc_loss(x, t.detach(), inner_values < t) +
                self.calc_loss(x, x_target.detach(), chosen) +
                self.calc_loss(x, x_target.detach(), ~chosen, self.not_chosen_weight) 
            )
            loss = cur_loss if loss is None else loss + cur_loss

        return loss


class MaxProdLoss(MistifyLoss):

    def __init__(self, maxprod: MaxProd, reduction='batchmean', not_chosen_weight: float=None, default_optim: ToOptim=ToOptim.BOTH):
        super().__init__(maxprod, reduction)
        self._maxprod = maxprod
        self.reduction = reduction
        self._default_optim = default_optim
        self._mse = nn.MSELoss(reduction='none')
        self.not_chosen_weight = not_chosen_weight or 1.0

    def calc_loss(self, x: torch.Tensor, t: torch.Tensor, mask: torch.BoolTensor=None, weight: torch.Tensor=None):
        
        result = self._mse.forward(x, t)
        if mask is not None:
            result = result * mask.float()
        if weight is not None:
            result = result * weight
        return 0.5 * self.reduce(result)
    
    def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
        return x * w

    def set_chosen(self, inner_values: torch.Tensor):

        val, _ = torch.max(inner_values, dim=-2, keepdim=True)
        return inner_values == val

        # _, idx = torch.max(inner_values, dim=-2, keepdim=True)
        # chosen = torch.zeros(inner_values.size(), dtype=torch.bool)
        # chosen.scatter_(1, idx,  1.0)
        # return chosen
    
    def clamp(self):

        self._maxprod.weight.data = torch.clamp(self._maxprod.weight.data, 0, 1).detach()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        x = x.unsqueeze(x.dim())
        y = y[:,None]
        t = t[:,None]
        w = self._maxprod.weight[None]

        if not self._default_optim.theta():
            w = w.detach()
        if not self._default_optim.x():
            x = x.detach()

        inner_values = self.calc_inner_values(x, w)
        chosen = self.set_chosen(inner_values)
        with torch.no_grad():
            dy = nn_func.relu(t - y)
            inner_target = torch.min(inner_values + dy, torch.tensor(1.0))
    
        if self._default_optim.theta():
            loss = (
                self.calc_loss(inner_values, t.detach(), inner_values > t) +
                self.calc_loss(inner_values, inner_target.detach(), chosen)  +
                self.calc_loss(inner_values, inner_target.detach(), ~chosen, self.not_chosen_weight) 
            )

        return loss




# class FuzzyCalcApprox(object):

#     def intersect(self, x: FuzzySet, y: FuzzySet):
#         pass

#     def union(self, x: FuzzySet, y: FuzzySet):
#         pass



# class FuzzySetParam(SetParam):

#     def __init__(self, set_: torch.Tensor, requires_grad: bool=True):
 
#         # if isinstance(set_, torch.Tensor):
#         #     set_ = FuzzySet(set_)
#         super().__init__(set_, requires_grad=requires_grad)



# class FuzzySet(Set):

#     def transpose(self, first_dim, second_dim) -> 'FuzzySet':
#         assert self._value_size is not None
#         return FuzzySet(self._data.transpose(first_dim, second_dim), self._is_batch)

#     def intersect_on(self, dim: int=-1):
#         return FuzzySet(torch.min(self.data, dim=dim)[0], self.is_batch)

#     def unify_on(self, dim: int=-1):
#         return FuzzySet(torch.max(self.data, dim=dim)[0], self.is_batch)

    
#     def inclusion(self, other: 'FuzzySet') -> 'FuzzySet':
#         return FuzzySet(
#             (1 - other.data) + torch.min(self.data, other.data), self._is_batch
#         )

#     def exclusion(self, other: 'FuzzySet') -> 'FuzzySet':
#         return FuzzySet(
#             (1 - self.data) + torch.min(self.data, other.data), self._is_batch
#         )
    
#     def differ(self, other: 'FuzzySet'):
#         return FuzzySet(torch.clamp(self.data - other.data, 0, 1), self.is_batch)
#     def unify(self, other: 'FuzzySet'):
#         return FuzzySet(torch.max(self.data, other.data), self.is_batch)

#     def intersect(self, other: 'FuzzySet'):
#         return FuzzySet(torch.min(self.data, other.data), self._is_batch)

#     def __sub__(self, other: 'FuzzySet'):
#         return self.differ(other)

#     def __mul__(self, other: 'FuzzySet'):
#         return intersect(self, other)

#     def __add__(self, other: 'FuzzySet'):
#         return self.unify(other)
    
#     def convert_variables(self, *size_after: int):
        
#         if self._is_batch:
#             return FuzzySet(
#                 self._data.view(self._data.size(0), *size_after, -1), True
#             )
#         return self.__class__(
#             self._data.view(*size_after, -1), False
#         )

#     def reshape(self, *size: int):
#         return FuzzySet(
#             self.data.reshape(*size), self.is_batch
#         )

#     @classmethod
#     def negatives(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

#         return FuzzySet(
#             torch.zeros(*size, dtype=dtype, device=device), is_batch
#         )
    
#     @classmethod
#     def positives(cls, *size: int, dtype=torch.float32, is_batch: bool=None, device='cpu'):

#         return FuzzySet(
#             torch.ones(*size, dtype=dtype, device=device), 
#             is_batch
#         )

#     @classmethod
#     def rand(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

#         return FuzzySet(
#             (torch.rand(*size, device=device)).type(dtype), 
#             is_batch
#         )

#     def transpose(self, first_dim, second_dim) -> 'FuzzySet':
#         assert self._value_size is not None
#         return FuzzySet(self._data.transpose(first_dim, second_dim), self._is_batch)

#     @property
#     def shape(self) -> torch.Size:
#         return self._data.shape

#     def __getitem__(self, key):
#         return FuzzySet(self.data[key])