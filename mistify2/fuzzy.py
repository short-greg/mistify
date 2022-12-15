import typing
import torch
import torch.nn as nn
from utils import reduce, get_comp_weight_size
from abc import abstractmethod
from .base import ISet


class FuzzySet(ISet):

    def __init__(self, data: torch.Tensor, is_batch: bool=False, multiple_variables: bool=False):

        self._data = data
        self._n_samples = None
        self._n_variables = None
        self._multiple_variables = multiple_variables
        self._is_batch = is_batch
        if is_batch and multiple_variables:
            assert data.dim() == 3
            self._n_variables = data.size(1)
            self._batch_size = data.size(0)
        elif multiple_variables:
            assert data.dim() == 2
            self._n_samples = data.size(0)
        elif is_batch:
            assert data.dim() == 2
            self._batch_size = data.size(0)
        else:
            assert data.dim() == 1
    
    @property
    def is_batch(self) -> bool:
        return self._is_batch

    @property
    def multiple_variables(self) -> bool:
        return self._multiple_variables
    
    @property
    def data(self) -> torch.Tensor:
        return self._data

    def swap_variables(self) -> 'FuzzySet':
        assert self._multiple_variables
        if self._is_batch:
            dim1, dim2 = 1, 2
        else:
            dim1, dim2 = 0, 1
        return FuzzySet(self._data.transpose(dim2, dim1), self._is_batch, True)

    def convert_variables(self, n_after: int) -> 'FuzzySet':
        
        if self._is_batch:
            return FuzzySet(
                self._data.view(self._data.size(0), n_after, -1), True, True
            )
        return FuzzySet(
            self._data.view(n_after, -1), False, True
        )

    def intersect_on(self, dim: int=-1):
        return FuzzySet(torch.min(self.data, dim=dim)[0], self.is_batch, self.multiple_variables)

    def unify_on(self, dim: int=-1):
        return FuzzySet(torch.max(self.data, dim=dim)[0], self.is_batch, self.multiple_variables)

    def differ(self, other: 'FuzzySet'):
        return FuzzySet(torch.clamp(self.data - other.data, 0, 1), self.is_batch, self.multiple_variables)
    
    def unify(self, other: 'FuzzySet'):
        return FuzzySet(torch.max(self.data, other.data), self.is_batch, self.multiple_variables)

    def intersect(self, other: 'FuzzySet'):
        return FuzzySet(torch.min(self.data, other.data), self._is_batch, self._multiple_variables)

    def inclusion(self, other: 'FuzzySet') -> 'FuzzySet':
        return FuzzySet(
            (1 - self.data) + torch.min(self.data, other.data), self._is_batch, self._multiple_variables
        )

    def exclusion(self, other: 'FuzzySet') -> 'FuzzySet':
        return FuzzySet(
            (1 - other.data) + torch.min(self.data, other.data), self._is_batch, self._multiple_variables
        )

    def __sub__(self, other: 'FuzzySet'):
        return self.differ(other)

    def __mul__(self, other: 'FuzzySet'):
        return intersect(self, other)

    def __add__(self, other: 'FuzzySet'):
        return self.unify(other)

    @classmethod
    def get_size(cls, n_features: int, batch_size: int=None, n_variables: int=None):

        if batch_size is not None and n_variables is not None:
            return (batch_size, n_variables, n_features)
        elif batch_size is not None:
            return (batch_size, n_features)
        elif n_variables is not None:
            return (n_variables, n_features)
        return (n_features,)
    
    @classmethod
    def zeros(cls, n_features: int, batch_size: int=None, n_variables: int=None, dtype=torch.float32, device='cpu'):

        size = cls.get_size(n_features, batch_size, n_variables)
        return FuzzySet(
            torch.zeros(*size, dtype=dtype, device=device), 
            batch_size is not None, n_variables is not None
        )

    @classmethod
    def ones(cls, n_features: int, batch_size: int=None, n_variables: int=None, dtype=torch.float32, device='cpu'):

        size = cls.get_size(n_features, batch_size, n_variables)
        return FuzzySet(
            torch.ones(*size, dtype=dtype, device=device), 
            batch_size is not None, n_variables is not None
        )

    @classmethod
    def rand(cls, n_features: int, batch_size: int=None, n_variables: int=None, dtype=torch.float32, device='cpu'):

        size = cls.get_size(n_features, batch_size, n_variables)
        return FuzzySet(
            torch.rand(*size, dtype=dtype, device=device), 
            batch_size is not None, n_variables is not None
        )


class FuzzyCalcApprox(object):

    def intersect(self, x: FuzzySet, y: FuzzySet):
        pass

    def union(self, x: FuzzySet, y: FuzzySet):
        pass


def intersect(m: FuzzySet, m2: FuzzySet):
    return FuzzySet(torch.min(m.data, m2.data))


def unify(m: FuzzySet, m2: FuzzySet):
    return FuzzySet(torch.max(m.data, m2.data))


def differ(m: FuzzySet, m2: FuzzySet):
    return FuzzySet((m.data - m2._data).clamp(0.0, 1.0))


class FuzzySetParam(nn.parameter.Parameter):

    def __init__(self, fuzzy_set: FuzzySet, requires_grad: bool=True):

        super().__init__(fuzzy_set.data, requires_grad=True)
        self._fuzzy_set = fuzzy_set

    @property
    def data(self) -> torch.Tensor:
        return self._fuzzy_set.data

    @data.setter
    def data(self, data: torch.Tensor):
        assert data.size() == self._fuzzy_set.data.size()
        super().data = data
        self._fuzzy_set.data = data

    @property
    def fuzzy_set(self) -> FuzzySet:
        return self._fuzzy_set
    
    @fuzzy_set.setter
    def fuzzy_set(self, fuzzy_set: 'FuzzySet'):
        self._fuzzy_set = fuzzy_set
        self.data = fuzzy_set.data
    

class FuzzyCompositionBase(nn.Module):

    def __init__(
        self, in_features: int, out_features: int, 
        complement_inputs: bool=False, in_variables: int=None
    ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._in
        self._complement_inputs = complement_inputs
        if complement_inputs:
            in_features *= 2
        self._multiple_variables = in_variables is not None
        # store weights as values between 0 and 1
        self.weight = nn.parameter.Parameter(
            torch.ones(get_comp_weight_size(in_features, out_features, in_variables))
        )

    @property
    def to_complement(self) -> bool:
        return self._complement_inputs
    
    @abstractmethod
    def forward(self, m: FuzzySet):
        pass


class MaxMin(FuzzyCompositionBase):

    def forward(self, m: torch.Tensor):
        # assume inputs are binary
        # binarize the weights
        if self._complement_inputs:
            m = torch.cat([m, 1 - m], dim=1)
        return FuzzySet(torch.max(
            torch.min(m[:,:,None], self.weight[None]), dim=-2
        )[0], True, self._multiple_variables)


class MaxProd(FuzzyCompositionBase):

    def forward(self, m: torch.Tensor):
        # assume inputs are binary
        # binarize the weights
        if self._complement_inputs:
            m = torch.cat([m, 1 - m], dim=1)
        return FuzzySet(torch.max(
            m[:,:,None] * self.weight[None], dim=-2
        )[0], True, self._multiple_variables)


class MaxMinAgg(nn.Module):

    def __init__(self, in_variables: int, in_features: int, out_features: int, complement_inputs: bool=False):

        super().__init__()
        self._max_min = MaxMin(in_features, out_features, complement_inputs, in_variables)
    
    @property
    def to_complement(self) -> bool:
        return self._max_min._complement_inputs
    
    def forward(self, m: torch.Tensor):
        return FuzzySet(self._max_min.forward(m).data.max(dim=-1)[0], True, False)


class MinMax(FuzzyCompositionBase):

    @property
    def to_complement(self) -> bool:
        return self._complement_inputs
    
    def forward(self, m: torch.Tensor):
        # assume inputs are binary
        # binarize the weights
        if self._complement_inputs:
            m = torch.cat([m, 1 - m], dim=1)
        return FuzzySet(torch.min(
            torch.max(m[:,:,None], self.weight[None]), dim=-2
        )[0], True, self._multiple_variables)


class FuzzyCompLoss(nn.Module):

    def __init__(self, to_complement: bool=False, relation_lr: float=1.0, reduction='mean', inner=torch.min, outer=torch.max):
        super().__init__()
        self._to_complement = to_complement
        self._fuzzy_comp = FuzzyCompLoss(reduction=reduction)
        self._fuzzy_comp_to_all = FuzzyCompToAllLoss(reduction=reduction)
        self._relation = None
        self._reduction = reduction
        self.relation_lr = relation_lr
        self._inner = inner
        self._outer = outer
    
    def set_chosen(self, x: torch.Tensor, w: torch.Tensor, idx: torch.LongTensor):

        chosen = torch.zeros(x.size(0), x.size(1), w.size(1), dtype=torch.bool)
        chosen.scatter_(1, idx,  1.0)
        return chosen

    @property
    def reduction(self) -> str:
        return self._reduction

    @reduction.setter
    def reduction(self, reduction: str):
        self._reduction = reduction
        self._fuzzy_comp.reduction = reduction
        self._fuzzy_comp_to_all.reduction = reduction

    def calc_relation(self, values: torch.Tensor, t: torch.Tensor, agg_dim=0):
        values_min = torch.min(values, t)
        relation = values_min.sum(dim=agg_dim) / values.sum(dim=agg_dim)

        if self._relation is not None and self.relation_lr is not None:
            relation = relation * self.lr + self._relation * (1 - self.relation_lr)
        self._relation = relation
        return relation
    
    def reset(self):
        self._relation = None

    def calc_idx(self, x: torch.Tensor, w: torch.Tensor):
        return self._outer(self._inner(
            x[:,:,None], w[None]
        ), dim=1, keepdim=True)[1]

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        pass

    def calc_y(self, x: torch.Tensor, w: torch.Tensor):
        return self._outer(self._inner(x[:,:,None], w[None]), dim=1)[0]


class MaxMinThetaLoss(FuzzyCompLoss):

    def __init__(self, to_complement: bool = False, relation_lr: float = 1, reduction='mean'):
        super().__init__(to_complement, relation_lr, reduction, inner=torch.min, outer=torch.max)

    def forward(self, x: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self._to_complement:
            x = torch.cat([x, 1 - x], dim=1)

        rel_idx = self.calc_idx(x, self.calc_relation(x[:,:,None], t[:,None], agg_dim=0))
        chosen = self.set_chosen(x, w, rel_idx)

        y = self.calc_y(x, w)[:,None]
        w = w[None]
        x = x[:,:,None].detach()
        t = t[:,None].detach()

        output_less_than = y > t

        return (
            self._fuzzy_comp.forward(
                w, t, mask=~output_less_than
            )
            + self._fuzzy_comp.forward(
                w, x, mask=chosen & (x < t) & output_less_than & (x > w)
            )
            + self._fuzzy_comp.forward(
                w, t, mask=chosen & (x > t) & output_less_than
            )
            + self._fuzzy_comp_to_all.forward(
                w, x, chosen
            )
        )
    

class MaxMinXLoss(FuzzyCompLoss):

    def __init__(self, to_complement: bool = False, relation_lr: float = 1, reduction='mean'):
        super().__init__(to_complement, relation_lr, reduction, inner=torch.min, outer=torch.max)

    def forward(self, x: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        
        if self._to_complement:
            x = torch.cat([x, 1 - x], dim=1)

        rel_idx = self.calc_idx(self.calc_relation(w[None], t[:,None], agg_dim=2), w)
        chosen = self.set_chosen(x, w, rel_idx)

        y = self.calc_y(x, w)[:,None]
        w = w[None].detach()
        x = x[:,:,None]
        t = t[:,None].detach()

        output_less_than = y < t

        return (
            self._fuzzy_comp.forward(
                x, t, mask=~output_less_than
            )
            + self._fuzzy_comp.forward(
                x, w, mask=chosen & (w < t) & output_less_than & (w > x)
            )
            + self._fuzzy_comp.forward(
                x, t, mask=chosen & (w > t) & output_less_than
            )
            + self._fuzzy_comp_to_all.forward(
                x, w, chosen
            )
        )


class MinMaxThetaLoss(FuzzyCompLoss):

    def __init__(self, to_complement: bool = False, relation_lr: float = 1, reduction='mean'):
        super().__init__(to_complement, relation_lr, reduction, inner=torch.max, outer=torch.min)

    def forward(self, x: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        
        if self._to_complement:
            x = torch.cat([x, 1 - x], dim=1)

        rel_idx = self.calc_idx(x, self.calc_relation(x[:,:,None], t[:,None], agg_dim=0))
        chosen = self.set_chosen(x, w, rel_idx)

        y = self.calc_y(x, w)[:,None]
        w = w[None]
        x = x[:,:,None].detach()
        t = t[:,None].detach()

        output_greater_than = y > t

        return (
            self._fuzzy_comp.forward(
                w, t, mask=~output_greater_than
            )
            + self._fuzzy_comp.forward(
                w, x, mask=chosen & (x > t) & output_greater_than & (x < w)
            )
            + self._fuzzy_comp.forward(
                w, t, mask=chosen & (x < t) & output_greater_than
            )
            + self._fuzzy_comp_to_all.forward(
                w, x, chosen
            )
        )


class MinMaxXLoss(FuzzyCompLoss):

    def __init__(self, to_complement: bool = False, relation_lr: float = 1, reduction='mean'):
        super().__init__(to_complement, relation_lr, reduction, inner=torch.max, outer=torch.min)

    def forward(self, x: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        
        if self._to_complement:
            x = torch.cat([x, 1 - x], dim=1)

        rel_idx = self.calc_idx(self.calc_relation(w[None], t[:,None], agg_dim=2), w)
        chosen = self.set_chosen(x, w, rel_idx)

        y = self.calc_y(x, w)[:,None]
        w = w[None].detach()
        x = x[:,:,None]
        t = t[:,None].detach()

        output_greater_than = y > t

        return (
            self._fuzzy_comp.forward(
                x, t, mask=~output_greater_than
            )
            + self._fuzzy_comp.forward(
                x, w, mask=chosen & (w > t) & output_greater_than & (w < x)
            )
            + self._fuzzy_comp.forward(
                x, t, mask=chosen & (w < t) & output_greater_than
            )
            + self._fuzzy_comp_to_all.forward(
                x, w, chosen
            )
        )


# TODO: MaxProd loss..

class FuzzyCompToAllLoss(nn.Module):

    def __init__(self, reduction='mean', inner=torch.max, outer=torch.min):
        super().__init__()
        self.reduction = reduction
        self.inner = inner
        self.outer = outer

    def forward(self, x: torch.Tensor, t: torch.Tensor, chosen: torch.Tensor):

        chosen_clone = chosen.clone()
        chosen_clone[chosen == 1] = -torch.inf
        chosen_clone[chosen == 0] = 1.0

        # input_difference[chosen.long()] = -torch.inf
        result = self.outer(
            self.inner((t - x) * chosen_clone, dim=0)[0], torch.tensor(0.0)
        )
        result[result.isinf()] = 0.0
        return reduce(result.mean(), self.reduction)


class FuzzyCompLoss(nn.Module):
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.BoolTensor):

        return reduce(((x - t) * mask.float()), self.reduction)




# class MaxMinComp(nn.Module):

#     def __init__(self, in_features: int, out_features: int, n_variables: int=None):

#         super().__init__()
#         self._n_variables = n_variables
#         self._in_features = in_features
#         self._out_features = out_features
#         size = (in_features, out_features) if n_variables is None else (n_variables, in_features, out_features)
#         self._weight_param = FuzzySetParam(
#             FuzzySet.ones(
#                 *size
#             )
#         )
    
#     def forward(self, m: FuzzySet):

#         assert m.is_batch

#         return FuzzySet(
#             torch.max(torch.min(m.data, self._weight_param.data), dim=-2),
#             is_batch=m.is_batch, multiple_variables=m.multiple_variables
#         )


# class MinMaxComp(nn.Module):

#     def __init__(self, in_features: int, out_features: int, n_variables: int=None):

#         super().__init__()
#         self._n_variables = n_variables
#         self._in_features = in_features
#         self._out_features = out_features
#         size = (in_features, out_features) if n_variables is None else (n_variables, in_features, out_features)
#         self._weight_param = FuzzySetParam(
#             FuzzySet.zeros(
#                 *size
#             )
#         )
    
#     def forward(self, m: FuzzySet):

#         return FuzzySet(
#             torch.min(torch.max(m.data, self._weight_param.data), dim=-2),
#             is_batch=m.is_batch, multiple_variables=m.multiple_variables
#         )
