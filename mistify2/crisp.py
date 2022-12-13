import typing
import torch
import torch.nn as nn
import typing
from abc import abstractmethod


class CrispSet(object):

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
    def data(self) -> torch.Tensor:
        return self._data

    def swap_variables(self) -> 'CrispSet':
        assert self._multiple_variables
        if self._is_batch:
            dim1, dim2 = 1, 2
        else:
            dim1, dim2 = 0, 1
        return CrispSet(self._data.transpose(dim2, dim1), self._is_batch, True)

    def differ(self, other: 'CrispSet'):
        return differ(self, other)
    
    def unify(self, other: 'CrispSet'):
        return unify(self, other)

    def intersect(self, other: 'CrispSet'):
        return intersect(self, other)
    
    def __sub__(self, other: 'CrispSet'):
        return self.differ(other)

    def __mul__(self, other: 'CrispSet'):
        return intersect(self, other)

    def __add__(self, other: 'CrispSet'):
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
        return CrispSet(
            torch.zeros(*size, dtype=dtype, device=device), 
            batch_size is not None, n_variables is not None
        )

    @classmethod
    def ones(cls, n_features: int, batch_size: int=None, n_variables: int=None, dtype=torch.float32, device='cpu'):

        size = cls.get_size(n_features, batch_size, n_variables)
        return CrispSet(
            torch.ones(*size, dtype=dtype, device=device), 
            batch_size is not None, n_variables is not None
        )

    @classmethod
    def rand(cls, n_features: int, batch_size: int=None, n_variables: int=None, dtype=torch.float32, device='cpu'):

        size = cls.get_size(n_features, batch_size, n_variables)
        return CrispSet(
            (torch.rand(*size, device=device) > 0.5).type(dtype), 
            batch_size is not None, n_variables is not None
        )


def intersect(m: CrispSet, m2: CrispSet):
    return CrispSet(torch.min(m.data, m2.data))


def unify(m: CrispSet, m2: CrispSet):
    return CrispSet(torch.max(m.data, m2.data))


def differ(m: CrispSet, m2: CrispSet):
    return CrispSet((m.data - m2._data).clamp(0.0, 1.0))


class CrispSetParam(nn.parameter.Parameter):

    def __init__(self, crisp_set: CrispSet, requires_grad: bool=True):

        super().__init__(crisp_set.data, requires_grad=True)
        self._crisp_set = crisp_set

    @property
    def data(self) -> torch.Tensor:
        return self._crisp_set.data

    @data.setter
    def data(self, data: torch.Tensor):
        assert data.size() == self._crisp_set.data.size()
        super().data = data
        self._crisp_set.data = data

    @property
    def crisp_set(self) -> CrispSet:
        return self._crisp_set
    
    @crisp_set.setter
    def crisp_set(self, crisp_set: 'CrispSet'):
        self._crisp_set = crisp_set
        self.data = crisp_set.data
    


class MaxMin(nn.Module):

    def __init__(self, in_features: int, out_features: int, complement_inputs: bool=False):

        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._complement_inputs = complement_inputs
        if complement_inputs:
            in_features *= 2
        # store weights as values between 0 and 1
        self.weight = nn.parameter.Parameter(
            torch.ones(in_features, self._out_features)
        )
    
    @property
    def to_complement(self) -> bool:
        return self._complement_inputs
    
    def forward(self, m: torch.Tensor):

        # assume inputs are binary
        # binarize the weights
        if self._complement_inputs:
            m = torch.cat([m, 1 - m], dim=1)
        return torch.max(
            torch.min(m[:,:,None], self.weight[None]), dim=1
        )[0]


class MinMax(nn.Module):

    def __init__(self, in_features: int, out_features: int, complement_inputs: bool=False):

        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._complement_inputs = complement_inputs
        if complement_inputs:
            in_features = in_features * 2
        # store weights as values between 0 and 1
        self.weight = nn.parameter.Parameter(
            torch.zeros(in_features, self._out_features)
        )
    
    @property
    def to_complement(self) -> bool:
        return self._complement_inputs
    
    def forward(self, m: torch.Tensor):
        # assume inputs are binary
        # binarize the weights
        if self._complement_inputs:
            m = torch.cat([m, 1 - m], dim=1)
        return torch.min(
            torch.max(m[:,:,None], self.weight[None]), dim=1
        )[0]


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


def reduce(value: torch.Tensor, reduction: str):

    if reduction == 'mean':
        return value.mean()
    elif reduction == 'sum':
        return value.sum()
    return value


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


