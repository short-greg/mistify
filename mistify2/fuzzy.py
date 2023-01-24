import typing
import torch
import torch.nn as nn
from .utils import reduce, get_comp_weight_size,  maxmin, minmax, maxprod
from abc import abstractmethod
from .base import Set, SetParam, CompositionBase


class FuzzySet(Set):

    def transpose(self, first_dim, second_dim) -> 'FuzzySet':
        assert self._value_size is not None
        return FuzzySet(self._data.transpose(first_dim, second_dim), self._is_batch)

    def intersect_on(self, dim: int=-1):
        return FuzzySet(torch.min(self.data, dim=dim)[0], self.is_batch)

    def unify_on(self, dim: int=-1):
        return FuzzySet(torch.max(self.data, dim=dim)[0], self.is_batch)

    def differ(self, other: 'FuzzySet'):
        return FuzzySet(torch.clamp(self.data - other.data, 0, 1), self.is_batch)
    
    def unify(self, other: 'FuzzySet'):
        return FuzzySet(torch.max(self.data, other.data), self.is_batch)

    def intersect(self, other: 'FuzzySet'):
        return FuzzySet(torch.min(self.data, other.data), self._is_batch)

    def inclusion(self, other: 'FuzzySet') -> 'FuzzySet':
        return FuzzySet(
            (1 - other.data) + torch.min(self.data, other.data), self._is_batch
        )

    def exclusion(self, other: 'FuzzySet') -> 'FuzzySet':
        return FuzzySet(
            (1 - self.data) + torch.min(self.data, other.data), self._is_batch
        )
    
    def __sub__(self, other: 'FuzzySet'):
        return self.differ(other)

    def __mul__(self, other: 'FuzzySet'):
        return intersect(self, other)

    def __add__(self, other: 'FuzzySet'):
        return self.unify(other)
    
    def convert_variables(self, *size_after: int):
        
        if self._is_batch:
            return FuzzySet(
                self._data.view(self._data.size(0), *size_after, -1), True
            )
        return self.__class__(
            self._data.view(*size_after, -1), False
        )

    @classmethod
    def negatives(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

        return FuzzySet(
            torch.zeros(*size, dtype=dtype, device=device), is_batch
        )
    
    @classmethod
    def positives(cls, *size: int, dtype=torch.float32, is_batch: bool=None, device='cpu'):

        return FuzzySet(
            torch.ones(*size, dtype=dtype, device=device), 
            is_batch
        )

    @classmethod
    def rand(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

        return FuzzySet(
            (torch.rand(*size, device=device)).type(dtype), 
            is_batch
        )

    def transpose(self, first_dim, second_dim) -> 'Set':
        assert self._value_size is not None
        return FuzzySet(self._data.transpose(first_dim, second_dim), self._is_batch)

    @property
    def shape(self) -> torch.Size:
        return self._data.shape

    def __getitem__(self, key):
        return FuzzySet(self.data[key])


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


class FuzzySetParam(SetParam):

    def __init__(self, set_: typing.Union[FuzzySet, torch.Tensor], requires_grad: bool=True):
 
        if isinstance(set_, torch.Tensor):
            set_ = FuzzySet(set_)
        super().__init__(set_, requires_grad=requires_grad)


class MaxMin(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> SetParam:
        return FuzzySetParam(
            FuzzySet.positives(get_comp_weight_size(in_features, out_features, in_variables))
        )

    def forward(self, m: FuzzySet):
        # assume inputs are binary
        # binarize the weights
        return FuzzySet(
            maxmin(self.prepare_inputs(m), self.weight.data
        ), m.is_batch)


class MaxProd(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> SetParam:
        return FuzzySetParam(
            FuzzySet.positives(get_comp_weight_size(in_features, out_features, in_variables))
        )

    def forward(self, m: FuzzySet):
        # assume inputs are binary
        # binarize the weights
        return FuzzySet(
            maxprod(self.prepare_inputs(m), self.weight.data), m.is_batch
        )


class MinMax(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> SetParam:
        return FuzzySetParam(
            FuzzySet.negatives(get_comp_weight_size(in_features, out_features, in_variables))
        )
    
    def forward(self, m: FuzzySet):
        # assume inputs are binary
        # binarize the weights
        return FuzzySet(
            minmax(self.prepare_inputs(m), self.weight.data), m.is_batch
        )


class FuzzyRelation(CompositionBase):

    def __init__(
        self, in_features: int, out_features: int, 
        in_variables: int=None, to_complement: bool=True, 
        inner=None, outer=None
    ):
        super().__init__()
        self.inner = inner or torch.min
        self.outer = outer or (lambda x: torch.max(x, dim=-2)[0])

        self.weight = FuzzySetParam(
            FuzzySet.positives(get_comp_weight_size(in_features, out_features, in_variables))
        )

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> SetParam:
        return FuzzySetParam(
            FuzzySet.positives(get_comp_weight_size(in_features, out_features, in_variables))
        )

    def forward(self, m: FuzzySet):

        return FuzzySet(
            self.outer(
                self.inner(self.prepare_inputs(m), self.weight.data[None])
            ), m.is_batch
        )


class MaxMinAgg(nn.Module):

    def __init__(self, in_variables: int, in_features: int, out_features: int, agg_features: int, complement_inputs: bool=False):

        super().__init__()
        self._max_min = MaxMin(in_features, out_features * agg_features, complement_inputs, in_variables)
        self._agg_features = agg_features
    
    @property
    def to_complement(self) -> bool:
        return self._max_min._complement_inputs
    
    def forward(self, m: FuzzySet):
        data = self._max_min.forward(m).data
        data = data.view(*data.shape[:-1], -1, self._agg_features).max(dim=-1)[0]
        return FuzzySet(data, m.is_batch)


class FuzzyCompLoss(nn.Module):

    def __init__(self, to_complement: bool=False, relation_lr: float=1.0, reduction='mean', inner=torch.min, outer=torch.max):
        super().__init__()
        self._to_complement = to_complement
        
        # TODO: Fix this error here
        self._fuzzy_comp = FuzzyLoss(reduction=reduction)
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

        output_less_than = y < t

        return (
            self._fuzzy_comp.forward(
                w, t, mask=~output_less_than
            )
            + self._fuzzy_comp.forward(
                w, x, mask=chosen & (x > t) & output_less_than & (x < w)
            )
            + self._fuzzy_comp.forward(
                w, t, mask=chosen & (x < t) & output_less_than
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
        output_less_than = ~output_greater_than

        # TODO: Need to increase if chosen is greater than but output is less than
        return (
            self._fuzzy_comp.forward(
                x, t, mask=~output_greater_than
            )
            + self._fuzzy_comp.forward(
                x, w, mask=chosen & (w > t) & output_less_than & (w < x)
            )
            + self._fuzzy_comp.forward(
                x, t, mask=chosen & (w < t) & output_less_than
            )
            + self._fuzzy_comp_to_all.forward(
                x, w, chosen
            )
        )

# TODO: MaxProd loss..


class MaxProdThetaLoss(FuzzyCompLoss):
    """TODO: Implement
    """

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

        output_less_than = y < t

        return (
            self._fuzzy_comp.forward(
                w, t, mask=~output_less_than
            )
            + self._fuzzy_comp.forward(
                w, x, mask=chosen & (x > t) & output_less_than & (x < w)
            )
            + self._fuzzy_comp.forward(
                w, t, mask=chosen & (x < t) & output_less_than
            )
            + self._fuzzy_comp_to_all.forward(
                w, x, chosen
            )
        )


class MaxProdXLoss(FuzzyCompLoss):
    """TODO: Implement
    """

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


class FuzzyLoss(nn.Module):
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.BoolTensor):

        return reduce(((x - t) * mask.float()), self.reduction)


class FuzzyElse(nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, m: FuzzySet):

        return FuzzySet(
            torch.clamp(1 - m.data.sum(self.dim, keepdim=True), 0, 1),
            is_batch=m.is_batch
        )


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

    def forward(self, m: FuzzySet):

        else_ = self.else_.forward(m)
        return FuzzySet(
            torch.cat([m.data, else_.data], dim=self.else_.dim),
            is_batch=m.is_batch
        )
