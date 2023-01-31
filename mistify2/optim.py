# 1st Party
from abc import abstractmethod
from enum import Enum

# 3rd Party
import torch
import torch.nn as nn


# Local
from .utils import reduce
from . import conversion
from . import fuzzy, crisp

# The performance of these will be bad so if 



class ToOptim(Enum):

    X = 'x'
    THETA = 'theta'
    BOTH = 'both'

    def x(self) -> bool:
        return self in (ToOptim.X, ToOptim.BOTH)

    def theta(self) -> bool:
        return self in (ToOptim.THETA, ToOptim.BOTH)
        

class BinaryWeightLoss(nn.Module):

    def __init__(self, to_binary: conversion.StepCrispConverter):
        """initialzier

        Args:
            linear (nn.Linear): Linear layer to optimize
            act_inverse (Reversible): The invertable activation of the layer
        """
        self._to_binary = to_binary

    def step(self, x: crisp.BinarySet, y: crisp.BinarySet, t: crisp.BinarySet):

        x, y, t = x.data, y.data, t.data
        # assessment, y, result = get_y_and_assessment(objective, x, t, result)
        # y = to_binary.forward(x)
        change = (y != t).type_as(y)
        if self._to_binary.same:
            loss = (self._to_binary.weight[None,None,:] * change) ** 2
        else:
            loss = (self._to_binary.weight[None,:,:] * change) ** 2

        # TODO: Reduce the loss
        return loss


class BinaryXLoss(nn.Module):

    def __init__(self, to_binary: conversion.StepCrispConverter):
        """initialzier

        Args:
            linear (nn.Linear): Linear layer to optimize
            act_inverse (Reversible): The invertable activation of the layer
        """
        self._to_binary = to_binary

    def step(self, x: crisp.BinarySet, y: crisp.BinarySet, t: crisp.BinarySet):
        x, y, t = x.data, y.data, t.data

        # assessment, y, result = get_y_and_assessment(objective, x, t, result)
        # y = to_binary.forward(x)
        change = (y != t).type_as(y)
        loss = (x[:,:,None] * change) ** 2

        # TODO: Reduce the loss
        return loss


class FuzzyCompLoss(nn.Module):

    def __init__(self, relation_lr: float=1.0, reduction='mean'):
        super().__init__()
        
        # TODO: Fix this error here
        self._fuzzy_comp = FuzzyLoss(reduction=reduction)
        self._fuzzy_comp_to_all = FuzzyCompToAllLoss(reduction=reduction)
        self._relation = None
        self._reduction = reduction
        self.relation_lr = relation_lr

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
        return self.outer(self.inner(
            x.unsqueeze(-1), w[None]
        ), dim=-1, keepdim=True)[1]

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def inner(self, x: torch.Tensor, w: torch.Tensor):
        pass

    @abstractmethod
    def outer(self, x: torch.Tensor):
        pass


class MaxMinLoss(FuzzyCompLoss):

    def __init__(self, maxmin: fuzzy.MaxMin, relation_lr: float = 1, reduction='mean', default_optim: ToOptim=ToOptim.BOTH):
        super().__init__(relation_lr, reduction)
        self.maxmin = maxmin
        self.default_optim = default_optim

    def forward(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        losses = []
        if self.default_optim.x():
            losses.append(self.forward_x(x, y, t))
        if self.default_optim.theta():
            losses.append(self.forward_theta(x, y, t))
        return sum(losses)
    
    def forward_x(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        x, y, t = x.data, y.data, t.data
        if self.maxmin.to_complement:
            x = torch.cat([x, 1 - x], dim=1)

        w = self.maxmin.weight

        rel_idx = self.calc_idx(self.calc_relation(w[None], t[:,None], agg_dim=2), w)
        chosen = self.set_chosen(x, w, rel_idx)

        y = y[:,None]
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

    def forward_theta(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:

        x, y, t = x.data, y.data, t.data
        if self.maxmin.to_complement:
            x = torch.cat([x, 1 - x], dim=1)

        w = self.maxmin.weight

        rel_idx = self.calc_idx(x, self.calc_relation(x[:,:,None], t[:,None], agg_dim=0))
        chosen = self.set_chosen(x, w, rel_idx)

        y = y[:,None]
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

    def inner(self, x: torch.Tensor, w: torch.Tensor):
        return torch.min(x, w)

    def outer(self, x: torch.Tensor):
        return torch.max(x)
        

class MinMaxLoss(FuzzyCompLoss):

    def __init__(self, minmax: fuzzy.MinMax, relation_lr: float = 1, reduction='mean', default_optim: ToOptim=ToOptim.BOTH):
        super().__init__(relation_lr, reduction)
        self.minmax = minmax
        self.default_optim = default_optim

    def forward(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        losses = []
        if self.default_optim.x():
            losses.append(self.forward_x(x, y, t))
        if self.default_optim.theta():
            losses.append(self.forward_theta(x, y, t))
        return sum(losses)

    def forward_theta(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        
        x, y, t = x.data, y.data, t.data
        if self.minmax.to_complement:
            x = torch.cat([x, 1 - x], dim=1)

        w = self.minmax.weight

        rel_idx = self.calc_idx(x, self.calc_relation(x[:,:,None], t[:,None], agg_dim=0))
        chosen = self.set_chosen(x, w, rel_idx)

        y = y[:,None]
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

    def forward_x(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        
        x, y, t = x.data, y.data, t.data
        if self.minmax.to_complement:
            x = torch.cat([x, 1 - x], dim=1)
    
        w = self.minmax.weight

        rel_idx = self.calc_idx(self.calc_relation(w[None], t[:,None], agg_dim=2), w)
        chosen = self.set_chosen(x, w, rel_idx)

        y = y[:,None]
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

    def inner(self, x: torch.Tensor, w: torch.Tensor):
        return torch.max(x, w)

    def outer(self, x: torch.Tensor):
        return torch.min(x)
        


class MaxProdThetaLoss(FuzzyCompLoss):
    """TODO: Implement
    """

    def __init__(self, maxprod: fuzzy.MaxProd, relation_lr: float = 1, reduction='mean'):
        super().__init__(relation_lr, reduction)
        self.maxprod = maxprod

    def forward_x(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        
        x, y, t = x.data, y.data, t.data
        if self.maxprod.to_complement:
            x = torch.cat([x, 1 - x], dim=1)
        w = self.maxprod.weight

        rel_idx = self.calc_idx(self.calc_relation(w[None], t[:,None], agg_dim=2), w)
        chosen = self.set_chosen(x, w, rel_idx)

        y = y[:,None]
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

    def forward_theta(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        
        x, y, t = x.data, y.data, t.data
        if self.maxprod.to_complement:
            x = torch.cat([x, 1 - x], dim=1)
        w = self.minmax.weight

        rel_idx = self.calc_idx(x, self.calc_relation(x[:,:,None], t[:,None], agg_dim=0))
        chosen = self.set_chosen(x, w, rel_idx)

        y = y[:,None]
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
    
    def inner(self, x: torch.Tensor, w: torch.Tensor):
        return x * w

    def outer(self, x: torch.Tensor):
        return torch.max(x)
        

class FuzzyCompToAllLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, t: torch.Tensor, chosen: torch.Tensor):

        chosen_clone = chosen.clone()
        chosen_clone[chosen == 1] = -torch.inf
        chosen_clone[chosen == 0] = 1.0

        # input_difference[chosen.long()] = -torch.inf
        # TODO: Figure out how to handle this
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


class IntersectOnLoss(nn.Module):

    def __init__(self, intersect: fuzzy.IntersectOn, reduction: str='mean'):
        super().__init__()
        self.intersect = intersect
        self.reduction = reduction

    def forward(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        x, y, t = x.data, y.data, t.data
        loss = 0.5 * (t.unsqueeze(self.intersect.dim) - x) ** 2
        y_greater_than = (y > t).float()  
        x_greater_than = (x > t[:,None]).float()

        return (
            (x_greater_than * loss * y_greater_than).float().mean() + (loss * ~y_greater_than).float().mean()
        )


class UnionOnLoss(nn.Module):

    def __init__(self, union: fuzzy.UnionOn, reduction: str='mean'):
        super().__init__()
        self.union = union
        self.reduction = reduction

    def forward(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
    
        x, y, t = x.data, y.data, t.data
        loss = 0.5 * (t.unsqueeze(self.intersect.dim) - x) ** 2
        y_greater_than = (y > t)
        x_greater_than = (x > t[:,None])

        return (
            (~x_greater_than * loss * ~y_greater_than).float().mean() 
            + (loss * y_greater_than).float().mean()
        )
