# 1st Party
from abc import abstractmethod

# 3rd Party
import torch
import torch.nn as nn


# Local
from .utils import reduce
from . import conversion
from . import fuzzy

# The performance of these will be bad so if 


class BinaryWeightLoss(nn.Module):

    def __init__(self, to_binary: conversion.StepCrispConverter):
        """initialzier

        Args:
            linear (nn.Linear): Linear layer to optimize
            act_inverse (Reversible): The invertable activation of the layer
        """
        self._to_binary = to_binary

    def step(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):

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

    def step(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):

        # assessment, y, result = get_y_and_assessment(objective, x, t, result)
        # y = to_binary.forward(x)
        change = (y != t).type_as(y)
        loss = (x[:,:,None] * change) ** 2

        # TODO: Reduce the loss
        return loss

# TODO: Add fuzzy losses


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
