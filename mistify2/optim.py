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
import torch.nn.functional as nn_func

# The performance of these will be slow so I possibly
# need to split up the calculation


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


class FuzzyCompToAllLoss(nn.Module):

    def __init__(self, inner, outer, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.inner = inner
        self.outer = outer

    def forward(self, x: torch.Tensor, t: torch.Tensor, chosen: torch.Tensor):

        chosen_clone = chosen.clone()
        chosen_clone[chosen == 1] = -torch.inf
        chosen_clone[chosen == 0] = 1.0

        # input_difference[chosen.long()] = -torch.inf
        # TODO: Figure out how to handle this
        result = self.outer(
            (t - x) * chosen_clone, dim=0)[0]
        # , torch.tensor(0.0)
        # )
        result[result.isinf()] = 0.0
        return reduce(result.mean(), self.reduction)


class FuzzyLoss(nn.Module):
    
    def __init__(self, reduction='samplemean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.BoolTensor):
        return reduce(
            torch.pow(((x - t.detach()) * mask.float()), 2), 
            self.reduction
        )


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


class MaxMinLoss(nn.Module):

    def __init__(self, maxmin: fuzzy.MaxMin, reduction='batchmean', not_chosen_weight: float=None, default_optim: ToOptim=ToOptim.BOTH):
        super().__init__()
        self._maxmin = maxmin
        self.reduction = reduction
        self._default_optim = default_optim
        self._mse = nn.MSELoss(reduction='none')
        self.not_chosen_weight = not_chosen_weight or 1.0

    def calc_loss(self, x: torch.Tensor, t: torch.Tensor, mask: torch.BoolTensor=None, weight: torch.Tensor=None):
        result = 0.5 * self._mse.forward(x, t)
        if mask is not None:
            result = result * mask.float()
        if weight is not None:
            result = result * weight
        return reduce(result, self.reduction)
    
    def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
        return torch.min(x, w)

    def set_chosen(self, inner_values: torch.Tensor):

        _, idx = torch.max(inner_values, dim=1, keepdim=True)
        chosen = torch.zeros(inner_values.size(), dtype=torch.bool)
        chosen.scatter_(1, idx,  1.0)
        return chosen
    
    def forward(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        
        if self._maxmin.to_complement:
            x = torch.cat([x, 1 - x], dim=1)
        x = x.data[:,:,None]
        y = y.data[:,None]
        t = t.data[:,None]
        w = self._maxmin.weight.data[None]
        inner_values = self.calc_inner_values(x, w)
        chosen = self.set_chosen(inner_values)
        with torch.no_grad():
            dy = nn_func.relu(t - y)

        if self._default_optim.theta():
            with torch.no_grad():
                w_target = torch.max(torch.min(w + dy, x), w)

            loss = (
                self.calc_loss(w, t.detach(), inner_values > t) +
                self.calc_loss(w, w_target.detach(), chosen) +
                self.calc_loss(w, w_target.detach(), ~chosen, self.not_chosen_weight)
            )
        if self._default_optim.x():
            with torch.no_grad():
                x_target = torch.max(torch.min(x + dy, w), x)

            cur_loss = (
                self.calc_loss(x, t.detach(), inner_values > t) +
                self.calc_loss(x, x_target.detach(), chosen) +
                self.calc_loss(x, x_target.detach(), ~chosen, self.not_chosen_weight) 
            )
            loss = cur_loss if loss is None else loss + cur_loss

        return loss


class MaxProdLoss(nn.Module):

    def __init__(self, maxprod: fuzzy.MaxProd, reduction='batchmean', not_chosen_weight: float=None, default_optim: ToOptim=ToOptim.BOTH):
        super().__init__()
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
        return 0.5 * reduce(result, self.reduction)
    
    def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
        return x * w

    def set_chosen(self, inner_values: torch.Tensor):

        _, idx = torch.max(inner_values, dim=1, keepdim=True)
        chosen = torch.zeros(inner_values.size(), dtype=torch.bool)
        chosen.scatter_(1, idx,  1.0)
        return chosen
    
    def clamp(self):

        self._maxprod.weight.data.data = torch.clamp(self._maxprod.weight.data, 0, 1).detach()
    
    def forward(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        
        if self._maxprod.to_complement:
            x = torch.cat([x, 1 - x], dim=1)
        x = x.data[:,:,None]
        y = y.data[:,None]
        t = t.data[:,None]
        w = self._maxprod.weight.data[None]

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


# class FuzzyCompLoss(nn.Module):

#     def __init__(self, relation_lr: float=None, reduction='mean'):
#         super().__init__()
        
#         # TODO: Fix this error here
#         self._fuzzy_comp = FuzzyLoss(reduction=reduction)
#         self._fuzzy_comp_to_all = FuzzyCompToAllLoss(self.inner, self.outer, reduction=reduction)
#         self._x_relation = None
#         self._theta_relation = None
#         self._reduction = reduction
#         self.relation_lr = relation_lr

#     def set_chosen(self, x: torch.Tensor, w: fuzzy.FuzzySetParam, idx: torch.LongTensor):
#         chosen = torch.zeros(x.size(0), x.size(1), w.size(1), dtype=torch.bool)
#         chosen.scatter_(1, idx,  1.0)
#         return chosen

#     @property
#     def reduction(self) -> str:
#         return self._reduction

#     @reduction.setter
#     def reduction(self, reduction: str):
#         self._reduction = reduction
#         self._fuzzy_comp.reduction = reduction
#         self._fuzzy_comp_to_all.reduction = reduction

#     def calc_relation(self, values: torch.Tensor, t: torch.Tensor, agg_dim=0):

#         values_min = torch.min(values, t)
#         numerator = values_min.sum(dim=agg_dim)
#         denominator = values.sum(dim=agg_dim)
#         relation = numerator / denominator
#         # relation = torch.min(relation, values_min.max(dim=agg_dim)[0])
#         return relation, numerator

#     def update_relation(self, cur_value, stored_value):
#         if stored_value is not None and self.relation_lr is not None:
#             return cur_value * self.relation_lr + stored_value * (1 - self.relation_lr)
#         return cur_value

#     def reset(self):
#         self._relation = None

#     def calc_chosen(self, x: torch.Tensor, w: torch.Tensor):
#         inner_value = self.inner(
#             x.unsqueeze(-1), w[None]
#         )
#         values, idx = self.outer(inner_value, dim=1, keepdim=True)
#         chosen_value = values
#         # print('Chosen Size: ', chosen_value.size())
#         return (inner_value == chosen_value), idx
#         # then get the agg_index?

#     def calc_idx(self, x: torch.Tensor, w: torch.Tensor):
#         return self.outer(self.inner(
#             x.unsqueeze(-1), w[None]
#         ), keepdim=True)[1]
#         # then get the agg_index?

#     @abstractmethod
#     def forward(self, x: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
#         pass

#     @abstractmethod
#     def inner(self, x: torch.Tensor, w: torch.Tensor):
#         pass

#     @abstractmethod
#     def outer(self, x: torch.Tensor, dim: int=-2, keepdim: bool=False):
#         pass


# class MaxMinLoss(FuzzyCompLoss):

#     def __init__(
#         self, maxmin: fuzzy.MaxMin, relation_lr: float = 1, 
#         reduction='mean', default_optim: ToOptim=ToOptim.BOTH
#     ):
#         super().__init__(relation_lr, reduction)
#         self.maxmin = maxmin
#         self.default_optim = default_optim

#     def forward(
#         self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, 
#         t: fuzzy.FuzzySet
#     ) -> torch.Tensor:
#         losses = []
#         if self.default_optim.x():
#             losses.append(self.forward_x(x, y, t))
#         if self.default_optim.theta():
#             losses.append(self.forward_theta(x, y, t))
#         return sum(losses)
    
#     def forward_x(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
#         x, y, t = x.data, y.data, t.data
#         if self.maxmin.to_complement:
#             x = torch.cat([x, 1 - x], dim=1)

#         w = self.maxmin.weight
        
#         chosen = self.calc_chosen(self.update_x_relation(w[None], t[:,None], agg_dim=2), w)
#         # chosen = self.set_chosen(x, w, rel_val)

#         y = y[:,None]
#         w = w[None].detach()
#         x = x[:,:,None]
#         t = t[:,None].detach()

#         output_less_than = y < t

#         return (
#             self._fuzzy_comp.forward(
#                 x, t, mask=~output_less_than
#             )
#             + self._fuzzy_comp.forward(
#                 x, w, mask=(w < t) & output_less_than & (w > x) & chosen
#             )
#             + self._fuzzy_comp.forward(
#                 x, t, mask=(w > t) & output_less_than & chosen
#             )
#             # + self._fuzzy_comp_to_all.forward(
#             #     x, w, chosen
#             # )
#         )

#     def forward_theta(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:

#         x, y, t = x.data, y.data, t.data
#         if self.maxmin.to_complement:
#             x = torch.cat([x, 1 - x], dim=1)

#         w = self.maxmin.weight

#         # rel_idx = self.calc_idx(x, )
#         # w_rel, base_rel = self.calc_relation(x[:,:,None], t[:,None])
#         # w_rel = self.update_relation(w_rel, self._theta_relation)

#         # sort by the numerator first to prioritize 
#         # items that have a larger numerator but are fequal
#         # _, base_sorted_indices = base_rel.sort(dim=0)
#         # w_rel_sorted = w_rel.gather(1, base_sorted_indices)

#         # _, w_rel_indices = w_rel.sort(dim=0)
#         # w_sorted, _ = w.data.sort(dim=0)
#         # w_sorted_rel = w.data.gather(0, w_rel_indices) # w.data[w_rel_indices]
#         # print('Sorted: ', w_sorted[1], w_sorted_rel[1])
#         # print('W: ', w_sorted, w_sorted_rel)
#         # y_w_rel = self.outer(self.inner(x[:,:,None], w_rel))[0]

#         # chosen, idx = self.calc_chosen(x, w_rel)
#         # chosen, idx2 = self.calc_chosen(x, w.data)
#         # chosen = self.calc_chosen(x, w.data)
#         # chosen = self.set_chosen(x, w.data, idx2)
#         # print(~(y < t))
#         y = y[:,None]
#         w = w[None]
#         x = x[:,:,None].detach()
#         t = t[:,None].detach()

#         output_less_than = y < t

#         x_less_than_t = x < t

#         inner_greater_than = torch.min(x, w) > t
#         # 
#         w_target = (w + (t - y)).detach()
#         w_target2 = torch.min(w_target, x.detach())
#         return (
#             self._fuzzy_comp.forward(
#                 w, t, mask=inner_greater_than
#             )
#             # investigate why there needs to be two testing for less than and
#             # if i can get rid of one
#             + self._fuzzy_comp.forward(
#                 w, w_target2, mask=x_less_than_t & output_less_than & (x > w) # & chosen
#             )
#             + self._fuzzy_comp.forward(
#                 w, w_target, mask=~x_less_than_t & output_less_than # & chosen
#             )
#             # i think including this isl
#             # + ((w_sorted_rel - w_sorted.detach()) ** 2).mean()
#             # + self._fuzzy_comp_to_all.forward(
#             #     w, x, chosen
#             # )
#         )

#     def inner(self, x: torch.Tensor, w: torch.Tensor):
#         return torch.min(x, w)

#     def outer(self, x: torch.Tensor, dim: int=-2, keepdim: bool=False):
#         return torch.max(x, dim=dim, keepdim=keepdim)




# class MinMaxLoss(FuzzyCompLoss):

#     def __init__(self, minmax: fuzzy.MinMax, relation_lr: float = 1, reduction='mean', default_optim: ToOptim=ToOptim.BOTH):
#         super().__init__(relation_lr, reduction)
#         self.minmax = minmax
#         self.default_optim = default_optim

#     def forward(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
#         losses = []
#         if self.default_optim.x():
#             losses.append(self.forward_x(x, y, t))
#         if self.default_optim.theta():
#             losses.append(self.forward_theta(x, y, t))
#         return sum(losses)

#     def forward_theta(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        
#         x, y, t = x.data, y.data, t.data
#         if self.minmax.to_complement:
#             x = torch.cat([x, 1 - x], dim=1)

#         w = self.minmax.weight

#         rel_idx = self.calc_idx(x, self.calc_relation(x[:,:,None], t[:,None], agg_dim=0))
#         chosen = self.set_chosen(x, w, rel_idx)

#         y = y[:,None]
#         w = w[None]
#         x = x[:,:,None].detach()
#         t = t[:,None].detach()

#         output_less_than = y < t

#         return (
#             self._fuzzy_comp.forward(
#                 w, t, mask=~output_less_than
#             )
#             + self._fuzzy_comp.forward(
#                 w, x, mask=chosen & (x > t) & output_less_than & (x < w)
#             )
#             + self._fuzzy_comp.forward(
#                 w, t, mask=chosen & (x < t) & output_less_than
#             )
#             + self._fuzzy_comp_to_all.forward(
#                 w, x, chosen
#             )
#         )

#     def forward_x(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        
#         x, y, t = x.data, y.data, t.data
#         if self.minmax.to_complement:
#             x = torch.cat([x, 1 - x], dim=1)
    
#         w = self.minmax.weight

#         rel_idx = self.calc_idx(self.calc_relation(w[None], t[:,None], agg_dim=2), w)
#         chosen = self.set_chosen(x, w, rel_idx)

#         y = y[:,None]
#         w = w[None].detach()
#         x = x[:,:,None]
#         t = t[:,None].detach()

#         output_greater_than = y > t
#         output_less_than = ~output_greater_than

#         # TODO: Need to increase if chosen is greater than but output is less than
#         return (
#             self._fuzzy_comp.forward(
#                 x, t, mask=~output_greater_than
#             )
#             + self._fuzzy_comp.forward(
#                 x, w, mask=chosen & (w > t) & output_less_than & (w < x)
#             )
#             + self._fuzzy_comp.forward(
#                 x, t, mask=chosen & (w < t) & output_less_than
#             )
#             + self._fuzzy_comp_to_all.forward(
#                 x, w, chosen
#             )
#         )

#     def inner(self, x: torch.Tensor, w: torch.Tensor):
#         return torch.max(x, w)

#     def outer(self, x: torch.Tensor, dim: int=-2, keepdim: bool=False):
#         return torch.min(x, dim=dim, keepdim=keepdim)
    

# class MaxProdLoss(FuzzyCompLoss):
#     """TODO: Implement
#     """

#     def __init__(self, maxprod: fuzzy.MaxProd, relation_lr: float = 1, reduction='mean'):
#         super().__init__(relation_lr, reduction)
#         self.maxprod = maxprod

#     def forward_x(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        
#         x, y, t = x.data, y.data, t.data
#         if self.maxprod.to_complement:
#             x = torch.cat([x, 1 - x], dim=1)
#         w = self.maxprod.weight

#         rel_idx = self.calc_idx(self.calc_relation(w[None], t[:,None], agg_dim=2), w)
#         chosen = self.set_chosen(x, w, rel_idx)

#         y = y[:,None]
#         w = w[None].detach()
#         x = x[:,:,None]
#         t = t[:,None].detach()

#         output_greater_than = y > t

#         return (
#             self._fuzzy_comp.forward(
#                 x, t, mask=~output_greater_than
#             )
#             + self._fuzzy_comp.forward(
#                 x, w, mask=chosen & (w > t) & output_greater_than & (w < x)
#             )
#             + self._fuzzy_comp.forward(
#                 x, t, mask=chosen & (w < t) & output_greater_than
#             )
#             + self._fuzzy_comp_to_all.forward(
#                 x, w, chosen
#             )
#         )

#     def forward_theta(self, x: fuzzy.FuzzySet, y: fuzzy.FuzzySet, t: fuzzy.FuzzySet) -> torch.Tensor:
        
#         x, y, t = x.data, y.data, t.data
#         if self.maxprod.to_complement:
#             x = torch.cat([x, 1 - x], dim=1)
#         w = self.minmax.weight

#         rel_idx = self.calc_idx(x, self.calc_relation(x[:,:,None], t[:,None], agg_dim=0))
#         chosen = self.set_chosen(x, w, rel_idx)

#         y = y[:,None]
#         w = w[None]
#         x = x[:,:,None].detach()
#         t = t[:,None].detach()

#         output_less_than = y < t

#         return (
#             self._fuzzy_comp.forward(
#                 w, t, mask=~output_less_than
#             )
#             + self._fuzzy_comp.forward(
#                 w, x, mask=chosen & (x > t) & output_less_than & (x < w)
#             )
#             + self._fuzzy_comp.forward(
#                 w, t, mask=chosen & (x < t) & output_less_than
#             )
#             + self._fuzzy_comp_to_all.forward(
#                 w, x, chosen
#             )
#         )
    
#     def inner(self, x: torch.Tensor, w: torch.Tensor):
#         return x * w

#     def outer(self, x: torch.Tensor, dim: int=-2, keepdim: bool=False):
#         return torch.max(x, dim=dim, keepdim=keepdim)
        
