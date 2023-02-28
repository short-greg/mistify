# 1st Party
from abc import abstractmethod
from enum import Enum

# 3rd Party
import torch
import torch.nn as nn


# Local
from ..utils import reduce
from .. import conversion
from .. import fuzzy, binary
import torch.nn.functional as nn_func
import typing
# The performance of these will be slow so I possibly
# need to split up the calculation

# class TMistifyLoss(nn.Module):
#     """Wraps a module and a loss together so that more advanced backpropagation
#     can be implemented
#     """

#     def __init__(self, module: nn.Module, loss: MistifyLoss):

#         super().__init__()
#         self._module = module
#         self._loss = loss

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self._module.forward(x)

#     def assess(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
#         return self._loss.forward(x, y, t)


# class LossGrad(torch.autograd.Function):
#     """Use to implement backpropagation that is closer to the standard 
#     form that sends information in a direct path backward through the layers
#     """

#     # TODO: consider making this more flexible
#     # Example: forward(*x, loss: ModLoss)
#     @staticmethod
#     def forward(ctx, x: torch.Tensor, loss: MistifyLoss):

#         x.requires_grad_(True)
#         ctx.loss = loss
#         # print('calling forward')
#         with torch.enable_grad():
#             y = ctx.loss.module(x)
#         ctx.save_for_backward(x, y)
#         return y

#     @staticmethod
#     def backward(ctx, dx: torch.Tensor):
#         # print('DX: ', dx[0])

#         with torch.enable_grad():
#             x, y = ctx.saved_tensors
#             # x = x.detach().requires_grad_()
#             # x.retain_grad()
#             ctx.loss(x, y, (y - dx)).backward()

#         return x.grad.detach(), None





# class FuzzyCompToAllLoss(nn.Module):

#     def __init__(self, inner, outer, reduction='mean'):
#         super().__init__()
#         self.reduction = reduction
#         self.inner = inner
#         self.outer = outer

#     def forward(self, x: torch.Tensor, t: torch.Tensor, chosen: torch.Tensor):

#         chosen_clone = chosen.clone()
#         chosen_clone[chosen == 1] = -torch.inf
#         chosen_clone[chosen == 0] = 1.0

#         # input_difference[chosen.long()] = -torch.inf
#         # TODO: Figure out how to handle this
#         result = self.outer(
#             (t - x) * chosen_clone, dim=0)[0]
#         # , torch.tensor(0.0)
#         # )
#         result[result.isinf()] = 0.0
#         return reduce(result.mean(), self.reduction)


# class FuzzyLoss(nn.Module):
    
#     def __init__(self, reduction='samplemean'):
#         super().__init__()
#         self.reduction = reduction

#     def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.BoolTensor):
#         return reduce(
#             torch.pow(((x - t.detach()) * mask.float()), 2), 
#             self.reduction
#         )


# class BinaryThetaLoss(MistifyLoss):

#     def __init__(self, lr: float=0.5):
#         """initializer

#         Args:
#             binary (Binary): Composition layer to optimize
#             lr (float, optional): learning rate value between 0 and 1. Defaults to 0.5.
#         """
#         self.lr = lr

#     def _calculate_positives(self, x: torch.Tensor, t: torch.Tensor):
#         positive = (t == 1)
#         return (
#             (x[:,:,None] == t[:,None]) & positive[:,None]
#         ).type_as(x).sum(dim=0)

#     def _calculate_negatives(self, x: torch.Tensor, t: torch.Tensor):
#         negative = (t != 1)
#         return (
#             (x[:,:,None] != t[:,None]) & negative[:,None]
#         ).type_as(x).sum(dim=0)
    
#     def _update_score(self, score, positives: torch.Tensor, negatives: torch.Tensor):
#         cur_score = positives / (negatives + positives)
#         cur_score[cur_score.isnan() | cur_score.isinf()] = 0.0
#         if score is not None and self.lr is not None:
#             return (1 - self.lr) * score + self.lr * cur_score
#         return cur_score
    
#     def _calculate_maximums(self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor):
#         positive = (t == 1)
#         y: torch.Tensor = torch.max(torch.min(x[:,:,None], score[None]), dim=1)[0]
#         return ((score[None] == y[:,None]) & positive[:,None]).type_as(x).sum(dim=0)
    
#     def _update_weight(self, relation: BinaryComposition, maximums: torch.Tensor, negatives: torch.Tensor):
#         cur_weight = maximums / (maximums + negatives)
#         cur_weight[cur_weight.isnan() | cur_weight.isinf()] = 0.0
#         return cur_weight
#         # if self.lr is not None:
#         #     relation.weight = nn.parameter.Parameter(
#         #         (1 - self.lr) * relation.weight + self.lr * cur_weight
#         #     )
#         # else:
#         #     relation.weight = cur_weight

#     def forward(self, relation: BinaryComposition, x: torch.Tensor, t: torch.Tensor, state: dict):
#         # TODO: Ensure doesn't need to be mean
#         score = state['score']
#         with torch.no_grad():
#             positives = self._calculate_positives(x, t)
#             negatives = self._calculate_negatives(x, t)
#             score = self._update_score(score, positives, negatives)
#             state['score'] = score
#             maximums = self._calculate_maximums(x, t, score)
#             target_weight = self._update_weight(relation, maximums, negatives)
#         return self._reduction((target_weight - relation.weight).abs())


# class BinaryXLoss(MistifyLoss):

#     def __init__(self, lr: float=None):
#         """initializer

#         Args:
#             lr (float, optional): learning rate value between 0 and 1. Defaults to 0.5.
#         """
#         self.lr = lr
    
#     def _calculate_positives(self, w: torch.Tensor, t: torch.Tensor):
#         positive = (t == 1)
#         return (
#             (w[None] == t[:,None]) & positive[:,None]
#         ).type_as(w).sum(dim=2)

#     def _calculate_negatives(self, w: torch.Tensor, t: torch.Tensor):
#         negative = (t != 1)
#         return (
#             (w[None] != t[:,None]) & negative[:,None]
#         ).type_as(w).sum(dim=2)
    
#     def _calculate_score(self, positives: torch.Tensor, negatives: torch.Tensor):
#         cur_score = positives / (negatives + positives)
#         cur_score[cur_score.isnan()] == 0.0
#         return cur_score
    
#     def _calculate_maximums(self, score: torch.Tensor, w: torch.Tensor, t: torch.Tensor):
#         positive = (t == 1)
        
#         y: torch.Tensor = torch.max(torch.min(w[None,], score[:,:,None]), dim=1)[0]
#         return ((score[:,:,None] == y[:,None]) & positive[:,None]).type_as(score).sum(dim=2)
    
#     def _update_base_inputs(self, binary: BinaryComposition, maximums: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor):

#         if binary.to_complement:
#             maximums.view(maximums.size(0), 2, -1)
#             negatives.view(negatives.size(0), 2, -1)
#             # only use the maximums + negatives
#             # is okay for other positives to be 1 since it is an
#             # "or" neuron
#             base = (
#                 maximums[:,0] / (maximums[:,0] + negatives[:,0])
#             )
#             # negatives for the complements must have 1 as the input 
#             complements = (
#                 negatives[:,1] / (positives[:,1] + negatives[:,1])
#             )

#             return ((0.5 * complements + 0.5 * base))

#         cur_inputs = (maximums / (maximums + negatives))
#         cur_inputs[cur_inputs.isnan()] = 0.0
#         return cur_inputs
    
#     def _update_inputs(self, state: dict, base_inputs: torch.Tensor):
#         if self.lr is not None:
#             base_inputs = (1 - self.lr) * state['base_inputs'] + self.lr * base_inputs        
#         return (base_inputs >= 0.5).type_as(base_inputs)

#     def forward(self, binary: BinaryComposition, x: torch.Tensor, t: torch.Tensor, state: dict):

#         # TODO: Update so it is a "loss"
#         w = binary.weight
#         with torch.no_grad():
#             positives = self._calculate_positives(w, t)
#             negatives = self._calculate_negatives(w, t)
#             score = self._calculate_score(positives, negatives)
#             maximums = self._calculate_maximums(score, w, t)
#             base_inputs = self._update_base_inputs(binary, maximums, positives, negatives)
#             x_prime = self._update_inputs(state, base_inputs)
#         return self._reduction((x_prime - x).abs())



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
        
