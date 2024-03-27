import torch.nn as nn
import torch
import zenkai
from zenkai import XCriterion, IO

import torch.nn as nn
import torch
import zenkai
from ..infer import Or, And, InterOnBase, UnionOnBase


class MaxMinLoss(nn.Module):

    def __init__(self, maxmin: Or, reduction='mean'):

        super().__init__()
        self.maxmin = maxmin
        self.reduction = reduction
        self.reducer = zenkai.Reduction[reduction]
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reduction_override: str=None) -> torch.Tensor:

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-2)
        t = t.unsqueeze(-2)
        t = t.clamp(0, 1)
        w = self.maxmin.w().unsqueeze(0)
        shape = list(w.shape)
        shape[0] = x.shape[0]
        inner = torch.min(x, w)

        chosen_idx = inner.max(dim=-2, keepdim=True)[1]
        
        # Only necessary for the first layer
        less_than_x = ((t < w.detach()) & (w.detach() < x)).type_as(w)
        less_than_x_loss = self.reducer.reduce(torch.gather(less_than_x * (x - t), -2, chosen_idx).pow(2), reduction_override=reduction_override)

        less_than_theta = ((t < x.detach()) & (x.detach() < w)).type_as(w)
        less_than_theta_loss = self.reducer.reduce(torch.gather(less_than_theta * (w - t), -2, chosen_idx).pow(2), reduction_override=reduction_override)

        greater_than = self.reducer.reduce((torch.relu(inner - t)).pow(2), reduction_override=reduction_override)
        less_than = self.reducer.reduce(torch.relu(t - y).pow(2), reduction_override=reduction_override)
        loss = less_than + greater_than # + less_than_x_loss + less_than_theta_loss
        return loss

    
class MaxMinPredictorLoss(nn.Module):

    def __init__(self, maxmin: Or, reduction: str='mean'):

        super().__init__()
        self.maxmin = maxmin
        self.w_local = maxmin.w().clone().detach()
        self.reducer = zenkai.Reduction[reduction]
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reduction_override: str=None) -> torch.Tensor:

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-2)
        t = t.unsqueeze(-2)
        w = self.maxmin.w().unsqueeze(0)
        shape = list(w.shape)
        shape[0] = x.shape[0]
        inner = torch.min(x, w)
        negatives = torch.relu(x - t)
        positives = torch.min(x, t)
        temp_w = positives.sum(dim=0, keepdim=True) / x.sum(dim=0, keepdim=True)

        temp_w[temp_w.isnan()] = 1.0

        inner2 = torch.min(x, temp_w)
        chosen_val = torch.max(inner2, dim=-2, keepdim=True)[0]

        maximum = (inner2 == chosen_val).type_as(positives) * inner2
        cur_w = maximum.sum(dim=0, keepdim=True) / (
            maximum.sum(dim=0, keepdim=True) + negatives.sum(dim=0, keepdim=True)
        )

        # Best to compare by val.. This will be faster though
        inner_validx = torch.max(torch.min(x, cur_w), dim=-2, keepdim=True)

        local_y = inner.gather(-2, inner_validx[1])
        return self.reducer.reduce((local_y - t).pow(2), reduction_override=reduction_override)


class MaxMinSortedPredictorLoss(nn.Module):

    def __init__(self, maxmin: Or, reduction: str='mean'):

        super().__init__()
        self.maxmin = maxmin
        self.score_local = None
        # self.base_maxmin = base_maxmin
        self.reducer = zenkai.Reduction[reduction]
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reduction_override: str=None) -> torch.Tensor:

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-2)
        t = t.unsqueeze(-2)
        w = self.maxmin.w().unsqueeze(0)
        shape = list(w.shape)
        shape[0] = x.shape[0]
        positives = torch.min(x, t)
        score = positives.sum(dim=0, keepdim=True) / x.sum(dim=0, keepdim=True)

        score[score.isnan()] = 1.0

        if self.score_local is not None:
            self.score_local = 0.9 * self.score_local + 0.1 * score
        else:
            self.score_local = score

        _, sorted_score_indices = self.score_local.sort(-2, True)

        # base_w2 = self.base_maxmin.w[None].gather(-2, sorted_score_indices)
        # y_base = torch.max(torch.min(base_w2, x), dim=-2, keepdim=True)[0]
        # print((y_base - t).pow(2).mean())

        sorted_w_vals, _ = w.sort(-2, True)
        target_w_vals = w.gather(-2, sorted_score_indices).detach()

        return self.reducer.reduce((sorted_w_vals - target_w_vals).pow(2), reduction_override=reduction_override)


class MinMax(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w = nn.parameter.Parameter(
            torch.rand(in_features, out_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.min(torch.max(x.unsqueeze(-1), self.w[None]), dim=-2)[0]


class MinMaxLoss(XCriterion):

    def __init__(self, minmax: And, reduction: str='mean'):

        super().__init__()
        self.minmax = minmax
        self.reducer = zenkai.Reduction[reduction]
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reduction_override: str=None) -> torch.Tensor:

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-2)
        t = t.unsqueeze(-2)
        w = self.minmax.w().unsqueeze(0)

        shape = list(w.shape)
        shape[0] = x.shape[0]
        inner = torch.max(x, w)
        
        greater_than_x = ((t > x.detach()) & (x.detach() > w)).type_as(w)
        greater_than_x_loss = self.reducer.reduce((greater_than_x * (w - t)).pow(2), reduction_override=reduction_override)

        greater_than_theta = ((t > w.detach()) & (w.detach() > x)).type_as(w)
        greater_than_theta_loss = self.reducer.reduce((greater_than_theta * (x - t)).pow(2), reduction_override=reduction_override)

        # Is this okay?
        less_than = self.reducer.reduce((torch.relu(t - inner)).pow(2), reduction_override=reduction_override)
        greater_than = self.reducer.reduce(torch.relu(y - t).pow(2), reduction_override=reduction_override)
        loss = less_than + greater_than # + greater_than_x_loss + greater_than_theta_loss
        return loss


class MinMaxPredictorLoss(nn.Module):

    def __init__(self, minmax: And, reduction: str='mean'):

        super().__init__()
        self.minmax = minmax
        self.score_local = None
        self.reducer = zenkai.Reduction[reduction]
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reduction_override: str=None) -> torch.Tensor:

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-2)
        t = t.unsqueeze(-2)
        w = self.minmax.w().unsqueeze(0)
        shape = list(w.shape)
        shape[0] = x.shape[0]
        x_comp = 1 - x
        inner = torch.min(x_comp, 1 - w)
        positives = torch.min(1 - x, 1 - t)
        x_comp = 1 - x

        # find out which of the xs
        # correspond to 
        score = 1 - (positives.sum(
            dim=0, keepdim=True
        )) / x_comp.sum(dim=0, keepdim=True)
        if self.score_local is not None:
            self.score_local = 0.9 * self.score_local + 0.1 * score
        else:
            self.score_local = score

        score[score.isnan()] = 0.0

        inner2 = torch.max(x, score)
        chosen_val = torch.min(inner2, dim=-2, keepdim=True)[0]

        minimum = (inner2 == chosen_val).type_as(positives) * inner2

        cur_w = 1 - ((1 - minimum.sum(dim=0, keepdim=True)) / (
            (1 - minimum.sum(dim=0, keepdim=True)) + positives.sum(dim=0, keepdim=True)
        ))

        # # Looks like this
        # # 
        # output = 0.0... Need to find out the cases
        # x = 0.0, t=0.0
        # x = 1.0, 
        #  # 
        # # # check which ones are in bounds and out of bounds
        # # # average based on the the targets that are "out of bounds"
        # # # continue this
        # (1 - x) and (1 - t) / sum(1 - x)
        #
        # sum(1 - t) / sum((1 - x) or (1 - t) ) 
        # t is 0 / (t is 0 or x is 0)


        # # Negative prediction
        #  sum(t) / union(x, t) <- don't do this
        # x = 1.0 t = 0.0
        # x = 1.0 t = 1.0

        # Best to compare by val.. This will be faster though
        inner_validx = torch.min(torch.max(x, cur_w), dim=-2, keepdim=True)

        local_y = inner.gather(-2, inner_validx[1])
        return self.reducer.reduce((local_y - t).pow(2), reduction_override=reduction_override)


class MinMaxSortedPredictorLoss(nn.Module):

    def __init__(self, minmax: And, reduction: str='mean'):

        super().__init__()
        self.minmax = minmax
        self.score_local = None
        self.reducer = zenkai.Reduction[reduction]
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reduction_override: str=None) -> torch.Tensor:

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-2)
        t = t.unsqueeze(-2)
        w = self.minmax.w().unsqueeze(0)
        shape = list(w.shape)
        shape[0] = x.shape[0]
        # negatives = torch.relu(x - t)
        positives = torch.min(1 - x, 1 - t)
        x_comp = 1 - x

        score = 1 - (positives.sum(
            dim=0, keepdim=True
        )) / x_comp.sum(dim=0, keepdim=True)

        score[score.isnan()] = 0.0
        if self.score_local is not None:
            self.score_local = 0.9 * self.score_local + 0.1 * score
        else:
            self.score_local = score

        _, sorted_score_indices = self.score_local.sort(-2, True)

        # base_w2 = self.base_minmax.w[None].gather(-2, sorted_score_indices)
        # y_base = torch.min(torch.max(base_w2, x), dim=-2, keepdim=True)[0]
        # print((y_base - t).pow(2).mean())

        sorted_w_vals, _ = w.sort(-2, True)
        target_w_vals = w.gather(-2, sorted_score_indices).detach()

        return self.reducer.reduce((sorted_w_vals - target_w_vals).pow(2), reduction_override=reduction_override)


class IntersectionOnLoss(nn.Module):

    def __init__(self, intersection_on: InterOnBase, reduction: str='mean'):

        super().__init__()
        self.intersection_on = intersection_on
        self.reducer = zenkai.Reduction[reduction]
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reduction_override: str=None) -> torch.Tensor:

        y = y.unsqueeze(self.intersection_on.dim)
        t = t.unsqueeze(self.intersection_on.dim)
        # x_max > t => only use x_max
        # x_max < t => use all xs that are less than t
        greater_than = self.reducer.reduce(torch.relu(y - t).pow(2), reduction_override)
        less_than = self.reducer.reduce(torch.relu(t - x).pow(2), reduction_override)
        return greater_than + less_than


class UnionOnLoss(nn.Module):

    def __init__(self, union_on: UnionOnBase, reduction: str='mean'):

        super().__init__()
        self.union_on = union_on
        self.reducer = zenkai.Reduction[reduction]
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reduction_override: str=None) -> torch.Tensor:

        # x_max > t => only use x_max
        # x_max < t => use all xs that are less than t
        less_than = self.reducer.reduce(torch.relu(t - y).pow(2), reduction_override)
        greater_than = self.reducer.reduce(torch.relu(x - t.unsqueeze(self.union_on.dim)).pow(2), reduction_override)
        return greater_than + less_than
