# %%

import initialize
import torch.nn as nn
import torch
import zenkai
from mistify.learn._infer import (
    MaxMin, MaxMinLoss, MaxMinSortedPredictorLoss, MaxMinPredictorLoss,
    MinMaxLoss, MinMax, MinMaxPredictorLoss, MinMaxSortedPredictorLoss
)
from torch.utils import data as torch_data
from zenkai import ThLoss, IO


# %%


# %%


# %%

X = torch.rand(2000, 10)

maxmin_t = MaxMin(10, 20)
T = maxmin_t(X).detach()

maxmin = MaxMin(10, 20)

maxmin_loss = MaxMinLoss(maxmin)
predictor_loss = MaxMinSortedPredictorLoss(maxmin, maxmin_t)

# %%

from torch.utils.data import DataLoader, TensorDataset

optim = torch.optim.Adam(maxmin.parameters(), lr=1e-3)
data = TensorDataset(X, T)


for i in range(1):
    for x_i, t_i in DataLoader(data, batch_size=2000, shuffle=True):

        optim.zero_grad()
        y_i = maxmin(x_i)
        loss = maxmin_loss(x_i, y_i, t_i) + 0.001 * predictor_loss(x_i, y_i, t_i)
        loss.backward()
        optim.step()
        mse = (y_i - t_i).pow(2).mean()
        if (i + 1) % 50 == 0:
            print(f'{i}: {mse.item()}')
        zenkai.utils.apply_to_parameters(
            maxmin.parameters(), lambda x: x.clamp(0, 1)
        )

        # for p in maxmin.parameters():
        #    print(p.grad)
        # print(loss.item())
    # if i == 5000:
    #    print('Changing Loss')
    #    maxmin_loss = MaxMinLoss3(maxmin, maxmin_t, local_y=False)

# loss = maxmin_loss(x, y, t)
# print(loss)

# %%

class MaxMinLearner(zenkai.kikai.CriterionNNAdapt):

    def __init__(self, in_features: int, out_features: int):
        
        maxmin = MaxMin(in_features, out_features)
        super().__init__(
            maxmin
        )
        self.maxmin_loss = MaxMinLoss(maxmin, 'sum')
        self.predictor_loss = MaxMinSortedPredictorLoss(maxmin, 'sum')
        self.to_use_predictor = True

    
    def assess(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reduction_override: str = None) -> torch.Tensor:

        loss1 = self.maxmin_loss(IO(x), IO(y), IO(t), reduction_override)
        if self.to_use_predictor:
            loss1 = loss1 + 0.001 * self.predictor_loss(IO(x), IO(y), IO(t), reduction_override)
        return 0.5 * loss1


class MinMaxLearner(zenkai.kikai.CriterionNNAdapt):

    def __init__(self, in_features: int, out_features: int):
        
        minmax = MinMax(in_features, out_features)
        super().__init__(
            minmax
        )
        self.minmax_loss = MinMaxLoss(minmax, 'sum')
        self.predictor_loss = MinMaxSortedPredictorLoss(minmax, 'sum')
        self.to_use_predictor = True
    
    def assess(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, reduction_override: str = None) -> torch.Tensor:

        # loss1 = 0.5 * (y - t).pow(2).sum()
        loss1 = self.minmax_loss(IO(x), IO(y), IO(t), reduction_override)
        if self.to_use_predictor:
            # pass
            loss1 = loss1 + 0.001 * self.predictor_loss(IO(x), IO(y), IO(t), reduction_override)
        return 0.5 * loss1


class FuzzyNet(nn.Module):

    def __init__(self, in_features: int, mid_features: int, out_features: int):

        super().__init__()
        # maxmin1 = MaxMin(in_features, mid_features)
        # maxmin2 = MaxMin(mid_features, out_features)
        # self.layer1 = zenkai.kikai.CriterionNNAdapt(
        #     maxmin1, ThLoss('MSELoss', reduction='sum') # MaxMinLoss(maxmin1, reduction='sum')
        # )
        # self.layer2 = zenkai.kikai.CriterionNNAdapt(
        #     maxmin2, ThLoss('MSELoss', reduction='sum') # MaxMinLoss(maxmin2, reduction='sum')
        # )
        self.layer1 = MinMaxLearner(in_features, mid_features)
        self.layer2 = MinMaxLearner(mid_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.layer1(x)
        return self.layer2(y)


x = torch.randn(20000, 8)
base = FuzzyNet(8, 16, 4)
t = base(x).detach()
trainable = FuzzyNet(8, 16, 4)

optim = torch.optim.Adam(trainable.parameters(), lr=1e-3)

dataset = torch_data.TensorDataset(x, t)

for i in range(50):
    for x_i, t_i in torch_data.DataLoader(dataset, batch_size=128, shuffle=True):

        y_i = trainable(x_i)
        optim.zero_grad()
        loss = (y_i - t_i).pow(2).mean()
        loss.backward()
        
        print(i, loss.item())
        optim.step()

        zenkai.utils.apply_to_parameters(
            trainable.parameters(), lambda x: torch.clamp(x, 0, 1)
        )
    if i == 25:
        trainable.layer1.to_use_predictor = False
        # trainable.layer2.to_use_predictor = False


# %%



# class MaxMinSortedPredictorLoss2(nn.Module):

#     def __init__(self, maxmin: MaxMin, base_maxmin: MaxMin):

#         super().__init__()
#         self.maxmin = maxmin
#         self.w_local = maxmin.w.clone().detach()
#         self.base_maxmin = base_maxmin
    
#     def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

#         x = x.unsqueeze(-1)
#         y = y.unsqueeze(-2)
#         t = t.unsqueeze(-2)
#         w = self.maxmin.w.unsqueeze(0)
#         shape = list(w.shape)
#         shape[0] = x.shape[0]
#         positives = torch.min(x, t)
#         negatives = torch.relu(x - t)
#         score = positives.sum(dim=0, keepdim=True) / x.sum(dim=0, keepdim=True)

#         score[score.isnan()] = 1.0

#         inner2 = torch.min(x, score)
#         chosen_val = torch.max(inner2, dim=-2, keepdim=True)[0]

#         maximum = (inner2 == chosen_val).type_as(positives) * inner2
#         cur_w = maximum.sum(dim=0, keepdim=True) / (
#             maximum.sum(dim=0, keepdim=True) + negatives.sum(dim=0, keepdim=True)
#         )

#         _, sorted_score_indices = cur_w.sort(-2, True)

#         base_w2 = self.base_maxmin.w[None].gather(-2, sorted_score_indices)
#         y_base = torch.max(torch.min(base_w2, x), dim=-2, keepdim=True)[0]
#         print((y_base - t).pow(2).mean().item())

#         sorted_w_vals, _ = w.sort(-2, True)
#         target_w_vals = w.gather(-2, sorted_score_indices).detach()

#         return (sorted_w_vals - target_w_vals).pow(2).mean()




# class MaxMinLoss1(nn.Module):

#     def __init__(self, maxmin: MaxMin, base_maxmin: MaxMin, local_y: bool=True):

#         super().__init__()
#         self.maxmin = maxmin
#         self.w_local = maxmin.w.clone().detach()
#         self.local_y = local_y
#         self.base_maxmin = base_maxmin
    
#     def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

#         x = x.unsqueeze(-1)
#         y = y.unsqueeze(-2)
#         t = t.unsqueeze(-2)
#         w = self.maxmin.w.unsqueeze(0)
#         shape = list(w.shape)
#         shape[0] = x.shape[0]
#         # w = w + torch.randn(shape) * 0.025
#         inner = torch.min(x, w)
#         # inner = inner + torch.randn_like(inner) * 0.01
#         # greater_than_t = (inner > t)

#         # Calculate a weight matrix based on predictability
#         cur_w = torch.min(x, t).sum(dim=0, keepdim=True) / x.sum(dim=0, keepdim=True)
        
#         # 
#         # chosen / not chosen
#         # 
        
#         # i can try using only the ones it's chosen on and recalculating the
#         # sum... For now this does not work though

#         # 0.2 => 0.8
#         # 0.8 => 
        
#         # cur_w = torch.min(x.max(dim=0, keepdim=True)[0], cur_w)
#         self.w_local = cur_w # * 0.1 + self.w_local * 0.9
        
#         # Determine which weights should be used based on predictability
#         inner_validx = torch.max(torch.min(x, cur_w), dim=-2, keepdim=True)

#         idx = inner_validx[1].detach()
#         # multiplier = torch.zeros_like(inner)

#         idx2 = torch.max(torch.min(x, self.base_maxmin.w[None]), dim=-2, keepdim=True)[1].detach()
#         # print((idx == idx2).float().sum())
#         # print(idx2.numel())

#         local_y = inner.gather(-2, inner_validx[1])
#         # local_y = inner_validx[0].detach().squeeze(-2)
#         # ones = torch.ones(idx.shape, device=idx.device)
#         # multiplier = multiplier.scatter(
#         #     -2, idx, ones
#         # )
#         # w_local = multiplier * w
#         # print(w_local)
#         # calculate the losses between the ideal weight
#         # outputs and also the values that are greater than t
#         # local_y = torch.max(torch.min(x, w_local.detach()), dim=-2)[0]

#         greater_than = (torch.relu(inner - t)).pow(2).mean()
    
#         less_than = torch.relu(t - y).pow(2).mean()
#         # less_than = (t - y).pow(2).mean()
#         loss = less_than + greater_than
#         if self.local_y:
            
#             predictor = (local_y - t).pow(2).mean()
#             loss = loss + 0.5 * predictor 
#         return loss
    

# # %%
    
# class MaxMinLoss2(nn.Module):

#     def __init__(self, maxmin: MaxMin, base_maxmin: MaxMin, local_y: bool=True):

#         super().__init__()
#         self.maxmin = maxmin
#         self.w_local = maxmin.w.clone().detach()
#         self.local_y = local_y
#         self.base_maxmin = base_maxmin
    
#     def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

#         x = x.unsqueeze(-1)
#         y = y.unsqueeze(-2)
#         t = t.unsqueeze(-2)
#         w = self.maxmin.w.unsqueeze(0)
#         shape = list(w.shape)
#         shape[0] = x.shape[0]
#         inner = torch.min(x, w)
#         negatives = torch.relu(x - t)
#         positives = torch.min(x, t)
#         temp_w = positives.sum(dim=0, keepdim=True) / x.sum(dim=0, keepdim=True)

#         temp_w[temp_w.isnan()] = 1.0

#         inner2 = torch.min(x, temp_w)
#         chosen_val = torch.max(inner2, dim=-2, keepdim=True)[0]

#         # maximum = (inner2 == chosen_val).type_as(positives) + torch.zeros_like(inner2)
#         # cur_w = maximum.sum(dim=0, keepdim=True) / (
#         #     maximum.sum(dim=0, keepdim=True) + negatives.sum(dim=0, keepdim=True)
#         # )
#         maximum = (inner2 == chosen_val).type_as(positives) * inner2
#         cur_w = maximum.sum(dim=0, keepdim=True) / (
#             maximum.sum(dim=0, keepdim=True) + negatives.sum(dim=0, keepdim=True)
#         )

#         # Best to compare by val.. This will be faster though
#         inner_validx = torch.max(torch.min(x, cur_w), dim=-2, keepdim=True)
#         # idx = inner_validx[1].detach()
#         # idx2 = torch.max(torch.min(x, self.base_maxmin.w[None]), dim=-2, keepdim=True)[1].detach()
#         # print((idx == idx2).float().sum())
#         # print(idx2.numel())

#         local_y = inner.gather(-2, inner_validx[1])
#         # local_y = inner_validx[0].detach().squeeze(-2)

#         less_than_x = ((t < x) & (x < w)).type_as(w)
#         less_than_x_loss = (less_than_x * (w - t)).pow(2).mean()
#         greater_than = (torch.relu(inner - t)).pow(2).mean()
    
#         # 28677 are the same out of 40,000. It seems
#         # this is close to being optimal.. At any rate this is pretty damn good
#         # almost 3 / 4 of them are the same compared to 3 / 8.. is it possible this
#         # is also optimal?

#         # less_than = torch.relu(t - y).pow(2).mean()
#         less_than = (t - y).pow(2).mean()
#         loss = less_than + greater_than + less_than_x_loss
#         if self.local_y: 
#             predictor = (local_y - t).pow(2).mean()
#             loss = loss + 0.01 * predictor 
#         return loss

# # %%

# class MaxMinLoss3(nn.Module):

#     def __init__(self, maxmin: MaxMin, base_maxmin: MaxMin, local_y: bool=True):

#         super().__init__()
#         self.maxmin = maxmin
#         self.w_local = maxmin.w.clone().detach()
#         self.local_y = local_y
#         self.base_maxmin = base_maxmin
    
#     def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

#         x = x.unsqueeze(-1)
#         y = y.unsqueeze(-2)
#         t = t.unsqueeze(-2)
#         w = self.maxmin.w.unsqueeze(0)
#         shape = list(w.shape)
#         shape[0] = x.shape[0]
#         inner = torch.min(x, w)
#         negatives = torch.relu(x - t)
#         positives = torch.min(x, t)
#         score = positives.sum(dim=0, keepdim=True) / x.sum(dim=0, keepdim=True)

#         score[score.isnan()] = 1.0
#         # score = torch.min(x.max(dim=0, keepdim=False)[0], score)

#         _, sorted_score_indices = score.sort(-2, True)

#         base_w2 = self.base_maxmin.w[None].gather(-2, sorted_score_indices)
#         y_base = torch.max(torch.min(base_w2, x), dim=-2, keepdim=True)[0]
#         print((y_base - t).pow(2).mean())

#         sorted_w_vals, _ = w.sort(-2, True)
#         target_w_vals = w.gather(-2, sorted_score_indices).detach()

#         # t < x < w => 

#         less_than_x = ((t < x) & (x < w)).type_as(w)

#         less_than_x_loss = (less_than_x * (w - t)).pow(2).mean()

#         greater_than = (torch.relu(inner - t)).pow(2).mean()
    
#         # 28677 are the same out of 40,000. It seems
#         # this is close to being optimal.. At any rate this is pretty damn good
#         # almost 3 / 4 of them are the same compared to 3 / 8.. is it possible this
#         # is also optimal?

#         # less_than = torch.relu(t - y).pow(2).mean()
#         less_than = torch.relu(t - y).pow(2).mean()
#         loss = less_than + greater_than + less_than_x_loss
#         if self.local_y: 
#             predictor = (sorted_w_vals - target_w_vals).pow(2).mean()
#             loss = loss + 1e-5 * predictor 
#         return loss


# class WeightResort(object):

#     def __init__(self, maxmin: MaxMin, base_maxmin: MaxMin):

#         super().__init__()
#         self.maxmin = maxmin
#         self.w_local = maxmin.w.clone().detach()
#         self.base_maxmin = base_maxmin
    
#     def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

#         x = x.unsqueeze(-1)
#         t = t.unsqueeze(-2)
#         w = self.maxmin.w.unsqueeze(0)
#         shape = list(w.shape)
#         shape[0] = x.shape[0]
#         inner = torch.min(x, w)
#         negatives = torch.relu(x - t)
#         positives = torch.min(x, t)
#         score = positives.sum(dim=0, keepdim=True) / x.sum(dim=0, keepdim=True)

#         score[score.isnan()] = 1.0
#         # score = torch.min(x.max(dim=0, keepdim=False)[0], score)

#         _, sorted_score_indices = score.sort(-2, True)
#         # sorted_w_vals, _ = w.sort(-2, True)
#         target_w_vals = w.gather(-2, sorted_score_indices).detach()
#         self.maxmin.w.data = target_w_vals.squeeze(0)
