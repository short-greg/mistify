from typing import Tuple
import torch
import torch.nn as nn
import zenkai
import typing
import torch.nn as nn
import torch.utils.data
import numpy as np
from mistify import smooth_inter_on, smooth_union_on


class MinMax(nn.Module):

    def __init__(self, in_features: int, out_features: int):

        super().__init__()
        self.weight = nn.parameter.Parameter(torch.rand(
            in_features, out_features
        ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.min(torch.max(
            x[...,None], self.weight[None]
        ), dim=-2)[0]


class MaxMin(nn.Module):

    def __init__(self, in_features: int, out_features: int):

        super().__init__()
        self.weight = nn.parameter.Parameter(torch.rand(
            in_features, out_features
        ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.max(torch.min(
            x[...,None], self.weight[None]
        ), dim=-2)[0]



class SmoothSTEMinMax(nn.Module):

    def __init__(self, in_features: int, out_features: int, a=5):

        super().__init__()
        self.weight = nn.parameter.Parameter(torch.rand(
            in_features, out_features
        ))
        self.a = a
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        inner = torch.max(
            x[...,None], self.weight[None]
        )
        smooth = smooth_inter_on(inner, dim=-2, a=self.a)
        hard = torch.min(inner, dim=-2)[0]
        return (
            hard - smooth
        ).detach() + smooth


class SmoothSTEMaxMin(nn.Module):

    def __init__(self, in_features: int, out_features: int, a=5):

        super().__init__()
        self.weight = nn.parameter.Parameter(torch.rand(
            in_features, out_features
        ))
        self.a = a
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        inner = torch.min(
            x[...,None], self.weight[None]
        )
        smooth = smooth_union_on(inner, dim=-2, a=self.a)
        hard = torch.max(inner, dim=-2)[0]
        return (
            hard - smooth
        ).detach() + smooth

# def calc_chosen(x, w):

#     inner = torch.min(x, w)
#     y = torch.max(inner, keepdim=True, dim=-2)[0]
#     chosen = (y == inner)
#     return y, chosen

def max_min_chosen(x, w):

    inner = torch.min(x, w)
    y = torch.max(
        inner, keepdim=True, dim=-2
    )[0]
    chosen = (y == inner)
    return y, chosen


def min_max_chosen(x, w):

    inner = torch.max(x, w)
    y = torch.min(
        inner, keepdim=True, dim=-2
    )[0]
    chosen = (y == inner)
    return y, chosen


def max_min_rel(x, t, chosen=None, dim=0):

    negatives = torch.relu(x - t)
    positives = torch.min(x, t)
    if chosen is not None:
        positives = chosen * positives

    return (
        positives.sum(dim=dim, keepdim=True) / 
        (positives.sum(dim=dim, keepdim=True) + negatives.sum(dim=dim, keepdim=True))
    )


def min_max_rel(x, t, chosen=None, dim=0):

    x = 1 - x
    t = 1 - t
    negatives = torch.relu(x - t)
    positives = torch.min(x, t)
    if chosen is not None:
        positives = chosen * positives

    return 1 - (
        positives.sum(dim=dim, keepdim=True) / 
        (positives.sum(dim=dim, keepdim=True) + negatives.sum(dim=dim, keepdim=True))
    )


class MinMaxWRel2(nn.Module):

    def forward(self, x: torch.Tensor, t: torch.Tensor):

        x = x[...,None]
        t = t.unsqueeze(-2)

        w_rel = min_max_rel(x, t, None, dim=0)
        _, chosen_rel = min_max_chosen(x, w_rel)

        w_rel = min_max_rel(x, t, chosen_rel, dim=0)
        _, chosen_rel = min_max_chosen(x, w_rel)
        return chosen_rel


class MinMaxXRel2(nn.Module):

    def forward(self, w: torch.Tensor, t: torch.Tensor):

        w = w[None]
        t = t.unsqueeze(-2)

        x_rel = min_max_rel(w, t, None, dim=-1)
        _, chosen_rel = min_max_chosen(x_rel, w)

        x_rel = min_max_rel(w, t, chosen_rel, dim=-1)
        _, chosen_rel = min_max_chosen(x_rel, w)
        return chosen_rel


class MaxMinWRel2(nn.Module):

    def forward(self, x: torch.Tensor, t: torch.Tensor):

        x = x[...,None]
        t = t.unsqueeze(-2)

        w_rel = max_min_rel(x, t, None, dim=0)
        _, chosen_rel = max_min_chosen(x, w_rel)

        w_rel = max_min_rel(x, t, chosen_rel, dim=0)
        _, chosen_rel = max_min_chosen(x, w_rel)
        return chosen_rel


class MaxMinXRel2(nn.Module):

    def forward(self, w: torch.Tensor, t: torch.Tensor):

        w = w[None]
        t = t.unsqueeze(-2)

        x_rel = max_min_rel(w, t, None, dim=-1)
        _, chosen_rel = max_min_chosen(x_rel, w)

        x_rel = max_min_rel(w, t, chosen_rel, dim=-1)
        _, chosen_rel = max_min_chosen(x_rel, w)
        return chosen_rel


class MinMaxLoss(nn.Module):

    def __init__(
        self, min_max: MinMax, reduction: str, rel_reduction: str,
        x_weight: float=1.0, w_weight: float=1.0
    ):

        super().__init__()
        self.min_max = min_max
        self.min_max_wrel2 = MinMaxWRel2()
        self.min_max_xrel2 = MinMaxXRel2()
        self.reduction = reduction
        self.rel_reduction = rel_reduction
        self.x_weight = x_weight
        self.w_weight = w_weight
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):

        base_loss = zenkai.reduce((y - t).pow(2), self.reduction)
        chosen_w = self.min_max_wrel2(x, t)
        inner_w = torch.max(
            x[...,None].detach(), 
            self.min_max.weight[None]
        )
        w_loss = zenkai.reduce(
            ((inner_w - t.unsqueeze(-2)).pow(2) * chosen_w),
            self.rel_reduction
        ) * self.w_weight
        inner_x = torch.max(
            x[...,None], self.min_max.weight[None].detach()
        )
        chosen_x = self.min_max_xrel2(self.min_max.weight, t)
        x_loss = zenkai.reduce(
            ((inner_x - t.unsqueeze(-2)).pow(2) * chosen_x),
            self.rel_reduction
        ) * self.x_weight
        # print(base_loss.shape, w_loss.shape, x_loss.shape)
        return (
            base_loss + x_loss + w_loss
        )


class MaxMinLoss(nn.Module):

    def __init__(
        self, max_min: MaxMin, reduction: str, rel_reduction: str,
        x_weight: float=1.0, w_weight: float=1.0
    ):

        super().__init__()
        self.max_min = max_min
        self.max_min_wrel2 = MinMaxWRel2()
        self.max_min_xrel2 = MaxMinXRel2()
        self.reduction = reduction
        self.rel_reduction = rel_reduction
        self.x_weight = x_weight
        self.w_weight = w_weight
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):

        base_loss = zenkai.reduce((y - t).pow(2), self.reduction)
        chosen_w = self.max_min_wrel2(x, t)
        inner_w = torch.min(
            x[...,None].detach(), 
            self.max_min.weight[None]
        )
        w_loss = zenkai.reduce(
            ((inner_w - t.unsqueeze(-2)).pow(2) * chosen_w),
            self.rel_reduction
        ) * self.w_weight
        inner_x = torch.min(
            x[...,None], self.max_min.weight[None].detach()
        )
        chosen_x = self.max_min_xrel2(self.max_min.weight, t)
        x_loss = zenkai.reduce(
            ((inner_x - t.unsqueeze(-2)).pow(2) * chosen_x),
            self.rel_reduction
        ) * self.x_weight
        # print(base_loss.shape, w_loss.shape, x_loss.shape)
        return (
            base_loss + x_loss + w_loss
        )


class RandNoise(nn.Module):

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        noise = torch.rand_like(x)
        keep = torch.rand_like(x) >= self.p
        return keep * x + ~keep * noise


class TruncatedGaussianNoise(nn.Module):

    def __init__(self, noise: float):
        super().__init__()
        self.noise = noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.clamp(
            x + torch.randn_like(x) * self.noise, 0, 1
        )


def train(n_epochs: int, x_weight: float=1.0, w_weight: float=1.0):

    torch.manual_seed(3)
    batch_size = 10000
    x = torch.rand(batch_size, 8)

    min_max_base = MinMax(
        8, 16
    )
    min_max = MinMax(
        8, 16
    )
    min_max_loss = MinMaxLoss(min_max, 'mean',  x_weight, w_weight)
    with torch.no_grad():
        t = min_max_base(x)
    
    dataset = torch.utils.data.TensorDataset(
        x, t
    )
    optim = torch.optim.Adam(min_max.parameters(), lr=1e-3)

    for i in range(n_epochs):
        results = []
        for x_i, t_i in torch.utils.data.DataLoader(
            dataset, 128, True
        ):
            y_i = min_max(x_i)
            loss = min_max_loss(x_i, y_i, t_i)
            optim.zero_grad()
            loss.backward()
            optim.step()
            results.append((y_i - t_i).pow(2).mean().item())
        print(np.mean(results))


class MinMaxLearner(zenkai.LearningMachine):

    def __init__(
        self, in_features: int, out_features: int, 
        x_weight: float=1.0, w_weight: float=1.0,
        reduction: str='mean', rel_reduction: str='mean',
        a: float=4
    ):
        super().__init__()
        self.min_max = MinMax(in_features, out_features)
        self.min_max_loss = MinMaxLoss(
            self.min_max, reduction, rel_reduction, x_weight, w_weight
        )
        self.k = 32
        self.select = 4
        self.topk = zenkai.tansaku.TopKSelector(
            self.select, 0
        )

    @property
    def x_weight(self):
        return self.min_max_loss.x_weight
    
    @x_weight.setter
    def x_weight(self, x_weight: float):
        self.min_max_loss.x_weight = x_weight
        return x_weight

    @property
    def w_weight(self):
        return self.min_max_loss.w_weight
    
    @w_weight.setter
    def w_weight(self, w_weight: float):
        self.min_max_loss.w_weight = w_weight
        return w_weight

    @property
    def a(self):
        return self.min_max.a
    
    @a.setter
    def a(self, a: float):
        self.min_max.a = a
        return a

    def accumulate(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs):
        t = torch.clamp(t.f, 0, 1)
        loss = self.min_max_loss(x.f, state._y.f, t) * 0.5
        loss.backward()

    def step_x(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs) -> zenkai.IO:

        x_pop = zenkai.tansaku.add_noise(
            x.f, 32, lambda x, t_info: zenkai.tansaku.gausian_noise(
                x, 0.025
            ).clamp(0, 1)
        )
        x_pop = zenkai.tansaku.collapse_batch(x_pop)
        y_pop = self.min_max(x_pop)
        y_pop = zenkai.tansaku.separate_batch(
            y_pop, 32
        )
        batch_loss = (y_pop - t.f[None]).pow(2).reshape(
            32, -1
        ).mean(1)
        selection = self.topk(batch_loss)
        selected = selection(x_pop)
        # print(batch_loss)
        # print((state._y.f - t.f).pow(2).mean())
        return zenkai.iou(selected.mean(dim=0))
        # x_prime = x.acc_grad()
        # return zenkai.iou(
        #     torch.clamp(x_prime.f + torch.randn_like(x_prime.f) * 0.005, 0, 1)
        # )

    def forward_nn(self, x: zenkai.IO, state: zenkai.State, **kwargs) -> typing.Union[Tuple, typing.Any]:
        
        return self.min_max(x.f)


class MaxMinLearner(zenkai.LearningMachine):

    def __init__(
        self, in_features: int, out_features: int, 
        x_weight: float=1.0, w_weight: float=1.0,
        reduction: str='mean', rel_reduction: str='mean',
        a: float=4
    ):
        super().__init__()
        self.max_min = MaxMin(in_features, out_features)
        self.max_min_loss = MaxMinLoss(
            self.max_min, reduction, rel_reduction, x_weight, w_weight
        )
        self.k = 32
        self.select = 4
        self.topk = zenkai.tansaku.TopKSelector(
            self.select, 0
        )

    @property
    def x_weight(self):
        return self.max_min_loss.x_weight
    
    @x_weight.setter
    def x_weight(self, x_weight: float):
        self.max_min_loss.x_weight = x_weight
        return x_weight

    @property
    def w_weight(self):
        return self.max_min_loss.w_weight
    
    @w_weight.setter
    def w_weight(self, w_weight: float):
        self.max_min_loss.w_weight = w_weight
        return w_weight

    @property
    def a(self):
        return self.max_min.a
    
    @a.setter
    def a(self, a: float):
        self.max_min.a = a
        return a

    def accumulate(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs):
        
        t = torch.clamp(t.f, 0, 1)
        loss = self.max_min_loss(x.f, state._y.f, t) * 0.5
        loss.backward()

    def step_x(self, x: zenkai.IO, t: zenkai.IO, state: zenkai.State, **kwargs) -> zenkai.IO:

        x_pop = zenkai.tansaku.add_noise(
            x.f, 32, lambda x, t_info: zenkai.tansaku.gausian_noise(
                x, 0.025
            ).clamp(0, 1)
        )
        x_pop = zenkai.tansaku.collapse_batch(x_pop)
        y_pop = self.max_min(x_pop)
        y_pop = zenkai.tansaku.separate_batch(
            y_pop, 32
        )
        batch_loss = (y_pop - t.f[None]).pow(2).reshape(
            32, -1
        ).mean(1)
        selection = self.topk(batch_loss)
        # print(batch_loss)
        # print((state._y.f - t.f).pow(2).mean())
        selected = selection.forward(x_pop)
        return zenkai.iou(selected.mean(dim=0))

    def forward_nn(self, x: zenkai.IO, state: zenkai.State, **kwargs) -> typing.Union[Tuple, typing.Any]:
        return self.max_min(x.f)
