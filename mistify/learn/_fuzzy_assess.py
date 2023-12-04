from abc import abstractmethod
import torch
import typing
from torch.nn import functional as nn_func
from torch import nn

from ._core import MistifyLoss, ToOptim
from ..infer._neurons import Or, And
from ..infer._ops import IntersectionOn, UnionOn
from zenkai import IO, Reduction


class FuzzyLoss(MistifyLoss):

    def calc_loss(self, x: torch.Tensor, t: torch.Tensor, mask: torch.BoolTensor=None, weight: torch.Tensor=None, reduction_override: str=None):
        
        result = (x - t) ** 2
        if mask is not None:
            result = result * mask.float()
        if weight is not None:
            result = result * weight
        return 0.5 * self.reduce(result, reduction_override)


class FuzzyAggregatorLoss(FuzzyLoss):
    
    @abstractmethod
    def forward(self, x: IO, y: IO, t: IO, reduction_override: float=None) -> torch.Tensor:
        pass


class IntersectionOnLoss(FuzzyAggregatorLoss):

    def __init__(self, intersect: IntersectionOn, reduction: str='mean', not_chosen_weight: float=1.0):
        super().__init__(intersect, reduction)
        self.intersect = intersect
        self.not_chosen_weight = not_chosen_weight
        self._mse = nn.MSELoss(reduction='none')

    def forward(self, x: IO, y: IO, t: IO, reduction_override: float=None) -> torch.Tensor:

        x = x.f
        y = y.f
        t = t.f
        if not self.intersect.keepdim:
            y = y.unsqueeze(self.intersect.dim)
            t = t.unsqueeze(self.intersect.dim)

        with torch.no_grad():
            min_difference = torch.abs(x - t).min(dim=self.intersect.dim, keepdim=True)[0]
            x_less_than = (x < t)
            x_not_less_than = ~x_less_than
            x_t = x - min_difference
            chosen = x == y

        return (
            self.calc_loss(x, t.detach(), x_less_than, reduction_override=reduction_override)
            + self.calc_loss(x, x_t.detach(), chosen & x_not_less_than, reduction_override=reduction_override)
            + self.calc_loss(x, x_t.detach(), ~chosen & x_not_less_than, self.not_chosen_weight, reduction_override=reduction_override)
        )

    @classmethod
    def factory(cls, reduction: str, not_chosen_weight: float=0.1) -> typing.Callable[[nn.Module], 'IntersectionOnLoss']:
        def _(intersect_on: IntersectionOn):
            return IntersectionOnLoss(
                intersect_on, reduction, not_chosen_weight
            )
        return _


class UnionOnLoss(FuzzyAggregatorLoss):

    def __init__(self, union: UnionOn, reduction: str='mean', not_chosen_weight: float=1.0):
        super().__init__(union, reduction)
        self.union = union
        self.not_chosen_weight = not_chosen_weight
        self._mse = nn.MSELoss(reduction='none')

    def forward(self, x: IO, y: IO, t: IO, reduction_override: float=None) -> torch.Tensor:
        x = x.f
        y = y.f
        t = t.f
        if not self.union.keepdim:
            y = y.unsqueeze(self.union.dim)
            t = t.unsqueeze(self.union.dim)

        with torch.no_grad():
            min_difference = torch.abs(x - t).min(dim=self.union.dim, keepdim=True)[0]
            x_greater_than = (x > t)
            x_not_greater_than = ~x_greater_than
            x_t = x + min_difference
            chosen = x == y

        return (
            self.calc_loss(x, t.detach(), x_greater_than, reduction_override=reduction_override)
            + self.calc_loss(x, x_t.detach(), chosen & x_not_greater_than, reduction_override=reduction_override)
            + self.calc_loss(x, x_t.detach(), ~chosen & x_not_greater_than, self.not_chosen_weight, reduction_override=reduction_override)
        )

    @classmethod
    def factory(cls, reduction: str, not_chosen_weight: float=0.1) -> typing.Callable[[nn.Module], 'IntersectionOnLoss']:
        def _(union_on: UnionOn):
            return UnionOnLoss(
                union_on, reduction, not_chosen_weight
            )
        return _


class MaxMinLoss(FuzzyLoss):

    def __init__(self, module: Or, reduction: str = 'mean'):
        super().__init__(reduction)
        self.module = module

    def forward_w(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        w = self.module.weight[None]
        x = x.unsqueeze(-1)
        t = t.unsqueeze(-2)
        y = y.unsqueeze(-2)

        # t - y  y < t   t
        # 
        less_than_t = torch.max(((t - y) + w), 0)[0].detach() # y < t

        less_than_error_multiplier = ((w < x) & (w < less_than_t)).type_as(x)
        greater_than_multiplier = (w > t).type_as(x)
        less_than_error = (0.5 * (w - less_than_t) ** 2) * less_than_error_multiplier
        greater_than_error = (0.5 * (w - t) ** 2) * greater_than_multiplier
        return less_than_error + greater_than_error

    def forward_x(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        w = self.module.weight[None]
        x = x.unsqueeze(-1)
        t = t.unsqueeze(-2)
        y = y.unsqueeze(-2)

        # t - y  y < t   t
        # 
        less_than_t = torch.max(((t - w) + x), 0)[0].detach() # y < t

        less_than_error_multiplier = ((x < w) & (x < less_than_t)).type_as(x)
        greater_than_multiplier = (x > t).type_as(x)
        less_than_error = (0.5 * (x - less_than_t) ** 2) * less_than_error_multiplier
        greater_than_error = (0.5 * (x - t) ** 2) * greater_than_multiplier
        return less_than_error + greater_than_error

    def forward(self, x: IO, y: IO, t: IO, reduction_override: float=None) -> torch.Tensor:
        x = x.f
        y = y.f
        t = torch.clamp(t.f, 0, 1)

        loss = self.forward_w(x, y, t) + self.forward_x(x, y, t)
        # TODO: add in redution
        return Reduction[reduction_override](loss)


class MaxMinLoss3(FuzzyLoss):

    def __init__(
        self, maxmin: Or, reduction='batchmean', 
        not_chosen_x_weight: float=1.0, not_chosen_theta_weight: float=1.0, 
        default_optim: ToOptim=ToOptim.BOTH
    ):
        super().__init__(reduction)
        self._maxmin = maxmin
        self._default_optim = default_optim
        self.not_chosen_theta_weight = not_chosen_theta_weight
        self.not_chosen_x_weight = not_chosen_x_weight
    
    def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
        return torch.min(x, w)

    def set_chosen(self, inner_values: torch.Tensor):

        val, idx = torch.max(inner_values, dim=-2, keepdim=True)
        return inner_values == val
        # chosen = torch.zeros(inner_values.size(), dtype=torch.bool, device=inner_values.device)
        # chosen.scatter_(1, idx,  1.0)
        # return chosen
    
    def forward(self, x: IO, y: IO, t: IO, reduction_override: float=None) -> torch.Tensor:
        
        x = x.f
        y = y.f

        t = torch.clamp(t.f, 0, 1)
        x = x.unsqueeze(x.dim())
        y = y.unsqueeze(x.dim() - 2)
        t = t.unsqueeze(x.dim() - 2)
        w = self._maxmin.weight[None]
        inner_values = self.calc_inner_values(x, w)
        chosen = self.set_chosen(inner_values)
        with torch.no_grad():
            # print(t.size(), inner_values.size())
            d_less = nn_func.relu(t - inner_values).min(dim=-2, keepdim=True)[0]
            d_greater = nn_func.relu(inner_values - t)

        loss = None
        if self._default_optim.theta():
            with torch.no_grad():
                # greater_than = w > t
                w_target = w + torch.sign(t - w) * d_less
                w_target_2 = w - d_greater

            # print('W loss:')
            loss = (
                self.calc_loss(w, w_target_2.detach(), reduction_override=reduction_override) +
                self.calc_loss(w, w_target.detach(), chosen, reduction_override=reduction_override) +
                self.calc_loss(w, w_target.detach(), ~chosen, self.not_chosen_theta_weight, reduction_override=reduction_override)
            )
        if self._default_optim.x():
            with torch.no_grad():
                # greater_than = x > t
                x_target = x + torch.sign(t - x) * d_less
                x_target_2 = x - d_greater

            cur_loss = (
                self.calc_loss(x, x_target_2.detach(), reduction_override=reduction_override) +
                self.calc_loss(x, x_target.detach(), chosen, reduction_override=reduction_override) +
                self.calc_loss(x, x_target.detach(), ~chosen, self.not_chosen_x_weight, reduction_override=reduction_override) 
            )
            loss = cur_loss if loss is None else loss + cur_loss

        return loss

    @classmethod
    def factory(
        cls, reduction: str, not_chosen_x_weight: float=0.1,
        not_chosen_theta_weight: float=0.1, default_optim: ToOptim=ToOptim.BOTH
    ) -> typing.Callable[[Or], 'MaxMinLoss3']:
        def _(maxmin: Or):
            return MaxMinLoss3(
                maxmin, reduction, not_chosen_x_weight,
                not_chosen_theta_weight, default_optim
            )
        return _


class MinMaxLoss3(FuzzyLoss):

    def __init__(
        self, minmax: And, reduction='batchmean', not_chosen_x_weight: float=1.0, 
        not_chosen_theta_weight: float=1.0, 
        default_optim: ToOptim=ToOptim.BOTH
    ):
        super().__init__(reduction)
        self._minmax = minmax
        self._default_optim = default_optim
        self.not_chosen_theta_weight = not_chosen_theta_weight
        self.not_chosen_x_weight = not_chosen_x_weight
    
    def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
        return torch.max(x, w)

    def set_chosen(self, inner_values: torch.Tensor):

        val, idx = torch.max(inner_values, dim=-2, keepdim=True)
        return inner_values == val
    
    def forward(self, x: IO, y: IO, t: IO, reduction_override: float=None) -> torch.Tensor:
        
        x = x.f
        y = y.f
        t = torch.clamp(t.f, 0, 1)
        x = x.unsqueeze(x.dim())
        y = y.unsqueeze(x.dim() - 2)
        t = t.unsqueeze(x.dim() - 2)
        w = self._minmax.weight[None]
        inner_values = self.calc_inner_values(x, w)
        chosen = self.set_chosen(inner_values)
        with torch.no_grad():
            d_greater = nn_func.relu(inner_values - t).min(dim=-2, keepdim=True)[0]
            d_inner = nn_func.relu(t - inner_values)

        loss = None
        if self._default_optim.theta():
            with torch.no_grad():
                w_target = w - torch.sign(w - t) * d_greater
                w_target_2 = w + d_inner
            
            loss = (
                self.calc_loss(w, w_target_2.detach(), reduction_override=reduction_override) +
                self.calc_loss(w, w_target.detach(), chosen, reduction_override=reduction_override) +
                self.calc_loss(w, w_target.detach(), ~chosen, self.not_chosen_theta_weight, reduction_override=reduction_override)
            )
        if self._default_optim.x():
            with torch.no_grad():
                x_target = x - torch.sign(x - t) * d_greater
                x_target_2 = x + d_inner

            cur_loss = (
                self.calc_loss(x, x_target_2.detach(), reduction_override=reduction_override) +
                self.calc_loss(x, x_target.detach(), chosen, reduction_override=reduction_override) +
                self.calc_loss(x, x_target.detach(), ~chosen, self.not_chosen_x_weight, reduction_override=reduction_override) 
            )
            loss = cur_loss if loss is None else loss + cur_loss

        return loss

    @classmethod
    def factory(
        cls, reduction: str, not_chosen_x_weight: float=0.1,
        not_chosen_theta_weight: float=0.1, default_optim: ToOptim=ToOptim.BOTH
    ) -> typing.Callable[[Or], 'MinMaxLoss3']:
        def _(minmax: Or):
            return MinMaxLoss3(
                minmax, reduction, not_chosen_x_weight,
                not_chosen_theta_weight, default_optim
            )
        return _


class MaxMinLoss2(FuzzyLoss):

    def __init__(
        self, maxmin: Or, reduction='batchmean', not_chosen_x_weight: float=1.0, not_chosen_theta_weight: float=1.0, 
        default_optim: ToOptim=ToOptim.BOTH
    ):
        super().__init__(reduction)
        self._maxmin = maxmin
        self._default_optim = default_optim
        self.not_chosen_theta_weight = not_chosen_theta_weight
        self.not_chosen_x_weight = not_chosen_x_weight
    
    def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
        return torch.min(x, w)

    def set_chosen(self, inner_values: torch.Tensor):

        val, idx = torch.max(inner_values, dim=-2, keepdim=True)
        return inner_values == val
        # chosen = torch.zeros(inner_values.size(), dtype=torch.bool, device=inner_values.device)
        # chosen.scatter_(1, idx,  1.0)
        # return chosen
    
    def forward(self, x: IO, y: IO, t: IO, reduction_override: float=None) -> torch.Tensor:
        
        x = x.f
        y = y.f
        t = torch.clamp(t.f, 0, 1)
        x = x.unsqueeze(x.dim())
        y = y.unsqueeze(x.dim() - 2)
        t = t.unsqueeze(x.dim() - 2)
        w = self._maxmin.weight[None]
        inner_values = self.calc_inner_values(x, w)
        chosen = self.set_chosen(inner_values)
        with torch.no_grad():
            dy = nn_func.relu(t - y)
            d_inner = nn_func.relu(inner_values - t)

        # inner > t.. okay
        # inner < t...
        #   w > t.... max(w - dy, t)
        #   w < t.... min(w + dy, t)
        #   w < t
        
        # x is the same... 
        # (w > t).float() * max(w - dy, t) + (w < t).float() * min(w + dy, t)
        # torch.relu(w - dy)  
        # sign(w - t) * max()
        # Still need not chosen weigt

        loss = None
        if self._default_optim.theta():
            with torch.no_grad():

                # value will not exceed the x
                greater_than = w > t

                w_target = (
                    greater_than.float() * torch.max(w - dy, t)
                    + (~greater_than).float() * torch.min(w + dy, t)
                )
                w_target_2 = w - d_inner

            # print('W loss:')
            loss = (
                self.calc_loss(w, w_target_2.detach(), reduction_override=reduction_override) +
                # self.calc_loss(w, t.detaåch(), inner_values > t) +
                self.calc_loss(w, w_target.detach(), chosen, reduction_override=reduction_override) +
                self.calc_loss(w, w_target.detach(), ~chosen, self.not_chosen_theta_weight, reduction_override=reduction_override)
            )
        if self._default_optim.x():
            with torch.no_grad():
                # x_target = torch.max(torch.min(x + dy, w), x)
                # this is wrong.. y can end up targetting a value greater than
                # one because of this...
                # x=0.95, w=0.1 -> y=0.75 t=0.8 ... x will end up targetting 1.0
                # this is also a conundrum because i may want to reduce the value of
                # x.. But the value is so high it does not get reduced

                # value will not exceed the target if smaller than target
                # if larger than target will not change
                greater_than = x > t
                x_target = (
                    greater_than.float() * torch.max(x - dy, t)
                    + (~greater_than).float() * torch.min(x + dy, t)
                )
                # print('Target: ', x[0, 0]) 
                # print(t[0]) 
                # print(y[0]) 
                # print(x_target[0, 0])
                x_target_2 = x - d_inner

            # print('X loss:')
            cur_loss = (
                self.calc_loss(x, x_target_2.detach(), reduction_override=reduction_override) +
                self.calc_loss(x, x_target.detach(), chosen, reduction_override=reduction_override) +
                self.calc_loss(x, x_target.detach(), ~chosen, self.not_chosen_x_weight, reduction_override=reduction_override) 
            )
            # print(cur_loss.sum() / len(cur_loss))
            loss = cur_loss if loss is None else loss + cur_loss
            # print(loss.sum() / len(loss))

        return loss

    @classmethod
    def factory(
        cls, reduction: str, not_chosen_x_weight: float=0.1,
        not_chosen_theta_weight: float=0.1, default_optim: ToOptim=ToOptim.BOTH
    ) -> typing.Callable[[Or], 'MaxMinLoss2']:
        def _(maxmin: Or):
            return MaxMinLoss2(
                maxmin, reduction, not_chosen_x_weight,
                not_chosen_theta_weight, default_optim
            )
        return _


class MinMaxLoss2(FuzzyLoss):

    def __init__(
        self, minmax: And, reduction='batchmean', not_chosen_x_weight: float=1.0, not_chosen_theta_weight: float=1.0, 
        default_optim: ToOptim=ToOptim.BOTH
    ):
        super().__init__(reduction)
        self._minmax = minmax
        self._default_optim = default_optim
        self.not_chosen_theta_weight = not_chosen_theta_weight
        self.not_chosen_x_weight = not_chosen_x_weight
    
    def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
        return torch.max(x, w)

    def set_chosen(self, inner_values: torch.Tensor):

        val, idx = torch.max(inner_values, dim=-2, keepdim=True)
        return inner_values == val
        # chosen = torch.zeros(inner_values.size(), dtype=torch.bool, device=inner_values.device)
        # chosen.scatter_(1, idx,  1.0)
        # return chosen
    
    def forward(self, x: IO, y: IO, t: IO, reduction_override: float=None) -> torch.Tensor:
        
        x = x.f
        y = y.f
        t = torch.clamp(t.f, 0, 1)
        x = x.unsqueeze(x.dim())
        y = y.unsqueeze(x.dim() - 2)
        t = t.unsqueeze(x.dim() - 2)
        w = self._minmax.weight[None]
        inner_values = self.calc_inner_values(x, w)
        chosen = self.set_chosen(inner_values)
        with torch.no_grad():
            dy = nn_func.relu(y - t)
            d_inner = nn_func.relu(t - inner_values)

        loss = None
        if self._default_optim.theta():
            with torch.no_grad():
                less_than = w < t

                w_target = (
                    less_than.float() * torch.min(w + dy, t)
                    + (~less_than).float() * torch.max(w - dy, t)
                )
                w_target_2 = w + d_inner
            
            loss = (
                self.calc_loss(w, w_target_2.detach(), reduction_override=reduction_override) +
                self.calc_loss(w, w_target.detach(), chosen, reduction_override=reduction_override) +
                self.calc_loss(w, w_target.detach(), ~chosen, self.not_chosen_theta_weight, reduction_override=reduction_override)
            )
        if self._default_optim.x():
            with torch.no_grad():
                less_than = x < t
                x_target = (
                    less_than.float() * torch.min(x + dy, t)
                    + (~less_than).float() * torch.max(x - dy, t)
                )
                x_target_2 = x + d_inner

            cur_loss = (
                self.calc_loss(x, x_target_2.detach(), reduction_override=reduction_override) +
                self.calc_loss(x, x_target.detach(), chosen, reduction_override=reduction_override) +
                self.calc_loss(x, x_target.detach(), ~chosen, self.not_chosen_x_weight, reduction_override=reduction_override) 
            )
            loss = cur_loss if loss is None else loss + cur_loss

        return loss

    @classmethod
    def factory(
        cls, reduction: str, not_chosen_x_weight: float=0.1,
        not_chosen_theta_weight: float=0.1, default_optim: ToOptim=ToOptim.BOTH
    ) -> typing.Callable[[And], 'MinMaxLoss2']:
        def _(minmax: And):
            return MinMaxLoss2(
                minmax, reduction, not_chosen_x_weight,
                not_chosen_theta_weight, default_optim
            )
        return _


class MaxProdLoss(FuzzyLoss):

    def __init__(self, maxprod: Or, reduction='batchmean', not_chosen_weight: float=None, default_optim: ToOptim=ToOptim.BOTH):
        super().__init__(maxprod, reduction)
        self._maxprod = maxprod
        self.reduction = reduction
        self._default_optim = default_optim
        self._mse = nn.MSELoss(reduction='none')
        self.not_chosen_weight = not_chosen_weight or 1.0

    def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
        return x * w

    def set_chosen(self, inner_values: torch.Tensor):

        val, _ = torch.max(inner_values, dim=-2, keepdim=True)
        return inner_values == val

        # _, idx = torch.max(inner_values, dim=-2, keepdim=True)
        # chosen = torch.zeros(inner_values.size(), dtype=torch.bool)
        # chosen.scatter_(1, idx,  1.0)
        # return chosen
    
    def clamp(self):

        self._maxprod.weight.data = torch.clamp(self._maxprod.weight.data, 0, 1).detach()
    
    def forward(self, x: IO, y: IO, t: IO, reduction_override: float=None) -> torch.Tensor:
        
        x = x.f
        y = y.f
        t = torch.clamp(t.f, 0, 1)
        x = x.unsqueeze(x.dim())
        y = y.unsqueeze(x.dim() - 2)
        t = t.unsqueeze(x.dim() - 2)
        w = self._maxprod.weight[None]

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
                self.calc_loss(inner_values, t.detach(), inner_values > t, reduction_override=reduction_override) +
                self.calc_loss(inner_values, inner_target.detach(), chosen, reduction_override=reduction_override)  +
                self.calc_loss(inner_values, inner_target.detach(), ~chosen, self.not_chosen_weight, reduction_override=reduction_override) 
            )

        return loss

    @classmethod
    def factory(
        cls, reduction: str, not_chosen_weight: float=0.1,
        default_optim: ToOptim=ToOptim.BOTH
    ) -> typing.Callable[[Or], 'MaxProdLoss']:
        def _(maxmin: Or):
            return MaxProdLoss(
                maxmin, reduction, not_chosen_weight,
                default_optim
            )
        return _




# class MaxMinLoss(FuzzyLoss):

#     def __init__(
#         self, maxmin: MaxMin, reduction='batchmean', not_chosen_x_weight: float=1.0, not_chosen_theta_weight: float=1.0, 
#         default_optim: ToOptim=ToOptim.BOTH
#     ):
#         super().__init__(maxmin, reduction)
#         self._maxmin = maxmin
#         self._default_optim = default_optim
#         self._mse = nn.MSELoss(reduction='none')
#         self.not_chosen_theta_weight = not_chosen_theta_weight
#         self.not_chosen_x_weight = not_chosen_x_weight
    
#     def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
#         return torch.min(x, w)

#     def set_chosen(self, inner_values: torch.Tensor):

#         val, idx = torch.max(inner_values, dim=-2, keepdim=True)
#         return inner_values == val
    
#     def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
#         x = x.unsqueeze(x.dim())
#         y = y.unsqueeze(x.dim() - 2)
#         t = t.unsqueeze(x.dim() - 2)
#         w = self._maxmin.weight[None]
#         inner_values = self.calc_inner_values(x, w)
#         chosen = self.set_chosen(inner_values)
#         with torch.no_grad():
#             dy = nn_func.relu(t - y)
#             d_inner = nn_func.relu(inner_values - t)

#         # inner > t.. okay
#         # inner < t...
#         #   w > t.... max(w - dy, t)
#         #   w < t.... min(w + dy, t)
#         #   w < t
        
#         # x is the same... 
#         # (w > t).float() * max(w - dy, t) + (w < t).float() * min(w + dy, t)
#         # torch.relu(w - dy)  
#         # sign(w - t) * max()
#         # Still need not chosen weigt

#         loss = None
#         if self._default_optim.theta():
#             with torch.no_grad():

#                 # value will not exceed the x
#                 w_target = torch.max(torch.min(w + dy, x), w)
#                 w_target_2 = w - d_inner

#             loss = (
#                 self.calc_loss(w, w_target_2.detach()) +
#                 # self.calc_loss(w, t.detach(), inner_values > t) +
#                 self.calc_loss(w, w_target.detach(), chosen) +
#                 self.calc_loss(w, w_target.detach(), ~chosen, self.not_chosen_theta_weight)
#             )
#         if self._default_optim.x():
#             with torch.no_grad():
#                 # x_target = torch.max(torch.min(x + dy, w), x)
#                 # this is wrong.. y can end up targetting a value greater than
#                 # one because of this...
#                 # x=0.95, w=0.1 -> y=0.75 t=0.8 ... x will end up targetting 1.0
#                 # this is also a conundrum because i may want to reduce the value of
#                 # x.. But the value is so high it does not get reduced

#                 # value will not exceed the target if smaller than target
#                 # if larger than target will not change
#                 x_target = torch.max(x, torch.min(x + dy, t))
#                 w_target_2 = x - d_inner

#             cur_loss = (
#                 self.calc_loss(x, w_target_2.detach()) +
#                 self.calc_loss(x, x_target.detach(), chosen) +
#                 self.calc_loss(x, x_target.detach(), ~chosen, self.not_chosen_x_weight) 
#             )
#             loss = cur_loss if loss is None else loss + cur_loss

#         return loss


class MinMaxLoss(FuzzyLoss):

    def __init__(self, minmax: And, reduction='batchmean', not_chosen_weight: float=None, default_optim: ToOptim=ToOptim.BOTH):
        super().__init__(minmax, reduction)
        self._minmax = minmax
        self._default_optim = default_optim
        self._mse = nn.MSELoss(reduction='none')
        self.not_chosen_weight = not_chosen_weight or 1.0
    
    def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
        return torch.max(x, w)

    def set_chosen(self, inner_values: torch.Tensor):

        val, idx = torch.min(inner_values, dim=-2, keepdim=True)
        return inner_values == val
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        # TODO: Update so that the number of dimensions is more flexible
        x = x.unsqueeze(x.dim())
        y = y.unsqueeze(x.dim() - 2)
        t = t.unsqueeze(x.dim() - 2)
        w = self._minmax.weight[None]
        inner_values = self.calc_inner_values(x, w)
        chosen = self.set_chosen(inner_values)
        with torch.no_grad():
            dy = nn_func.relu(y - t)
            d_inner = nn_func.relu(t - inner_values)

        loss = None
        if self._default_optim.theta():
            with torch.no_grad():
                w_target = torch.min(torch.max(w - dy, x), w)
                w_target_2 = d_inner - w

            loss = (
                self.calc_loss(w, w_target_2.detach()) +
                self.calc_loss(w, w_target.detach(), chosen) +
                self.calc_loss(w, w_target.detach(), ~chosen, self.not_chosen_weight)
            )
        if self._default_optim.x():
            with torch.no_grad():
                x_target = torch.min(x, torch.max(x - dy, t))
                # x_target = torch.min(torch.max(x - dy, w), x)
                w_target_2 = d_inner - x

            cur_loss = (
                self.calc_loss(x, w_target_2.detach()) +
                self.calc_loss(x, x_target.detach(), chosen) +
                self.calc_loss(x, x_target.detach(), ~chosen, self.not_chosen_weight) 
            )
            loss = cur_loss if loss is None else loss + cur_loss

        return loss


# class MaxMinLoss3(FuzzyLoss):

#     def __init__(
#         self, maxmin: MaxMin, reduction='batchmean', not_chosen_x_weight: float=1.0, not_chosen_theta_weight: float=1.0, 
#         default_optim: ToOptim=ToOptim.BOTH
#     ):
#         super().__init__(maxmin, reduction)
#         self._maxmin = maxmin
#         self._default_optim = default_optim
#         self._mse = nn.MSELoss(reduction='none')
#         self.not_chosen_theta_weight = not_chosen_theta_weight
#         self.not_chosen_x_weight = not_chosen_x_weight
    
#     def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
#         return torch.min(x, w)

#     def set_chosen(self, inner_values: torch.Tensor):

#         val, idx = torch.max(inner_values, dim=-2, keepdim=True)
#         return inner_values == val
    
#     def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
#         x = x.unsqueeze(x.dim())
#         y = y[:,None]
#         t = t[:,None]
#         w = self._maxmin.weight[None]
#         inner_values = self.calc_inner_values(x, w)
#         chosen = self.set_chosen(inner_values)
#         with torch.no_grad():
#             dy = (t - y).abs()
#             d_inner = nn_func.relu(inner_values - t)
#             inner_greater = inner_values > t
#             inner_less = ~inner_greater

#         loss = None
#         if self._default_optim.theta():
#             with torch.no_grad():
#                 dw = (t - w)
#                 w_target = w + torch.sign(dw) * torch.min(dw.abs(), dy) * inner_less.float()
#                 w_target_2 = w - d_inner
            
#             loss = (
#                 self.calc_loss(w, w_target_2.detach()) +
#                 self.calc_loss(w, w_target.detach(), chosen) +
#                 self.calc_loss(w, w_target.detach(), ~chosen, self.not_chosen_theta_weight)
#             )
#         if self._default_optim.x():
#             with torch.no_grad():
#                 dx = (t - x)
#                 sign_x = torch.sign(dx)
#                 x_target = x + sign_x * torch.min(dx.sign(), dy.sign()) * inner_less.float()
#                 x_target_2 = x - d_inner

#             cur_loss = (
#                 self.calc_loss(x, x_target_2.detach()) +
#                 self.calc_loss(x, x_target.detach(), chosen) +
#                 self.calc_loss(x, x_target.detach(), ~chosen, self.not_chosen_x_weight) 
#             )
#             loss = cur_loss if loss is None else loss + cur_loss
#         return loss


# class MixMaxLoss3(FuzzyLoss):

#     def __init__(
#         self, minmax: MinMax, reduction='batchmean', not_chosen_x_weight: float=1.0, not_chosen_theta_weight: float=1.0, 
#         default_optim: ToOptim=ToOptim.BOTH
#     ):
#         super().__init__(minmax, reduction)
#         self._minmax = minmax
#         self._default_optim = default_optim
#         self.not_chosen_theta_weight = not_chosen_theta_weight
#         self.not_chosen_x_weight = not_chosen_x_weight
    
#     def calc_inner_values(self, x: torch.Tensor, w: torch.Tensor):
#         return torch.min(x, w)

#     def set_chosen(self, inner_values: torch.Tensor):

#         val, idx = torch.max(inner_values, dim=-2, keepdim=True)
#         return inner_values == val
    
#     def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
#         x = x.unsqueeze(x.dim())
#         y = y[:,None]
#         t = t[:,None]
#         w = self._minmax.weight[None]
#         inner_values = self.calc_inner_values(x, w)
#         chosen = self.set_chosen(inner_values)
#         with torch.no_grad():
#             dy = nn_func.relu(y - t)
#             d_inner = nn_func.relu(t - inner_values)

#         loss = None
#         if self._default_optim.theta():
#             with torch.no_grad():

#                 less_than = w < t

#                 w_target = (
#                     less_than.float() * torch.min(w + dy, t)
#                     + (~less_than).float() * torch.max(w - dy, t)
#                 )
#                 w_target_2 = w + d_inner

#             loss = (
#                 self.calc_loss(w, w_target_2.detach()) +
#                 self.calc_loss(w, w_target.detach(), chosen) +
#                 self.calc_loss(w, w_target.detach(), ~chosen, self.not_chosen_theta_weight)
#             )
#         if self._default_optim.x():
#             with torch.no_grad():
#                 less_than = x < t
#                 x_target = (
#                     less_than.float() * torch.min(x + dy, t)
#                     + (~less_than).float() * torch.max(x - dy, t)
#                 )
#                 x_target_2 = x - d_inner

#             cur_loss = (
#                 self.calc_loss(x, x_target_2.detach()) +
#                 self.calc_loss(x, x_target.detach(), chosen) +
#                 self.calc_loss(x, x_target.detach(), ~chosen, self.not_chosen_x_weight) 
#             )
#             loss = cur_loss if loss is None else loss + cur_loss

#         return loss



# 1st Party
from abc import abstractmethod
from enum import Enum

# 3rd Party
#import torch
#import torch.nn as nn


# Local
# from ..utils import reduce
#from .. import conversion
# from .. import fuzzy, binary
# import torch.nn.functional as nn_func
#import typing
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
        
