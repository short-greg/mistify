from zenkai.kaku._io2 import IO as IO
from ..infer import Or, And
import torch.nn as nn
import torch
import typing
from functools import partial
from abc import abstractmethod, ABC
from ..infer import LogicalNeuron
# from zenkai.kikai import WrapNN, WrapState
from typing_extensions import Self
from zenkai import XCriterion, Criterion


class Rel(nn.Module, ABC):
    """Function to use for predicting the relation between two values
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, dim: int) -> torch.Tensor:
        pass

    @classmethod
    def x_rel(cls, *args, **kwargs) -> 'XRel':
        return XRel(
            cls(*args, **kwargs)
        )

    @classmethod
    def w_rel(cls, *args, **kwargs) -> 'WRel':
        return WRel(
            cls(*args, **kwargs)
        )


class XRel(nn.Module):

    def __init__(self, base_rel: Rel) -> None:
        """Predict the x from W and T
        """
        super().__init__()
        self.base_rel = base_rel

    def forward(self, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The weight
            t (torch.Tensor): The target

        Returns:
            torch.Tensor: The "x"
        """
        # add in the unsqueeze
        w = w.unsqueeze(0)
        t = t.unsqueeze(-2)
        return self.base_rel(w, t, dim=-1).squeeze(dim=-1)


class WRel(nn.Module):
    """Predict the weight from X and T
    """

    def __init__(self, base_rel: Rel) -> None:
        super().__init__()
        self.base_rel = base_rel

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input
            t (torch.Tensor): The target

        Returns:
            torch.Tensor: The "weight"
        """
        x = x.unsqueeze(-1)
        t = t.unsqueeze(-2)
        return self.base_rel(x, t, dim=0).squeeze(dim=0)


class MaxMinRel(Rel):
    """The relation for a "MaxMin" function
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor, dim: int=0) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input
            t (torch.Tensor): The target
            dim (int, optional): The dimension to calculate the rel with. Defaults to 0.

        Returns:
            torch.Tensor: The relation
        """
        return torch.min(
            x, t
        ).sum(dim=dim, keepdim=True) / (torch.sum(x, dim=dim, keepdim=True) + 1e-7)


class MinMaxRel(Rel):
    """The relation for a "MinMax" function
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor, dim: int=0) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input
            t (torch.Tensor): The target
            dim (int, optional): The dimension to calculate the rel with. Defaults to 0.

        Returns:
            torch.Tensor: The relation
        """
        x_comp = 1 - x
        t_comp = 1 - t

        return 1 - torch.min(
            x_comp, t_comp
        ).sum(dim=dim, keepdim=True) / (
            torch.sum(x_comp, dim=dim, keepdim=True) + 1e-7
        )


class MaxProdRel(Rel):
    """The relation for a "MaxProd" function
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor, dim: int=0) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input
            t (torch.Tensor): The target
            dim (int, optional): The dimension to calculate the rel with. Defaults to 0.

        Returns:
            torch.Tensor: The relation
        """
        comp = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        return torch.min(
            t / (x + 1e-7), comp
        ).sum(dim=dim, keepdim=True) / torch.sum(x, dim=dim, keepdim=True)


class MinSumRel(Rel):
    """The relation for a "MinSum" function
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor, dim: int=0) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input
            t (torch.Tensor): The target
            dim (int, optional): The dimension to calculate the rel with. Defaults to 0.

        Returns:
            torch.Tensor: The relation
        """
        x_comp = 1 - x
        t_comp = 1 - t
        comp = torch.tensor(1.0, dtype=x_comp.dtype, device=x_comp.device)

        return 1 - torch.min(
            (t_comp - x_comp) / (1 - x_comp), comp
        ).sum(dim=dim, keepdim=True) / torch.sum(x_comp, dim=dim, keepdim=True)


def align_sort(x1: torch.Tensor, x2: torch.Tensor, dim: int, descending: bool=False) -> torch.Tensor:

    x1_sort, _ = x1.sort(dim)
    _, x2_ind = x2.sort(dim)

    x_t = torch.empty_like(x1)
    x_t.scatter_(dim, x2_ind, x1_sort)
    return x_t

    # return x1_sort.gather(dim, x2_ind)

    # x1_sort, x1_ind = x1.sort(dim, descending)
    # x2, x2_ind = x2.sort(dim, descending)

    # a = torch.arange(x2.size(dim))
    # r = []
    # if dim < 0:
    #     dim = x2.dim() + dim
    # for i, s in enumerate(x2.shape):
    #     if i == dim:
    #         print('I = dim')
    #         r.append(1)
    #     else:
    #         r.append(s)
    #         a = a.unsqueeze(i)
    # a = a.repeat(r)


    inverted_index = torch.empty_like(x2_ind)
    inverted_index.scatter_(0, x2_ind, a)
    return x1_sort.detach().gather(dim, inverted_index)


class AlignLoss(XCriterion):

    def __init__(
        self, base_loss: nn.Module, neuron: LogicalNeuron, 
        x_rel: Rel=None, w_rel: Rel=None, 
        x_weight: float=1.0, w_weight: float=1.0
    ):
        """Use the outputs using w_rel and x_rel as regularizers for the loss. 
        Based on which are more related to the output

        Args:
            base_loss (nn.Module): The loss function to use. Must use a reduction that reduces to a scalar
            neuron (nn.Module): The fuzzy neuron. Assumes the weight is the member variable w
            x_rel (Rel, optional): The x relation to calculate the loss with. Defaults to None.
            w_rel (Rel, optional): The w relation to calculate the loss with. Defaults to None.
            x_weight (float, optional): The weight on x_rel. Defaults to 1.0.
            w_weight (float, optional): The weight on w_rel. Defaults to 1.0.
        """
        super().__init__()
        self.base_loss = base_loss
        self.neuron = neuron
        self.x_rel = XRel(x_rel) if x_rel is not None else None
        self.w_rel = WRel(w_rel) if w_rel is not None else None
        self.x_weight = x_weight
        self.w_weight = w_weight

    def forward(self, x: IO, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        """

        Args:
            x (IO): The input to the model
            y (IO): The output of the model
            t (IO): The target

        Returns:
            torch.Tensor: 
        """
        # if reduction_override is not None:
        #     raise ValueError('Cannnot override the reduction for AlignLoss')
        
        x, y, t = x.f, y.f, t.f
        t = t.clamp(0, 1)
        base_loss = self.base_loss(y, t)
        # print('----')
        # print(base_loss.item())
        if self.x_rel is not None:
            x_rel = self.x_rel(self.neuron.w(), t)
            x_t = align_sort(x.detach(), x_rel.detach(), dim=-1)
            x_loss = (x - x_t).pow(2).sum()
            # print(x_loss.item())
            base_loss = base_loss + self.x_weight * x_loss
        if self.w_rel is not None:
            w = self.neuron.w()
            w_rel = self.w_rel(x, t)
            w_t = align_sort(w.detach(), w_rel.detach(), dim=-1)
            w_loss = (w - w_t).pow(2).sum()
            # print(w_loss.item())
            base_loss = base_loss + self.w_weight * w_loss
        return base_loss


class RelLoss(XCriterion):

    def __init__(self, base_loss: nn.Module, neuron: LogicalNeuron, x_rel: Rel=None, w_rel: Rel=None, x_weight: float=1.0, w_weight: float=1.0):
        """Use the outputs using w_rel and x_rel as regularizers for the loss

        Args:
            base_loss (nn.Module): The loss function to use. 
            neuron (nn.Module): The fuzzy neuron. Assumes the weight is the member variable w
            x_rel (Rel, optional): The x relation to calculate the loss with. Defaults to None.
            w_rel (Rel, optional): The w relation to calculate the loss with. Defaults to None.
            x_weight (float, optional): The weight on x_rel. Defaults to 1.0.
            w_weight (float, optional): The weight on w_rel. Defaults to 1.0.
        """
        super().__init__()
        self.base_loss = base_loss
        self.neuron = neuron
        self.x_rel = XRel(x_rel) if x_rel is not None else None
        self.w_rel = WRel(w_rel) if w_rel is not None else None
        self.x_weight = x_weight
        self.w_weight = w_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): The input to the model
            y (torch.Tensor): The output of the model
            t (torch.Tensor): The target

        Returns:
            torch.Tensor: _description_
        """
        x, y, t = x.f, y.f, t.f
        base_loss = self.base_loss(y, t)
        if self.x_rel is not None:
            x_rel = self.x_rel(self.neuron.w(), t)
            yx = self.neuron.f(x_rel.detach(), self.neuron.w())
            base_loss = base_loss + self.x_weight * self.base_loss(yx, t)
        if self.w_rel is not None:
            w_rel = self.w_rel(x, t)
            yw = self.neuron.f(x, w_rel.detach())
            base_loss = base_loss + self.w_weight * self.base_loss(yw, t)
        return base_loss


# class MaxMinRelOut(nn.Module):

#     def forward(self, x: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

#         x = x.unsqueeze(-1)
#         t = t.unsqueeze(-2)
#         w = w.unsqueeze(0)

#         with torch.no_grad():
#             rel_x = torch.min(
#                 x, t
#             ).sum(dim=0, keepdim=True) / torch.sum(x, dim=0, keepdim=True)

#             rel_w = torch.min(
#                 w, t
#             ).sum(dim=-2, keepdim=True) / torch.sum(w, dim=-2, keepdim=True)

#             # for max prod use this for the inner
#             # t = 0.2, x=0.4 => min(t / x, 1.0)

#             ind_w = torch.max(torch.min(x, rel_x), keepdim=True, dim=-2)[1]
#             ind_x = torch.max(torch.min(rel_w, w), keepdim=True, dim=-2)[1]
#         inner_x = torch.min(x, w.detach())
#         inner_w = torch.min(x.detach(), w)

#         chosen_x = inner_x.gather(-2, ind_x)
#         chosen_w = inner_w.gather(-2, ind_w)
#         return chosen_x.squeeze(-2), chosen_w.squeeze(-2)


# class MaxProdRelOut(nn.Module):

#     def forward(self, x: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

#         x = x.unsqueeze(-1)
#         t = t.unsqueeze(-2)
#         w = w.unsqueeze(0)
#         comp = torch.tensor(1.0, dtype=x.dtype, device=x.device)

#         rel_x = torch.min(
#             t / (x + 1e-7),  comp
#         ).sum(dim=0, keepdim=True) / torch.sum(x, dim=0, keepdim=True)

#         rel_w = torch.min(
#             t / (w + 1e-7), comp
#         ).sum(dim=-2, keepdim=True) / torch.sum(x, dim=-2, keepdim=True)

#         # for max prod use this for the inner
#         # t = 0.2, x=0.4 => min(t / x, 1.0)

#         ind_x = torch.max(x * rel_x, keepdim=True, dim=-2)[1]
#         ind_w = torch.max(rel_w * w, keepdim=True, dim=-2)[1]
#         inner_x = x * w.detach()
#         inner_w = x.detach() * w

#         chosen_x = inner_x.gather(-2, ind_x)
#         chosen_w = inner_w.gather(-2, ind_w)
#         return chosen_x.squeeze(-2), chosen_w.squeeze(-2)


# class MinMaxRelOut(nn.Module):

#     def forward(self, x: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

#         x = x.unsqueeze(-1)
#         t = t.unsqueeze(-2)
#         w = w.unsqueeze(0)

#         x_comp = 1 - x
#         t_comp = 1 - t
#         w_comp = 1 - w

#         rel_x = 1 - torch.min(
#             x_comp, t_comp
#         ).sum(dim=0, keepdim=True) / torch.sum(x_comp, dim=0, keepdim=True)

#         rel_w = 1 - torch.min(
#             w_comp, t_comp
#         ).sum(dim=-2, keepdim=True) / torch.sum(w_comp, dim=-2, keepdim=True)

#         # for max prod use this for the inner
#         # t = 0.2, x=0.4 => min(t / x, 1.0)

#         ind_x = torch.min(torch.max(x, rel_x), keepdim=True, dim=-2)[1]
#         ind_w = torch.min(torch.max(rel_w, w), keepdim=True, dim=-2)[1]
#         inner_x = torch.max(x, w.detach())
#         inner_w = torch.max(x.detach(), w)

#         chosen_x = inner_x.gather(-2, ind_x)
#         chosen_w = inner_w.gather(-2, ind_w)
#         return chosen_x.squeeze(-2), chosen_w.squeeze(-2)


# class MinSumRelOut(nn.Module):

#     def forward(self, x: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

#         x = x.unsqueeze(-1)
#         t = t.unsqueeze(-2)
#         w = w.unsqueeze(0)

#         x_comp = 1 - x
#         t_comp = 1 - t
#         w_comp = 1 - w

#         comp = torch.tensor(1.0, dtype=x_comp.dtype, device=x_comp.device)

#         rel_x = 1 - torch.min(
#             (t_comp - x_comp) / (1 - x_comp), comp
#         ).sum(dim=0, keepdim=True) / torch.sum(x_comp, dim=0, keepdim=True)

#         rel_w = 1 - torch.min(
#             (t_comp - w_comp) / (1 - w_comp), comp
#         ).sum(dim=-2, keepdim=True) / torch.sum(w_comp, dim=-2, keepdim=True)

# #         # for max prod use this for the inner
# #         # t = 0.2, x=0.4 => min(t / x, 1.0)

#         ind_x = torch.min(x + rel_x - x * rel_x, keepdim=True, dim=-2)[1]
#         ind_w = torch.min(w + rel_w - w * rel_w, keepdim=True, dim=-2)[1]
#         inner_x = torch.max(x, w.detach())
#         inner_w = torch.max(x.detach(), w)

#         chosen_x = inner_x.gather(-2, ind_x)
#         chosen_w = inner_w.gather(-2, ind_w)
#         return chosen_x.squeeze(-2), chosen_w.squeeze(-2)


# class WrapNeuron(WrapNN):

#     def __init__(self, f, rel: nn.Module):
#         super().__init__(
#             [self.x_hook, self.weight_hook], [self.out_hook]
#         )
#         self.f = f
#         self.rel = rel
#         self.loss = nn.MSELoss()
#         self.x_weight = 1.0
#         self.w_weight = 1.0

#     def out_hook(self, grad: torch.Tensor, state: WrapState, idx: int):
        
#         state.set_grad(grad, state, idx)
#         x = state.x[0].detach().clone()
#         w = state.x[1].detach().clone()
#         t = state.t.detach()
#         # diff = (state.y - t).abs()
#         # print(t.min(), t.max(), diff.min(), diff.max())
#         with torch.enable_grad():
#             x.requires_grad_()
#             x.retain_grad()
#             w.requires_grad_()
#             w.retain_grad()
#             chosen_x, chosen_w = self.rel(
#                 x, w, t
#             )
#             loss1 = self.loss(chosen_x, t)
#             loss2 = self.loss(chosen_w, t)
#             (loss1 + loss2).backward()
#             state.state['x_grad'] = x.grad
#             state.state['w_grad'] = w.grad
#         return grad

#     def weight_hook(self, grad: torch.Tensor, state: WrapState, idx: int):
        
#         return grad + self.w_weight * state.state['w_grad']
        
#     def x_hook(self, grad: torch.Tensor, state: WrapState, idx: int):
#         return grad + self.x_weight * state.state['x_grad']

#     def __call__(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:

#         return self.wrap(
#             self.f, x, w
#         )
