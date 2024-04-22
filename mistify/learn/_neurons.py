from ..infer import Or, And
import torch.nn as nn
import torch
import typing
from functools import partial
from abc import abstractmethod
from zenkai import WrapNN, WrapState
from typing_extensions import Self


# Later simplify these

class MaxMinRelOut(nn.Module):

    def forward(self, x: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(-1)
        t = t.unsqueeze(-2)
        w = w.unsqueeze(0)

        with torch.no_grad():
            rel_x = torch.min(
                x, t
            ).sum(dim=0, keepdim=True) / torch.sum(x, dim=0, keepdim=True)

            rel_w = torch.min(
                w, t
            ).sum(dim=-2, keepdim=True) / torch.sum(w, dim=-2, keepdim=True)

            # for max prod use this for the inner
            # t = 0.2, x=0.4 => min(t / x, 1.0)

            ind_w = torch.max(torch.min(x, rel_x), keepdim=True, dim=-2)[1]
            ind_x = torch.max(torch.min(rel_w, w), keepdim=True, dim=-2)[1]
        inner_x = torch.min(x, w.detach())
        inner_w = torch.min(x.detach(), w)

        chosen_x = inner_x.gather(-2, ind_x)
        chosen_w = inner_w.gather(-2, ind_w)
        return chosen_x.squeeze(-2), chosen_w.squeeze(-2)


class MaxProdRelOut(nn.Module):

    def forward(self, x: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(-1)
        t = t.unsqueeze(-2)
        w = w.unsqueeze(0)
        comp = torch.tensor(1.0, dtype=x.dtype, device=x.device)

        rel_x = torch.min(
            t / (x + 1e-7),  comp
        ).sum(dim=0, keepdim=True) / torch.sum(x, dim=0, keepdim=True)

        rel_w = torch.min(
            t / (w + 1e-7), comp
        ).sum(dim=-2, keepdim=True) / torch.sum(x, dim=-2, keepdim=True)

        # for max prod use this for the inner
        # t = 0.2, x=0.4 => min(t / x, 1.0)

        ind_x = torch.max(x * rel_x, keepdim=True, dim=-2)[1]
        ind_w = torch.max(rel_w * w, keepdim=True, dim=-2)[1]
        inner_x = x * w.detach()
        inner_w = x.detach() * w

        chosen_x = inner_x.gather(-2, ind_x)
        chosen_w = inner_w.gather(-2, ind_w)
        return chosen_x.squeeze(-2), chosen_w.squeeze(-2)


class MinMaxRelOut(nn.Module):

    def forward(self, x: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(-1)
        t = t.unsqueeze(-2)
        w = w.unsqueeze(0)

        x_comp = 1 - x
        t_comp = 1 - t
        w_comp = 1 - w

        rel_x = 1 - torch.min(
            x_comp, t_comp
        ).sum(dim=0, keepdim=True) / torch.sum(x_comp, dim=0, keepdim=True)

        rel_w = 1 - torch.min(
            w_comp, t_comp
        ).sum(dim=-2, keepdim=True) / torch.sum(w_comp, dim=-2, keepdim=True)

        # for max prod use this for the inner
        # t = 0.2, x=0.4 => min(t / x, 1.0)

        ind_x = torch.min(torch.max(x, rel_x), keepdim=True, dim=-2)[1]
        ind_w = torch.min(torch.max(rel_w, w), keepdim=True, dim=-2)[1]
        inner_x = torch.max(x, w.detach())
        inner_w = torch.max(x.detach(), w)

        chosen_x = inner_x.gather(-2, ind_x)
        chosen_w = inner_w.gather(-2, ind_w)
        return chosen_x.squeeze(-2), chosen_w.squeeze(-2)


class MinSumRelOut(nn.Module):

    def forward(self, x: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(-1)
        t = t.unsqueeze(-2)
        w = w.unsqueeze(0)

        x_comp = 1 - x
        t_comp = 1 - t
        w_comp = 1 - w

        comp = torch.tensor(1.0, dtype=x_comp.dtype, device=x_comp.device)

        rel_x = 1 - torch.min(
            (t_comp - x_comp) / (1 - x_comp), comp
        ).sum(dim=0, keepdim=True) / torch.sum(x_comp, dim=0, keepdim=True)

        rel_w = 1 - torch.min(
            (t_comp - w_comp) / (1 - w_comp), comp
        ).sum(dim=-2, keepdim=True) / torch.sum(w_comp, dim=-2, keepdim=True)

#         # for max prod use this for the inner
#         # t = 0.2, x=0.4 => min(t / x, 1.0)

        ind_x = torch.min(x + rel_x - x * rel_x, keepdim=True, dim=-2)[1]
        ind_w = torch.min(w + rel_w - w * rel_w, keepdim=True, dim=-2)[1]
        inner_x = torch.max(x, w.detach())
        inner_w = torch.max(x.detach(), w)

        chosen_x = inner_x.gather(-2, ind_x)
        chosen_w = inner_w.gather(-2, ind_w)
        return chosen_x.squeeze(-2), chosen_w.squeeze(-2)


class WrapNeuron(WrapNN):

    def __init__(self, f, rel: nn.Module):
        super().__init__(
            [self.x_hook, self.weight_hook], [self.out_hook]
        )
        self.f = f
        self.rel = rel
        self.loss = nn.MSELoss()
        self.x_weight = 1.0
        self.w_weight = 1.0

    def out_hook(self, grad: torch.Tensor, state: WrapState, idx: int):
        
        state.set_grad(grad, state, idx)
        x = state.x[0].detach().clone()
        w = state.x[1].detach().clone()
        t = state.t.detach()
        # diff = (state.y - t).abs()
        # print(t.min(), t.max(), diff.min(), diff.max())
        with torch.enable_grad():
            x.requires_grad_()
            x.retain_grad()
            w.requires_grad_()
            w.retain_grad()
            chosen_x, chosen_w = self.rel(
                x, w, t
            )
            loss1 = self.loss(chosen_x, t)
            loss2 = self.loss(chosen_w, t)
            (loss1 + loss2).backward()
            state.state['x_grad'] = x.grad
            state.state['w_grad'] = w.grad
        return grad

    def weight_hook(self, grad: torch.Tensor, state: WrapState, idx: int):
        
        return grad + self.w_weight * state.state['w_grad']
        
    def x_hook(self, grad: torch.Tensor, state: WrapState, idx: int):
        return grad + self.x_weight * state.state['x_grad']

    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:

        return self.wrap(
            self.f, x, w
        )
