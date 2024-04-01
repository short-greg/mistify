from ..infer import Or, And
import torch.nn as nn
import torch
import typing
from functools import partial
from abc import abstractmethod


# Later simplify these

class MaxMinRelOut(nn.Module):

    def forward(self, x: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(-1)
        t = t.unsqueeze(-2)
        w = w.unsqueeze(0)

        rel_x = torch.min(
            x, t
        ).sum(dim=0, keepdim=True) / torch.sum(x, dim=0, keepdim=True)

        rel_w = torch.min(
            w, t
        ).sum(dim=-2, keepdim=True) / torch.sum(x, dim=-2, keepdim=True)

        # for max prod use this for the inner
        # t = 0.2, x=0.4 => min(t / x, 1.0)

        ind_x = torch.max(torch.min(x, rel_x), keepdim=True, dim=-2)[1]
        ind_w = torch.max(torch.min(rel_w, w), keepdim=True, dim=-2)[1]
        inner_x = torch.min(x, w.detach())
        inner_w = torch.min(x.detach(), w)

        chosen_x = inner_x.gather(-2, ind_x)
        chosen_w = inner_w.gather(-2, ind_w)
        return chosen_x, chosen_w


class MaxProdRelOut(nn.Module):

    def forward(self, x: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(-1)
        t = t.unsqueeze(-2)
        w = w.unsqueeze(0)

        rel_x = torch.min(
            t / (x + 1e-7),  1.0
        ).sum(dim=0, keepdim=True) / torch.sum(x, dim=0, keepdim=True)

        rel_w = torch.min(
            t / (w + 1e-7), 1.0
        ).sum(dim=-2, keepdim=True) / torch.sum(x, dim=-2, keepdim=True)

        # for max prod use this for the inner
        # t = 0.2, x=0.4 => min(t / x, 1.0)

        ind_x = torch.max(x * rel_x, keepdim=True, dim=-2)[1]
        ind_w = torch.max(rel_w * w, keepdim=True, dim=-2)[1]
        inner_x = x * w.detach()
        inner_w = x.detach() * w

        chosen_x = inner_x.gather(-2, ind_x)
        chosen_w = inner_w.gather(-2, ind_w)
        return chosen_x, chosen_w


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
        return chosen_x, chosen_w


class MinSumRelOut(nn.Module):

    def forward(self, x: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(-1)
        t = t.unsqueeze(-2)
        w = w.unsqueeze(0)

        x_comp = 1 - x
        t_comp = 1 - t
        w_comp = 1 - w

        rel_x = 1 - torch.min(
            (t_comp - x_comp) / (1 - x_comp), 1.0
        ).sum(dim=0, keepdim=True) / torch.sum(x_comp, dim=0, keepdim=True)

        rel_w = 1 - torch.min(
            (t_comp - w_comp) / (1 - w_comp), 1.0
        ).sum(dim=-2, keepdim=True) / torch.sum(w_comp, dim=-2, keepdim=True)

#         # for max prod use this for the inner
#         # t = 0.2, x=0.4 => min(t / x, 1.0)

        ind_x = torch.min(x + rel_x - x * rel_x, keepdim=True, dim=-2)[1]
        ind_w = torch.min(w + rel_w - w * rel_w, keepdim=True, dim=-2)[1]
        inner_x = torch.max(x, w.detach())
        inner_w = torch.max(x.detach(), w)

        chosen_x = inner_x.gather(-2, ind_x)
        chosen_w = inner_w.gather(-2, ind_w)
        return chosen_x, chosen_w


#  var, operation, context,  x = op.constrain(x)


class WrapNeuron(nn.Module):

    def __init__(self, neuronf: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__()
        
        self.neuronf = neuronf
        self.enabled = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def x_update(
        self, grad: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, state: typing.Dict
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def weight_update(
        self, grad: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, state: typing.Dict
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        pass

    def store(self, grad, y, state):

        state[grad] = grad
        state[y] = y

    def forward(self, x: torch.Tensor, weight: torch.Tensor):

        if self.enabled:
            state = {}
            x.register_hook(partial(self.x_update, x=x, weight=weight, state=state))
            weight.register_hook(partial(self.weight_update, x=x, weight=weight, state=state))
            y = self.neuronf(x, weight)
            y.register_hook(partial(self.store, y=y, state=state))
            return y
        return self.neuronf(x, weight)
