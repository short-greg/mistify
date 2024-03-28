from ..infer import Or, And
import torch.nn as nn
import torch
import typing
from functools import partial
from abc import abstractmethod


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
