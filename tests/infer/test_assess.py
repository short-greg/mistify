# Test all conversion modules
import typing
import torch
from torch import nn
import torch.nn
from mistify.infer._assess import MistifyLoss, MistifyLossFactory


class ExampleMistifyLoss(MistifyLoss):

    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)
        self.mse = nn.MSELoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.mse((x + y), t)
    
    @classmethod
    def factory(cls, reduction: str="mean") -> 'MistifyLossFactory':
        return super().factory(reduction=reduction)


class TestMistifyLoss:

    def test_mistify_loss_factory_creates_example_mistify_loss(self):

        factory = ExampleMistifyLoss.factory('mean')
        assert isinstance(factory(), ExampleMistifyLoss)

    def test_forward_computes_mse(self):

        loss = ExampleMistifyLoss()
        x = torch.rand(2, 4)
        y = torch.rand(2, 4)
        t = torch.rand(2, 4)
        assert (((x + y) - t).pow(2).mean()) == loss(x, y, t)

