# 3rd party
import torch
import torch.nn as nn


class Dropout(nn.Module):
    """Dropout is designed to work with logical neurons
    It does not divide the output by p and can be set to "dropout" to
    any value. This is because for instance And neurons should
    dropout to 1 not to 0.
    """

    def __init__(self, p: float, val: float=0.0):
        """Create a dropout neuron to dropout logical inputs

        Args:
            p (float): The dropout probability
            val (float, optional): The value to dropout to. Defaults to 0.0.
        """

        super().__init__()
        self.p = p
        self.val = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): the input

        Returns:
            torch.Tensor: the dropped out tensor
        """
        if self.training:
            x[(torch.rand_like(x) > self.p)] = self.val
        
        return x
