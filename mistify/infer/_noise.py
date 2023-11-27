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
            x = x.clone()
            x[(torch.rand_like(x) > self.p)] = self.val
        
        return x


class Gaussian(nn.Module):
    """Gaussian noise adds random Gaussian noise to the membership
    """

    def __init__(self, std: float, min_=0.0, max_=1.0, dim: int=-1):
        """Add Gaussian noise to the input

        Args:
            std (float): the standard deviation of the noise
            min_ (float, optional): the min value for the output. Defaults to 0.0.
            max_ (float, optional): the max value for the output. Defaults to 1.0.
            dim (int, optional): the dimension
        """
        super().__init__()

        self.std = std
        self.min_ = min_
        self.max_ = max_
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): the input

        Returns:
            torch.Tensor: the dropped out tensor
        """
        if self.training:
            shape = list(x.shape)
            shape[self.dim] = 1
            return torch.clamp(x + torch.randn(*shape, dtype=x.dtype, device=x.device) * self.std, self.min_, self.max_)
        
        return x


class Exp(nn.Module):
    """Gaussian noise adds random Gaussian noise to the membership
    """

    def __init__(self, min_:float =0.8, max_: float=1.2, dim: int=-1):
        """_summary_

        Args:
            min_ (float, optional): The minimum value for the exponent. Defaults to 0.8.
            max_ (float, optional): The maximum value for the exponent. Defaults to 1.2.
            dim (int, optional): The dimension to add noise to. Defaults to -1.
        """
        super().__init__()

        self._min = min_
        self._max = max_
        self._mul = self._max - self._min
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): the input

        Returns:
            torch.Tensor: the dropped out tensor
        """
        if self.training:
            shape = list(x.shape)
            shape[self.dim] = 1
            return x.exp(torch.rand(*shape, dtype=x.dtype, device=x.device) * self._mul + self._min)
        
        return x
