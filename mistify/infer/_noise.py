# 3rd party
import torch
import torch.nn as nn


class DropoutNoise(nn.Module):
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

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): the input

        Returns:
            torch.Tensor: the dropped out tensor
        """
        if self.training and self.p is not None:
            m = m.clone()
            m[(torch.rand_like(m) < self.p)] = self.val
        
        return m


class GaussianClampNoise(nn.Module):
    """Gaussian noise adds random Gaussian noise to the membership
    """

    def __init__(self, std: float, min_=0.0, max_=1.0, dim: int=-1):
        """Add Gaussian noise to the input that will be clamped to be
        between a lower bound and an upper bound

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

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Add gaussian noise to m that will be clamped

        Args:
            m: (torch.Tensor): the input

        Returns:
            torch.Tensor: the dropped out tensor
        """
        if self.training:
            shape = list(m.shape)
            shape[self.dim] = 1
            return torch.clamp(m + torch.randn(*shape, dtype=m.dtype, device=m.device) * self.std, self.min_, self.max_)
        
        return m


class ExpNoise(nn.Module):
    """Gaussian noise adds random Gaussian noise to the membership
    """

    def __init__(self, min_:float =0.8, max_: float=1.2, dim: int=-1):
        """Create Exponential noise which multiplies the input by an exponent. The exponent must
        be greater than 0 to work properly

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

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Add exponential noise to m

        Args:
            m (torch.Tensor): the input

        Returns:
            torch.Tensor: the dropped out tensor
        """
        if self.training:
            shape = list(m.shape)
            shape[self.dim] = 1
            return m.exp(torch.rand(*shape, dtype=m.dtype, device=m.device) * self._mul + self._min)
        
        return m
