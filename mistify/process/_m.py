# 3rd party
from torch import nn
import torch

# local
from .._functional import (
    binarize, signify, clamp, BindG
)


class Argmax(nn.Module):
    """An Argmax module
    """

    def __init__(self, dim=-1):
        """Take the argmax

        Args:
            dim (int, optional): The dimension to take the max on. Defaults to -1.
        """
        super().__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        """Compute the Argmax

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.LongTensor: The argmax
        """
        return torch.argmax(x, dim=-1)


class Sign(nn.Module):
    """The Sign module
    """

    def __init__(self, grad: bool = True):
        """Create a Sign module

        Args:
            grad (bool, optional): The gradient. Defaults to True.
        """
        super().__init__()
        self._grad = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return signify(x, BindG(1.0))


class Boolean(nn.Module):
    """The step function
    """

    def __init__(self, grad: bool = True):
        """Instantiate the step function

        Args:
            grad (bool, optional): Whether to use the straight through estimator. Defaults to True.
        """
        super().__init__()
        self._grad = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Binarize the input

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The binarized input
        """
        return binarize(x, BindG(0.0, 1.0))


class Clamp(nn.Module):
    """Clamps the input between two values
    """

    def __init__(self, lower: float=-1.0, upper: float=1.0):
        """Create a clamp module

        Args:
            lower (float, optional): The lower value in the range. Defaults to -1.0.
            upper (float, optional): The upper value in the range. Defaults to 1.0.
            grad (bool, optional): Wht. Defaults to True.
        """
        super().__init__()
        self._lower = lower
        self._upper = upper

    def forward(self, x: torch.Tensor):
        return clamp(x, self._lower, self._upper)
