# 3rd party
from torch import nn
import torch

# local
from .._functional import (
    heaviside, sign, clamp, BindG, G
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

    def __init__(self, grad: bool = True, g: G=BindG):
        """Create a Sign module

        Args:
            grad (bool, optional): The gradient. Defaults to True.
        """
        super().__init__()
        self._grad = grad
        if g is BindG:
            g = BindG(1.0)
        self.g = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return sign(x, self.g)


class Heaviside(nn.Module):
    """The heaviside step function
    """

    def __init__(self, grad: bool = True, g: G=BindG):
        """Instantiate the step function

        Args:
            grad (bool, optional): Whether to use the straight through estimator. Defaults to True.
        """
        super().__init__()
        self._grad = grad
        if g is BindG:
            g = BindG(0.0, 1.0)
        self.g = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Binarize the input

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The binarized input
        """
        return heaviside(x, self.g)


class Clamp(nn.Module):
    """Clamps the input between two values
    """

    def __init__(self, lower: float=-1.0, upper: float=1.0, g: G=None):
        """Create a clamp module

        Args:
            lower (float, optional): The lower value in the range. Defaults to -1.0.
            upper (float, optional): The upper value in the range. Defaults to 1.0.
            g (G, optional): The gradient estimator to use. Defaults to True.
        """
        super().__init__()
        self._lower = lower
        self._upper = upper
        self.g = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take the clamp of the input

        Args:
            x (torch.Tensor): the input

        Returns:
            torch.Tensor: The clamped input
        """
        return clamp(x, self._lower, self._upper, self.g)
