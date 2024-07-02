# 3rd party
import torch.nn as nn
import torch
import torch.nn.functional as nn_func


class Reversible(object):
    """Mixin for a reversible module
    """
        
    def reverse(self, *y: torch.Tensor) -> torch.Tensor:
        """Reverse the input

        Returns:
            torch.Tensor: The output
        """
        pass


class Softplus(nn.Module, Reversible):
    """A module implementing Softplus
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the softplus

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The softplus function
        """
        return nn_func.softplus(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Compute the inverse of the softplus

        Args:
            y (torch.Tensor): The output

        Returns:
            torch.Tensor: The inverse of the softplus
        """
        return torch.log(
            torch.exp(y - 1)
        )


class Exp(nn.Module, Reversible):
    """A module implementing the Exp
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the input

        Args:
            x (torch.Tensor): the input

        Returns:
            torch.Tensor: the output
        """
        return torch.exp(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Reverse the output

        Args:
            y (torch.Tensor): The output

        Returns:
            torch.Tensor: The inverse of the exponent (log)
        """
        return torch.log(y)
