# 1st party
import typing

# 3rd party
import torch

TENSOR_FLOAT = typing.Union[torch.Tensor, float]

from ._grad import (
    HeavisideG, ClampG,
    SignG, G
)


def heaviside(x: torch.Tensor, g: G=None) -> torch.Tensor:
    """Convenience function to use the straight through estimator for binary

    Args:
        x1 (torch.Tensor): The tensor to binarize

    Returns:
        torch.Tensor: The binarized tensor
    """
    if g is None:
        return (x >= 0.0).type_as(x)
    return HeavisideG.apply(x, g)


def sign(x: torch.Tensor, g: G=None) -> torch.Tensor:
    """Convenience function to use the straight through estimator for sign

    Args:
        x1 (torch.Tensor): First tensor

    Returns:
        torch.Tensor: The signed tensor
    """
    if g is None:
        return x.sign()
    return SignG.apply(x, g)


def clamp(x: torch.Tensor, min_val: float=0.0, max_val: float=1.0, g: G=None) -> torch.Tensor:
    """Convenience function to use the straight through estimator for ramp

    Args:
        x1 (torch.Tensor): First tensor

    Returns:
        torch.Tensor: The signed tensor
    """
    if g is None:
        return x.clamp(min_val, max_val)
    return ClampG.apply(x, min_val, max_val, g)


def threshold(x: torch.Tensor, threshold: torch.Tensor, g: G=None) -> torch.Tensor:
    """

    Args:
        x (torch.Tensor): The value to threshold
        threshold (torch.Tensor): The threshold
        g (bool, optional): Where to use the ste. Defaults to False.

    Returns:
        torch.Tensor: The thresholded value
    """
    adjusted = x - threshold
    return heaviside(adjusted, g)


def ramp(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, g: G=None) -> torch.Tensor:
    """Use a ramp 

    Args:
        x (torch.Tensor): The value to ramp
        lower (torch.Tensor): The lower value for the ramp
        upper (torch.Tensor): The upper value for the ramp
        g (bool, optional): Whether to use the ste. Defaults to False.
        clip (float, optional): Value to clip by if using g. Defaults to None.

    Returns:
        torch.Tensor: The ramped 
    """
    scaled = (x - lower) / (upper - lower + 1e-5)
    return clamp(scaled, 0.0, 1.0, g)


def to_boolean(signed: torch.Tensor) -> torch.Tensor:
    """Convert a signed (neg ones/ones) tensor to a binary one (zeros/ones)

    Args:
        signed (torch.Tensor): The signed tensor

    Returns:
        torch.Tensor: The boolean tensor
    """
    return (signed + 1) / 2


def to_signed(boolean: torch.Tensor) -> torch.Tensor:
    """Convert a binary (zeros/ones) tensor to a signed one

    Args:
        boolean (torch.Tensor): The binary tensor

    Returns:
        torch.Tensor: The signed tensor
    """
    return (boolean * 2) - 1
