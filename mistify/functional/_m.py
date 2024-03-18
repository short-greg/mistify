# 1st party
import typing

# 3rd party
import torch

TENSOR_FLOAT = typing.Union[torch.Tensor, float]

from ._grad import (
    BinaryG, ClampG,
    SignG
)

def binary(x: torch.Tensor, g: bool=False, clip: float=None) -> torch.Tensor:
    """Convenience function to use the straight through estimator for binary

    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor

    Returns:
        torch.Tensor: The binarized tensor
    """
    if g is False:
        clip = None
    return BinaryG.apply(x, clip=clip)


def sign(x: torch.Tensor, g: bool=False, clip: float=None) -> torch.Tensor:
    """Convenience function to use the straight through estimator for sign

    Args:
        x1 (torch.Tensor): First tensor

    Returns:
        torch.Tensor: The signed tensor
    """
    if g is False:
        clip = None
    return SignG.apply(x, clip=clip)


def ramp(x: torch.Tensor, g: bool=False, clip: float=None) -> torch.Tensor:
    """Convenience function to use the straight through estimator for ramp

    Args:
        x1 (torch.Tensor): First tensor

    Returns:
        torch.Tensor: The signed tensor
    """
    if g is False:
        clip = None
    return ClampG.apply(x, clip=clip)


def to_binary(signed: torch.Tensor) -> torch.Tensor:
    """Convert a signed (neg ones/ones) tensor to a binary one (zeros/ones)

    Args:
        signed (torch.Tensor): The signed tensor

    Returns:
        torch.Tensor: The binary tensor
    """
    return (signed + 1) / 2


def to_signed(binary: torch.Tensor) -> torch.Tensor:
    """Convert a binary (zeros/ones) tensor to a signed one

    Args:
        binary (torch.Tensor): The binary tensor

    Returns:
        torch.Tensor: The signed tensor
    """
    return (binary * 2) - 1
