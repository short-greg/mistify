import torch.nn as nn
import torch

class ANFIS(nn.Module):
    """
    """
    
    def __init__(self, fuzzy_system: nn.Module, net: nn.Module, out_features: int):
        """Define various types of ANFIS

        Args:
            fuzzy_system (nn.Module): A system that outputs membership values. The output
                shape must be the compatible with net's
            net (nn.Module): A standard neural network. Possibly just a linear
            out_features (int): The number of out features for ANFIS
        """
        super().__init__()
        self._fuzzy_system = fuzzy_system
        self._net = net
        self._out_features = out_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self._net(x)
        x = x.reshape(x.shape[0], self._out_features, -1)
        m = self._fuzzy_system(x).view(x.shape[0], self._out_features, -1)
        return torch.sum(m * x, dim=-1) / torch.sum(m, dim=-1)