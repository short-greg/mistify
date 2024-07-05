# 3rd party
import torch.nn as nn
import torch

# local
from ..fuzzify import Fuzzifier
from ..infer import And


class Sugeno(nn.Module):
    """Allows to define variations on the ANFIS algorithm  
    """
    
    def __init__(
        self, in_features: int, out_features: int, out_terms: int, 
        fuzzifier: Fuzzifier
    ):
        """Create an Sugeno model

        Args:
            in_features (int): The number of in features
            out_features (int): The number of out features
            out_terms (int): The number of out terms
            fuzzifier (Fuzzifier): The fuzzifier to use for fuzzification
        """
        super().__init__()
        self._and = And(in_features * fuzzifier.n_terms, out_features * out_terms)
        self._linear = nn.Linear(in_features, out_features * out_terms)
        self._fuzzifier = fuzzifier
        self._out_features = out_features
        self._out_terms = out_terms
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the output of ANFIS

        Args:
            x (torch.Tensor): The input to the network

        Returns:
            torch.Tensor: The output
        """
        x = self._linear(x)
        m = self._fuzzifier(x).view(x.shape[0], self._out_features * self._out_terms)
        m = self._and(m)

        x = x.reshape(x.shape[0], self._out_features, -1)
        m = m.reshape(m.shape[0], self._out_features, -1)
        return torch.sum(m * x, dim=-1) / torch.sum(m, dim=-1)


# class ANFIS(nn.Module):
#     """Allows to define variations on the ANFIS algorithm  
#     """
    
#     def __init__(self, fuzzy_system: nn.Module, net: nn.Module, out_features: int):
#         """Create a system that implements ANFIS

#         Args:
#             fuzzy_system (nn.Module): A system that outputs membership values. The output
#                 shape must be the compatible with net's
#             net (nn.Module): A standard neural network. Possibly just a linear
#             out_features (int): The number of out features for ANFIS
#         """
#         super().__init__()
#         self._fuzzy_system = fuzzy_system
#         self._net = net
#         self._out_features = out_features
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Evaluate the output of ANFIS

#         Args:
#             x (torch.Tensor): The input to the network

#         Returns:
#             torch.Tensor: The output
#         """
#         x = self._net(x)
#         x = x.reshape(x.shape[0], self._out_features, -1)
#         m = self._fuzzy_system(x).view(x.shape[0], self._out_features, -1)
#         return torch.sum(m * x, dim=-1) / torch.sum(m, dim=-1)
