from abc import abstractmethod

from torch import nn
import torch
import typing


class ComplementBase(nn.Module):
    """Base complement class for calculating complement of a set
    """

    def __init__(self, concatenate_dim: int=None):
        """initializer

        Args:
            concatenate_dim (int, optional): 
              Dim to concatenate the complement with. If None, it does not concatenate.
              Defaults to None.
        """
        super().__init__()
        self.concatenate_dim = concatenate_dim

    def postprocess(self, m: torch.Tensor, m_complement: torch.Tensor) -> torch.Tensor:
        """Postprocess the complement

        Args:
            m (torch.Tensor): The input tensor
            m_complement (torch.Tensor): The complemented tensor

        Returns:
            torch.Tensor: The postprocessed tensor
        """
        if self.concatenate_dim is None:
            return m_complement
        
        return torch.cat(
            [m, m_complement], dim=self.concatenate_dim
        )
    
    @abstractmethod
    def complement(self, m: torch.Tensor) -> torch.Tensor:
        """Take complemento f tensor

        Args:
            m (torch.Tensor): Tensor to take complement of

        Returns:
            torch.Tensor: Complemented tensor
        """
        raise NotImplementedError

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Take complement of tesor

        Args:
            m (torch.Tensor): 

        Returns:
            torch.Tensor: 
        """
        return self.postprocess(m, self.complement(m))


class CompositionBase(nn.Module):

    def __init__(
        self, in_features: int, out_features: int, in_variables: int=None
    ):
        """Base class for taking relations between two tensor

        Args:
            in_features (int): Number of input features (i.e. terms)
            out_features (int): Number of outputs features (i.e. terms)
            in_variables (int, optional): Number of linguistic variables in. Defaults to None.
        """
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._multiple_variables = in_variables is not None
        self.weight = torch.nn.parameter.Parameter(
            self.init_weight(in_features, out_features, in_variables)
        )

    @abstractmethod
    def clamp_weights(self):
        pass
    
    @abstractmethod
    def init_weight(self, in_features: int, out_features: int, in_variables: int=None) -> torch.Tensor:
        pass
