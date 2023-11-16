import torch.nn as nn
import torch
import numpy as np
import sklearn.linear_model

class StdDev(nn.Module):

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, divisor: float=1):

        super().__init__()
        self.mean = mean
        self._divisor = 1
        self.std = std
        self.divisor = divisor
        self._divide_by = None

    @property
    def divisor(self) -> float:
        return self._divisor
    
    @divisor.setter
    def divisor(self, divisor: float) -> float:
        assert self._divisor > 0
        self._divide_by = self._std * self._divisor
        self._divisor = divisor
        return self._divisor

    @property
    def std(self) -> torch.Tensor:

        return self._std
    
    @std.setter
    def std(self, std: torch.Tensor) -> torch.Tensor:

        assert (std > 0).all()
        self._divide_by = self._std * self._divisor
        self._std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            
        return (x - self.mean) / self._divide_by

    @classmethod
    def fit(cls, X: torch.Tensor, divisor: float) -> torch.Tensor:

        return StdDev(X.mean(dim=0), X.median(dim=0), divisor=divisor)
        

class CumGaussian(nn.Module):

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):

        super().__init__()
        self.mean = mean[None]
        self.std = std[None]

    @property
    def std(self) -> torch.Tensor:

        return self._std
    
    @std.setter
    def std(self, std: torch.Tensor) -> torch.Tensor:

        assert (std > 0).all()
        self._divide_by = self._std * self._divisor
        self._std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.erf(
            (x - self.mean) / self._std
        )
    
    @classmethod
    def fit(cls, X: torch.Tensor) -> torch.Tensor:

        return CumGaussian(X.mean(dim=0), X.median(dim=0))
        

class CumLogistic(nn.Module):

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):

        super().__init__()
        self.loc = loc
        self.scale = scale

    @property
    def scale(self) -> torch.Tensor:

        return self._scale
    
    @scale.setter
    def std(self, scale: torch.Tensor) -> torch.Tensor:

        assert (scale > 0).all()
        self._scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.sigmoid(
            (x - self.loc) / self._scale
        )
    
    @classmethod
    def log_pdf(cls, X: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        
        mean = mean[None]
        scale = scale[None]

        core = torch.exp(-(X - mean) / scale) 

        numerator = torch.log(core)
        denominator = torch.log(
            scale * (1 + core) ** 2
        )
        return numerator - denominator

    @classmethod
    def fit(cls, X: torch.Tensor, lr: float=1e-2, iterations: int=1000) -> 'CumLogistic':

        mean = X.mean(dim=0)
        scale = torch.ones_like(mean) + torch.randn_like(mean) * 0.05
        scale.retain_grad()
        scale.requires_grad_()
        
        for _ in iterations:
            log_likelihood = cls.log_pdf(X, mean, scale)
            (-log_likelihood.mean()).backward()
            scale = scale - lr * scale.grad
            scale.grad = None

        scale = scale.detach()
        scale.requires_grad_(False)
        return CumLogistic(mean, scale)


class MinMaxScaler(nn.Module):

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):

        super().__init__()
        self._lower = lower[None]
        self._upper = upper[None]

    # @property
    # def lower(self) -> torch.Tensor:

    #     return self._lower
    
    # @lower.setter
    # def std(self, scale: torch.Tensor) -> torch.Tensor:

    #     assert (scale > 0).all()
    #     self._scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return (x - self._lower) / (self._upper - self._lower + 1e-5)
    
    @classmethod
    def fit(cls, X: torch.Tensor) -> 'MinMaxScaler':

        return MinMaxScaler(
            X.min(dim=0), X.max(dim=0)
        )
