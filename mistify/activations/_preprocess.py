import torch.nn as nn
import torch
import numpy as np


class GaussianBase(nn.Module):

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):

        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError

    @property
    def std(self) -> torch.Tensor:

        return self._std
    
    @std.setter
    def std(self, std: torch.Tensor) -> torch.Tensor:

        assert (std > 0).all()
        self._divide_by = self._std * self._divisor
        self._std = std

    @classmethod
    def fit(cls, X: torch.Tensor) -> torch.Tensor:

        return cls(X.mean(dim=0), X.median(dim=0))
        

class StdDev(GaussianBase):

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, divisor: float=1):

        super().__init__(mean, std)
        self._divisor = 1
        self.divisor = divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            
        return (x - self.mean) / (self._std * self._divisor)

    @classmethod
    def fit(cls, X: torch.Tensor, divisor: float) -> torch.Tensor:

        return cls(X.mean(dim=0), X.median(dim=0), divisor=divisor)


class CumGaussian(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.erf(
            (x - self.mean) / self._std
        )

class LogisticBase(nn.Module):

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):

        super().__init__()
        self.loc = loc
        self.scale = scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError

    @property
    def scale(self) -> torch.Tensor:

        return self._scale
    
    @scale.setter
    def std(self, scale: torch.Tensor) -> torch.Tensor:

        assert (scale > 0).all()
        self._scale = scale

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



class CumLogistic(LogisticBase):

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.sigmoid(
            (x - self.loc) / self._scale
        )


class SigmoidP(nn.Module):

    def __init__(self, n_terms: int, dim: int=-1):

        super().__init__()
        self._scale = nn.parameter.Parameter(torch.randn(n_terms))
        self._loc = nn.parameter.Parameter(torch.randn(n_terms))
        self._dim = dim

    @property
    def scale(self) -> torch.Tensor:

        return self._scale
    
    @property
    def loc(self) -> torch.Tensor:

        return self._loc
    
    def _align(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:

        unsqueeze = [1] * x.dim()
        unsqueeze[self._dim] = 0
        for i, u in enumerate(unsqueeze):
            if u == 1:
                p = p.unsqueeze(i)
        return p

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        loc = self._align(x, self._loc, self._dim)
        scale = self._align(x, self._scale, self._dim)

        return torch.sigmoid(
            (x - loc) / scale
        )


class MinMaxScaler(nn.Module):

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):

        super().__init__()
        self._lower = lower[None]
        self._upper = upper[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return (x - self._lower) / (self._upper - self._lower + 1e-5)
    
    @classmethod
    def fit(cls, X: torch.Tensor) -> 'MinMaxScaler':

        return MinMaxScaler(
            X.min(dim=0), X.max(dim=0)
        )
