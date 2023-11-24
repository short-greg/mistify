# 1st party
from abc import abstractmethod

# 3rd party
import torch.nn as nn
import torch
from torch.distributions import Normal


class Processor(nn.Module):
    """
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        pass


class GaussianBase(Processor):

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):

        super().__init__()
        self._mean = mean[None]
        self._std = std[None]
        self._normal = Normal(mean, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError

    @property
    def std(self) -> torch.Tensor:

        return self._std
    
    @property
    def mean(self) -> torch.Tensor:

        return self._mean

    @classmethod
    def fit(cls, X: torch.Tensor) -> torch.Tensor:

        return cls(X.mean(dim=0), X.std(dim=0))
        

class StdDev(GaussianBase):
    """Use the standard deviation to preprocess the inputs
    """

    def __init__(self, std: torch.Tensor, mean: torch.Tensor=None, divisor: float=1, offset: float=0.):
        """Preprocess with the standard deviation. The standard deviation will be
        multiplied by the divisor. If the default 1 is used roughly 67% of the data will
        be in the range -1 and 1. The percentage can be increased by increasing the divisor

        Args:
            std (torch.Tensor): The standard deviation of the data
            mean (torch.Tensor): The mean to offset by
            divisor (float, optional): The amount to multiply t. Defaults to 1.
            offset (float, optional): The amount to offset the input by. Can use to get
            the final output to be mostly between 0 and 1
        """
        super().__init__(mean if isinstance(mean, torch.Tensor) else 0.0, std)
        self._divisor = divisor
        self._offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        return (x - self._mean + self._offset) / (self._std * self._divisor)

    def reverse(self, x: torch.Tensor) -> torch.Tensor:

        # std = self._align(x, self._std, self._dim)
        # mean = self._align(x, self._mean, self._dim)

        return (x * (self._std * self._divisor)) + self._mean - self._offset

    @classmethod
    def fit(cls, X: torch.Tensor, divisor: float=1.0, offset: float=0.0) -> torch.Tensor:

        return cls(X.std(dim=0), X.mean(dim=0), divisor=divisor, offset=offset)


class CumGaussian(GaussianBase):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map the data to percentiles

        Args:
            x (torch.Tensor): the input

        Returns:
            torch.Tensor: The percentile output
        """

        z = (x - self._mean) / (self._std * torch.sqrt(torch.tensor(2.0)))
        return 0.5 * (1 + torch.erf(z))

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        """Map the data from percentiles to values

        Args:
            x (torch.Tensor): percentile

        Returns:
            torch.Tensor: value
        """
        return self._mean + self._std * torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * x - 1)


class LogisticBase(Processor):

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):

        super().__init__()
        self._loc = loc[None]
        self._scale = scale[None]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError

    @property
    def scale(self) -> torch.Tensor:

        return self._scale
    
    @property
    def loc(self) -> torch.Tensor:

        return self._loc

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
        scale.requires_grad_()
        scale.retain_grad()
        
        for _ in range(iterations):
            log_likelihood = cls.log_pdf(X, mean, scale)
            (-log_likelihood.mean()).backward()
            scale.data = scale - lr * scale.grad
            scale.grad = None

        scale = scale.detach()
        scale.requires_grad_(False)
        return cls(mean, scale)


class CumLogistic(LogisticBase):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map the data to percentiles

        Args:
            x (torch.Tensor): the input

        Returns:
            torch.Tensor: The percentile output
        """
        return torch.sigmoid(
            (x - self.loc) / self._scale
        )

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        """Map the data from percentiles to values

        Args:
            x (torch.Tensor): The percentile input

        Returns:
            torch.Tensor: The value
        """

        return (torch.logit(x) * self._scale) + self._loc


class SigmoidParam(LogisticBase):

    def __init__(self, n_features: int):

        super().__init__(
            nn.parameter.Parameter(torch.randn(n_features)),
            nn.parameter.Parameter(torch.rand(n_features))
        )
    
    # def _align(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:

    #     unsqueeze = [1] * x.dim()
    #     unsqueeze[self._dim] = 0
    #     for i, u in enumerate(unsqueeze):
    #         if u == 1:
    #             p = p.unsqueeze(i)
    #     return p

    def reverse(self, x: torch.Tensor) -> torch.Tensor:

        # loc = self._align(x, self._loc, self._dim)
        # scale = self._align(x, self._scale, self._dim)

        return (torch.logit(x) * self._scale) + self._loc

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # loc = self._align(x, self._loc, self._dim)
        # scale = self._align(x, self._scale, self._dim)

        return torch.sigmoid(
            (x - self._loc) / self._scale
        )


class MinMaxScaler(Processor):

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        """

        Args:
            lower (torch.Tensor): The lower value for scaling
            upper (torch.Tensor): The upper value for scaling
        """
        super().__init__()
        self._lower = lower[None]
        self._upper = upper[None]

    @property
    def lower(self) -> torch.Tensor:
        return self._lower

    @property
    def upper(self) -> torch.Tensor:
        return self._upper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform MinMax scaling

        Args:
            x (torch.Tensor): The unscaled x

        Returns:
            torch.Tensor: the scaled x
        """
        return (x - self._lower) / (self._upper - self._lower + 1e-5)
    
    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        """Undo MinMax scaling

        Args:
            x (torch.Tensor): The scaled input

        Returns:
            torch.Tensor: The unscaled output
        """
        return (x * (self._upper - self._lower + 1e-5)) + self._lower

    @classmethod
    def fit(cls, X: torch.Tensor) -> 'MinMaxScaler':
        """Fit the scaling parameters

        Args:
            X (torch.Tensor): The training data

        Returns:
            MinMaxScaler: The scaler based on the X
        """

        return MinMaxScaler(
            X.min(dim=0)[0], X.max(dim=0)[0]
        )
