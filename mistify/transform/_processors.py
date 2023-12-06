# 1st party
from abc import abstractmethod

# 3rd party
import torch.nn as nn
import torch
from torch.distributions import Normal
import typing


class Transform(nn.Module):
    """
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def fit(self, X: torch.Tensor, t=None, *args, **kwargs):
        pass

    def fit_transform(self, X: torch.Tensor, t=None, *args, **kwargs):
        
        self.fit(X, t, *args, **kwargs)
        return self(X)


class GaussianBase(Transform):

    def __init__(self, mean: torch.Tensor=0.0, std: torch.Tensor=1.0):

        super().__init__()
        if not isinstance(mean, torch.Tensor) and mean is not None:
            mean = torch.tensor(mean, dtype=torch.float32)
        if not isinstance(std, torch.Tensor) and std is not None:
            std = torch.tensor(std, dtype=torch.float32)
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

    def fit(self, X: torch.Tensor, t=None):

        self._mean = X.mean(dim=0, keepdim=True)
        self._std = X.std(dim=0, keepdim=True)
        

class StdDev(GaussianBase):
    """Use the standard deviation to preprocess the inputs
    """

    def __init__(self, std: torch.Tensor=1.0, mean: torch.Tensor=None, divisor: float=1, offset: float=0.):
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
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        return (x - self._mean + self.offset) / (self._std * self._divisor)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:

        # std = self._align(x, self._std, self._dim)
        # mean = self._align(x, self._mean, self._dim)

        return (y * (self._std * self._divisor)) + self._mean - self.offset

    @property
    def divisor(self) -> float:
        return self._divisor
    
    @divisor.setter
    def divisor(self, divisor: float) -> float:

        if divisor <= 0:
            raise ValueError(f'Divisor must be greater than 0 not {divisor}')
        self._divisor = divisor

    def fit(self, X: torch.Tensor, t=None) -> torch.Tensor:

        self._mean = X.mean(dim=0, keepdim=True)
        self._std = X.std(dim=0, keepdim=True)


class Compound(Transform):

    def __init__(self, transforms: typing.List[Transform], no_fit: typing.Set[int]=None):

        super().__init__()
        self._no_fit = no_fit or set()
        self._transforms: nn.ModuleList = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for transform in self._transforms:
            x = transform(x)
        return x
    
    def to_fit(self, i: int, to_fit: bool):

        if not to_fit:
            try:
                self._no_fit.remove(i)
            except KeyError:
                # Don't need to throw error if fit is set to false
                pass
        else:
            self._no_fit.add(i)
    
    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        
        for transform in reversed(self._transforms):
            y = transform.reverse(y)
        return y
    
    def fit(self, X: torch.Tensor, t: typing.Dict[int, torch.Tensor] = None, kwargs: typing.Dict[int, typing.Dict]=None):

        kwargs = kwargs or {}
        t = t or {}

        for i, transform in enumerate(self._transforms):
            cur_kwargs = kwargs.get(i, {})
            if i not in self._no_fit:
                transform.fit(X, t.get(i), **cur_kwargs)
            X = transform(X)


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

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Map the data from percentiles to values

        Args:
            x (torch.Tensor): percentile

        Returns:
            torch.Tensor: value
        """
        return self._mean + self._std * torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * y - 1)


class LogisticBase(Transform):

    def __init__(self, loc: torch.Tensor=0.0, scale: torch.Tensor=1.0):

        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32)
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

    def fit(self, X: torch.Tensor, t=None, lr: float=1e-2, iterations: int=1000):
        """Fit the logistic distribution function

        Args:
            X (torch.Tensor): the training inputs
            t (torch.Tensor, optional): the training targets. Defaults to None.
            lr (float, optional): the learning rate. Defaults to 1e-2.
            iterations (int, optional): the number of iterations to run. Defaults to 1000.
        """
        mean = X.mean(dim=0)
        scale = torch.ones_like(mean) + torch.randn_like(mean) * 0.05
        scale.requires_grad_()
        scale.retain_grad()
        
        for _ in range(iterations):
            log_likelihood = self.log_pdf(X, mean, scale)
            (-log_likelihood.mean()).backward()
            scale.data = scale - lr * scale.grad
            scale.grad = None

        scale = scale.detach()
        scale.requires_grad_(False)
        self._loc = mean[None]
        self._scale = scale[None]


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

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Map the data from percentiles to values

        Args:
            x (torch.Tensor): The percentile input

        Returns:
            torch.Tensor: The value
        """

        return (torch.logit(y) * self._scale) + self._loc


class SigmoidParam(LogisticBase):

    def __init__(self, n_features: int):

        super().__init__(
            nn.parameter.Parameter(torch.randn(n_features)),
            nn.parameter.Parameter(torch.rand(n_features))
        )

    def reverse(self, y: torch.Tensor) -> torch.Tensor:

        # loc = self._align(x, self._loc, self._dim)
        # scale = self._align(x, self._scale, self._dim)

        return (torch.logit(y) * self._scale) + self._loc

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # loc = self._align(x, self._loc, self._dim)
        # scale = self._align(x, self._scale, self._dim)

        return torch.sigmoid(
            (x - self._loc) / self._scale
        )


class MinMaxScaler(Transform):

    def __init__(self, lower: torch.Tensor=0.0, upper: torch.Tensor=1.0):
        """

        Args:
            lower (torch.Tensor): The lower value for scaling
            upper (torch.Tensor): The upper value for scaling
        """
        super().__init__()
        if not isinstance(lower, torch.Tensor) :
            lower = torch.tensor(lower, dtype=torch.float32)
        
        if not isinstance(upper, torch.Tensor):
            upper = torch.tensor(upper, dtype=torch.float32)
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
    
    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Undo MinMax scaling

        Args:
            x (torch.Tensor): The scaled input

        Returns:
            torch.Tensor: The unscaled output
        """
        return (y * (self._upper - self._lower + 1e-5)) + self._lower

    def fit(self, X: torch.Tensor, t=None):
        """Fit the scaling parameters

        Args:
            X (torch.Tensor): The training data

        Returns:
            MinMaxScaler: The scaler based on the X
        """

        self._lower = X.min(dim=0, keepdim=True)[0]
        self._upper = X.max(dim=0, keepdim=True)[0]


class Reverse(Transform):

    def __init__(self, processor: Transform):
        super().__init__()
        self.processor = processor

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.processor.reverse(x)
    
    def reverse(self, y: torch.Tensor) -> torch.Tensor:

        return self.processor(y)
    
    def fit(self, Y: torch.Tensor, t=None, *args, **kwargs):

        self.processor.fit(Y, t, *args, **kwargs)

    def fit_transform(self, Y: torch.Tensor, t=None, *args, **kwargs):

        self.processor.fit(Y, t, *args, **kwargs)
        return self.processor(Y)
