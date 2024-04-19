# 1st party
from abc import abstractmethod
import typing

# 3rd party
import torch.nn as nn
import torch
import torch.nn.functional
from torch.distributions import Normal

# local
from ._reverse import Reversible


class Transform(nn.Module, Reversible):
    """Preprocess or postprocess the input
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the transform

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The transformed input
        """
        pass

    @abstractmethod
    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Reverse the transform

        Args:
            y (torch.Tensor): The output of the transform

        Returns:
            torch.Tensor: The transform reversed
        """
        pass

    @abstractmethod
    def fit(self, X: torch.Tensor, t=None, *args, **kwargs):
        """Fit the transform on data

        Args:
            X (torch.Tensor): The input to fit on
            t (optional): The target to fit on if necessary. Defaults to None.
        """
        pass

    def fit_transform(self, X: torch.Tensor, t=None, *args, **kwargs) -> torch.Tensor:
        """Convenience method to fit the transform then apply the transform

        Args:
            X (torch.Tensor): The input
            t (optional): The target if needed. Defaults to None.

        Returns:
            torch.Tensor: the transformed tensor
        """
        self.fit(X, t, *args, **kwargs)
        return self(X)


class NullTransform(Transform):
    """Preprocess or postprocess the input
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the transform

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The transformed input
        """
        return x

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Reverse the transform

        Args:
            y (torch.Tensor): The output of the transform

        Returns:
            torch.Tensor: The transform reversed
        """
        return y

    def fit(self, X: torch.Tensor, t=None, *args, **kwargs):
        """Fit the transform on data

        Args:
            X (torch.Tensor): The input to fit on
            t (optional): The target to fit on if necessary. Defaults to None.
        """
        pass


class GaussianBase(Transform):

    def __init__(self, mean: torch.Tensor=0.0, std: torch.Tensor=1.0):
        """Base class for a Gaussian Transform

        Args:
            mean (torch.Tensor, optional): The mean. Defaults to 0.0.
            std (torch.Tensor, optional): The standard deviation. Defaults to 1.0.
        """
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
        """
        Returns:
            torch.Tensor: The standard deviation used
        """
        return self._std
    
    @property
    def mean(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The mean used
        """
        return self._mean

    def fit(self, X: torch.Tensor, t=None):
        """Fit the Gaussian with the input. t is not used for this

        Args:
            X (torch.Tensor): The input to fit on
            t (optional): The target to fit on. Defaults to None.
        """
        self._mean = X.mean(dim=0, keepdim=True)
        self._std = X.std(dim=0, keepdim=True)
        

class StdDev(GaussianBase):
    """Use the standard deviation to preprocess the inputs
    """

    def __init__(self, std: torch.Tensor=1.0, mean: torch.Tensor=None, divisor: float=1, offset: float=0.):
        """Preprocess with the standard deviation. The standard deviation will be
        multiplied by the divisor. If the default 1 is used roughly 67% of the data will
        be in the range -1 and 1. The percentage can be increased by increasing the divisor

        Useful if you use clamp after normalizing, otherwise it is just like scaling

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
        """Normalize the input to have mean of 0 and std of 1 and then divide
        by divisor

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The normalized input
        """
        return (x - self._mean + self.offset) / (self._std * self._divisor)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Descale the input using the standard deviation and the mean

        Args:
            y (torch.Tensor): The value to reverse

        Returns:
            torch.Tensor: The original value
        """

        return (y * (self._std * self._divisor)) + self._mean - self.offset

    @property
    def divisor(self) -> float:
        """
        Returns:
            float: The value to divide by after normalizing
        """
        return self._divisor
    
    @divisor.setter
    def divisor(self, divisor: float) -> float:
        """

        Args:
            divisor (float): The value to divide by after normalizing

        Raises:
            ValueError: If divisor is less than or equal to zero

        Returns:
            float: The divisor
        """
        if divisor <= 0:
            raise ValueError(f'Divisor must be greater than 0 not {divisor}')
        self._divisor = divisor


class Compound(Transform):

    def __init__(self, transforms: typing.List[Transform], no_fit: typing.Set[int]=None):
        """Perform multiple transformations

        Args:
            transforms (typing.List[Transform]): The transformations
            no_fit (typing.Set[int], optional): Set the transformations not to fit when fit is called. Defaults to None.
        """
        super().__init__()
        self._no_fit = no_fit or set()
        self._transforms: nn.ModuleList = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Send the input through each of the transformatios

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The transformed input
        """
        for transform in self._transforms:
            x = transform(x)
        return x
    
    def to_fit(self, i: int, to_fit: bool):
        """Set one of the transforms to fit or not to fit

        Args:
            i (int): the index
            to_fit (bool): whether to fit or not to fit
        """
        if not to_fit:
            try:
                self._no_fit.remove(i)
            except KeyError:
                # Don't need to throw error if fit is set to false
                pass
        else:
            self._no_fit.add(i)
    
    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """

        Args:
            y (torch.Tensor): 

        Returns:
            torch.Tensor: 
        """
        for transform in reversed(self._transforms):
            y = transform.reverse(y)
        return y
    
    def fit(self, X: torch.Tensor, t: typing.Dict[int, torch.Tensor] = None, kwargs: typing.Dict[int, typing.Dict]=None):
        """Fit the composite transform

        Args:
            X (torch.Tensor): The input
            t (typing.Dict[int, torch.Tensor], optional): The targets for each of the transforms if needed. Defaults to None.
            kwargs (typing.Dict[int, typing.Dict], optional): The kwargs for each of the transforms if needed. Defaults to None.
        """
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
        """The base logistic function

        Args:
            loc (torch.Tensor, optional): The location parameter. Defaults to 0.0.
            scale (torch.Tensor, optional): The scale parameter. Defaults to 1.0.
        """
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
        """
        Returns:
            torch.Tensor: The scale parameter
        """
        return self._scale
    
    @property
    def loc(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The location parameter
        """
        return self._loc

    @classmethod
    def log_pdf(cls, X: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Get the probability density output for an input

        Args:
            X (torch.Tensor): The input
            mean (torch.Tensor): The mean of the logistic
            scale (torch.Tensor): The scale of the logistic

        Returns:
            torch.Tensor: The density value
        """
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
        """Use a parameterized sigmoid for the transform

        Args:
            n_features (int): The number of features to train
        """
        super().__init__(
            nn.parameter.Parameter(torch.randn(n_features)),
            nn.parameter.Parameter(torch.rand(n_features))
        )

    def reverse(self, y: torch.Tensor) -> torch.Tensor:

        return (torch.logit(y) * self._scale) + self._loc

    def forward(self, x: torch.Tensor) -> torch.Tensor:

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
        """
        Returns:
            torch.Tensor: The lower bound for the scale
        """
        return self._lower

    @property
    def upper(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The upper bound for the scale
        """
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

        self._lower = X.min(dim=0, keepdim=False)[0][None]
        self._upper = X.max(dim=0, keepdim=False)[0][None]


class Reverse(Transform):

    def __init__(self, transform: Transform):
        """

        Args:
            transform (Transform): The transform to reverse
        """
        super().__init__()
        self.transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the member transform

        Args:
            x (torch.Tensor): The tensor to reverse

        Returns:
            torch.Tensor: The reversed tensor
        """

        return self.transform.reverse(x)
    
    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Compute the forward value of the member transform

        Args:
            y (torch.Tensor): The output

        Returns:
            torch.Tensor: The input
        """
        return self.transform(y)
    
    def fit(self, Y: torch.Tensor, t=None, *args, **kwargs):
        """Fit the reversed transform. Note: That this expects 
        the input to the transform that is reversed not the input
        to this transform

        Args:
            Y (torch.Tensor): The output of the reversed transform
            t (optional): The target if necessary. Defaults to None.
        """
        self.transform.fit(Y, t, *args, **kwargs)

    def fit_transform(self, Y: torch.Tensor, t=None, *args, **kwargs) -> torch.Tensor:
        """Run the fit process followed by the transform

        Args:
            Y (torch.Tensor): The value for the output to train on
            t (optional): The target to use if necessary. Defaults to None.

        Returns:
            torch.Tensor: The output of the transform
        """
        self.transform.fit(Y, t, *args, **kwargs)
        return self.transform(Y)


class PieceRange(nn.Module):

    def __init__(self, pieces: torch.Tensor, lower: float=0., upper: float=1.0, tunable: bool=False):
        """Use to create a range of values for the piecewise computation

        Args:
            n_pieces (int): The number of pieces in the range
            n_features (int, optional): Number of features - If none, use the same value for all featurse. Defaults to None.
            lower (float, optional): The lower bound on the range. Defaults to 0..
            upper (float, optional): The upper bound on the rane. Defaults to 1.0.
            tunable (bool, optional): Whether the parameters can be tuned. Defaults to False.
        """
        super().__init__()
        if pieces.dim() == 1:
            pieces = pieces[None, None]
            self._n_features = None
        else:
            pieces = pieces[None]
            self._n_features = pieces.size(1)

        if tunable:
            self._pieces = torch.nn.parameter.Parameter(pieces)
        else:
            self._pieces = pieces
        self._tunable = tunable
        self._lower = lower
        self._upper = upper
        self._diff = self._upper - self._lower

    @classmethod
    def linspace(self, n_pieces: int, n_features: int=None, lower: float=0., upper: float=1.0, tunable: bool=False):
        
        pieces = torch.linspace(lower, upper, n_pieces + 1)
        if n_features is not None:
            pieces = pieces[None].repeat(n_features, 1)

        return PieceRange(pieces, lower, upper, tunable)

    @classmethod
    def expand(self, pieces: torch.Tensor, n_features: int, lower: float=0., upper: float=1., tunable: bool=False):
        pieces = pieces[None].repeat(n_features, 1)[None]
        return PieceRange(pieces, lower, upper, tunable)

    @property
    def n_pieces(self) -> int:
        return self._pieces.size(-1)
    
    def pieces(self) -> torch.Tensor:
        """

        Returns:
            torch.Tensor: The pieces making upt hte piecewise transform
        """
        if not self._tunable:
            return self._pieces # * self._diff + self._lower
        
        pieces = torch.cumsum(torch.nn.functional.softplus(self._pieces), dim=-1)
        max_ = pieces.max(dim=-1, keepdim=True)[0]
        min_ = pieces.min(dim=-1, keepdim=True)[0]
        return ((pieces - min_) / (max_ - min_ + 1e-6)) * self._diff + self._lower
    
    def diff(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """helper method to determine the difference between two points in the 
        function

        Args:
            x (torch.Tensor): The input

        Returns:
            typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: pairs of points
        """
        x = x.unsqueeze(-1)
        pieces = self.pieces()
        lower = pieces[:,:,:-1]
        upper = pieces[:,:,1:]
        within = ((x >= lower) & (x <= upper)).type_as(x)
        chosen = within.argmax(dim=-1, keepdim=True)
        result = within * x
        return result.gather(-1, chosen), chosen

    def oob(self, x: torch.Tensor) -> torch.BoolTensor:
        """
        Args:
            x (torch.Tensor): Whether x is out of bounds

        Returns:
            torch.BoolTensor: Whether 
        """
        return ((x < self._lower) | (x > self._upper)).type_as(x)

    def range(self, ind: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            ind (torch.LongTensor): The indices to choose

        Returns:
            torch.Tensor: The lower and upper bound for those indices
        """
        pieces = self.pieces()
        lower = pieces[:,:,:-1]
        upper = pieces[:,:,1:]
        if self._n_features is None:
            lower = lower.repeat(ind.shape[0], ind.shape[1], 1)
            upper = upper.repeat(ind.shape[0], ind.shape[1], 1)
        else:
            lower = lower.repeat(ind.shape[0], 1, 1)
            upper = upper.repeat(ind.shape[0], 1, 1)

        return upper.gather(-1, ind).squeeze(dim=-1), lower.gather(-1, ind).squeeze(dim=-1)


class Piecewise(Transform):

    def __init__(self, x_range: PieceRange, y_range: PieceRange, eps: float=1e-6):
        """Linear piecewise transform. Use to do flexible non-linear transformations 

        Args:
            x_range (PieceRange): The range of x values
            y_range (PieceRange): The range of y values
            eps (float, optional): An error threshold to prevent numerical issues. Defaults to 1e-6.

        Raises:
            ValueError: If the number of pieces don't match for x and y
        """
        super().__init__()
        self.x_range = x_range
        self.y_range = y_range
        if x_range.n_pieces != y_range.n_pieces:
            raise ValueError(
                f'The number of pieces for x_range {x_range.n_pieces} '
                f'is not equal to that for y {y_range.n_pieces}'
            )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the piecewise linear function

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The output
        """
        out_of_bounds = self.x_range.oob(x)
        x_diff, ind = self.x_range.diff(x)
        upper_x, lower_x = self.x_range.range(ind)
        upper, lower = self.y_range.range(ind)
        
        m = (upper - lower) / (upper_x - lower_x + self.eps)
        value = (x_diff.squeeze(-1) - lower_x) * m + lower

        return (value * (1 - out_of_bounds)) + x * out_of_bounds

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Compute the reverse of the piecewise linear function

        Args:
            y (torch.Tensor): The output

        Returns:
            torch.Tensor: The input
        """
        out_of_bounds = self.y_range.oob(y)
        y_diff, ind = self.y_range.diff(y)
        upper_y, lower_y = self.y_range.range(ind)
        upper, lower = self.x_range.range(ind)
        m = (upper - lower) / (upper_y - lower_y + self.eps)
        value = (y_diff.squeeze(-1) - lower_y) * m + lower
        return (value * (1 - out_of_bounds)) + y * out_of_bounds
