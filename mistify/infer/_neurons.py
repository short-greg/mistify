# 1st party
import typing
from functools import partial

# 3rd party
import torch
import torch.nn as nn

# local
from .. import _functional as functional
from .._functional import G
from ..utils import EnumFactory
from abc import abstractmethod
from ._ops import (
    Union, Inter, InterOn, UnionOn,
    ProbInter, ProbUnion, SmoothInter, SmoothInterOn,
    SmoothUnion, SmoothUnionOn
)
from .._functional._factory import OrF, AndF
from .._base import Constrained


WEIGHT_FACTORY = EnumFactory(
    sigmoid=torch.sigmoid,
    clamp=partial(torch.clamp, min=0, max=1),
    sign=torch.sign,
    boolean=lambda x: torch.clamp(torch.round(x), 0, 1),
    none=lambda x: x
)
WEIGHT_FACTORY[None] = (lambda x: x)


def validate_weight_range(w: torch.Tensor, min_value: float, max_value: float) -> typing.Tuple[int, int]:
    """Calculate the number of weights outside the valid range

    Args:
        w (torch.Tensor): the weights
        min_value (float): Min value for weights
        max_value (float): Max value for weights

    Returns:
        typing.Tuple[int, int]: [lesser count, greater count]
    """
    greater = (w > max_value).float().sum()
    lesser = (w < min_value).float().sum()
    return lesser.item(), greater.item()


def validate_binary_weight(w: torch.Tensor, neg_value: float=0.0, pos_value: float=1.0) -> int:
    """Validate that the weight is either 0 or 1

    Args:
        w (torch.Tensor): The weight to validate
        neg_value (float, optional): The negative value of the binary representation. Defaults to 0.0.
        pos_value (float, optional): The positive value of the binary representation. Defaults to 1.0.

    Returns:
        int: The number of invalid entries in w
    """
    return ((w != neg_value) & (w != pos_value)).float().sum().item()


class LogicalNeuron(nn.Module, Constrained):
    """Neuron to implement a logical function
    """

    @abstractmethod
    def w(self) -> torch.Tensor:
        """Get the weight of the neuron

        Returns:
            torch.Tensor: The weight of the neuron
        """
        pass

    @abstractmethod
    def init_weight(self, f: typing.Callable[[torch.Tensor], torch.Tensor]=None):
        """

        Args:
            f (typing.Callable[[torch.Tensor], torch.Tensor], optional): Initialize the weight. Defaults to None.
        """
        pass

    @property
    @abstractmethod
    def f(self) -> typing.Callable:
        """Get the function the logical neuron wraps

        Returns:
            typing.Callable: The logical function
        """
        pass

    @property
    @abstractmethod
    def inner(self) -> typing.Callable:
        """Get the "inner" function of the LogicalNeruon

        Returns:
            typing.Callable: The inner function
        """
        pass


class Or(LogicalNeuron):
    """An Or neuron implements a T-norm between the weights and the inputs 
    followed by an S-norm for aggregation
    """

    def __init__(
        self, in_features: int, out_features: int, n_terms: int=None, n_members: int=None,
        f: OrF=None, wf: 'WeightF'=None
    ) -> None:
        """Create an Or neuron

        Args:
            in_features (int): The number of features into the neuron
            out_features (int): The number of features out of the neuron.
            n_terms (int, optional): The number of terms for the neuron. Defaults to None.
            n_members (int, optional): The population size
            f (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The function to use for the neuron. Defaults to "min_max".
            wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The weight function to use for the neuron. Defaults to "clamp".
        """
        super().__init__()
        
        shape = [in_features, out_features]
        
        if n_terms is not None:
            shape.insert(0, n_terms)
        if n_members is not None:
            shape.insert(0, n_members)
        pop = n_members is not None

        self.weight_base = nn.parameter.Parameter(torch.ones(*shape))
        self.wf = wf or NullWeightF()
        if isinstance(f, typing.Tuple):
            f = OrF(f[0], f[1], pop=pop)
        self._f = f or OrF('std', 'std', pop=pop)

        self._n_terms = n_terms
        self._in_features = in_features
        self._out_features = out_features
        self.init_weight()

    def w(self) -> torch.Tensor:
        """Compute the weight function

        Returns:
            torch.Tensor: The weight
        """
        return self.wf(self.weight_base)

    def init_weight(self, f: typing.Callable[[torch.Tensor], torch.Tensor]=None):
        """_summary_

        Args:
            f (typing.Callable[[torch.Tensor], torch.Tensor], optional): . Defaults to None.
        """
        if f is None:
            f = torch.rand_like
        self.weight_base.data = f(self.weight_base.data)

    @property
    def f(self) -> typing.Callable:
        """The function

        Returns:
            typing.Callable: The function
        """
        return self._f
    
    def inner(self, x: torch.Tensor) -> typing.Callable:
        """The inner function

        Args:
            x (torch.Tensor): The input

        Returns:
            typing.Callable: The 
        """
        return self._f.inner(x, self.w())

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership to get the output for

        Returns:
            torch.Tensor: The output of the membership
        """
        w = self.w()
        return self._f(m, w)

    def constrain(self, lower: float=0., upper=1.):
        """Constrain the weight between two values
        """
        self.weight.data = torch.clamp(
            self.weight.data, lower, upper
        )


class And(LogicalNeuron):
    """An And neuron implements a S-norm between the weights and the inputs 
    followed by an T-norm for aggregation. 
    """

    def __init__(self, in_features: int, out_features: int, n_terms: int=None, n_members: int=None,
        f: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=None, wf: 'WeightF'=None,
        sub1: bool=False
    ) -> None:
        """Create an Logical neuron

        Args:
            in_features (int): The number of features into the neuron
            out_features (int): The number of features out of the neuron.
            n_terms (int, optional): The number of terms for the neuron. Defaults to None.
            n_members (int, optional): The population size
            f (typing.Union[str, typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional): The function to use for the neuron. Defaults to "min_max".
            wf (typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]], optional): The weight function to use for the neuron. Defaults to "clamp".
        """
        super().__init__()
        
        shape = [in_features, out_features]
        
        if n_terms is not None:
            shape.insert(0, n_terms)
        if n_members is not None:
            shape.insert(0, n_members)

        self.weight_base = nn.parameter.Parameter(torch.ones(*shape))
        pop = n_members is not None
        if isinstance(f, typing.Tuple):
            f = AndF(f[0], f[1], pop=pop)
        self._f = f or AndF('std', 'std', pop=pop)
        self.wf = wf or NullWeightF()
        self.sub1 = sub1

        self._n_terms = n_terms
        self._in_features = in_features
        self._out_features = out_features
        self.init_weight()

    def w(self) -> torch.Tensor:
        """Get the weight after passing through the weigth function

        Returns:
            torch.Tensor: The weight
        """
        w = self.wf(self.weight_base)
        if self.sub1:
            return 1 - w
        return w

    def init_weight(self, f: typing.Callable[[torch.Tensor], torch.Tensor]=None):
        """Initialize the weight

        Args:
            f (typing.Callable[[torch.Tensor], torch.Tensor], optional): Initialize the weight. Defaults to None.
        """
        if f is None:
            f = torch.rand_like
        self.weight_base.data = f(self.weight_base.data)
    
    @property
    def f(self) -> typing.Callable:
        """The logical function used by And

        Returns:
            typing.Callable: The function
        """
        return self._f

    def inner(self, x: torch.Tensor) -> typing.Callable:
        """The inner function for the And neuron

        Args:
            x (torch.Tensor): The input

        Returns:
            typing.Callable: The 
        """
        return self._f.inner(x, self.w())

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Compute the And function using m

        Args:
            m (torch.Tensor): The membership to get the output for

        Returns:
            torch.Tensor: The output of the membership
        """
        w = self.w()
        return self._f(m, w)

    def constrain(self, lower: float=0.0, upper: float=1.0):
        """Constrain the weight between two values
        """
        self.weight.data = torch.clamp(
            self.weight.data, lower, upper
        )


class MaxMin(Or):

    def __init__(
        self, in_features: int, out_features: int, 
        n_terms: int = None, n_members: int=None, wf: 'WeightF'=None,
        g: typing.Union[typing.Tuple[G, G], G]=None
    ) -> None:
        """Create an Or neuron

        Args:
            in_features (int): The in features
            out_features (int): The out features
            n_terms (int, optional): The number of terms. Defaults to None.
            n_members (int, optional): The population size. Defaults to None.
            wf (WeightF, optional): The weight function to use. Defaults to None.
            g (typing.Union[typing.Tuple[G, G], G], optional): The g to use, if tuple, will be inner, outer. Defaults to None.
        """
        if isinstance(g, typing.Tuple):
            inner_g, outer_g = g
        else:
            inner_g, outer_g = g, g
        super().__init__(
            in_features, out_features, 
            n_terms, n_members, 
            (Inter(g=inner_g), UnionOn(dim=-2, g=outer_g)), 
            wf)


class SmoothMaxMin(Or):

    def __init__(
        self, in_features: int, out_features: int, 
        n_terms: int = None, n_members: int=None, wf: 'WeightF'=None,
        a: float=None
    ) -> None:
        """Create a mooth Or neuron

        Args:
            in_features (int): The number of input features
            out_features (int): The number of output features
            n_terms (int, optional): The number of terms. Defaults to None.
            n_members (int, optional): The number of individuals to use. Defaults to None.
            wf (WeightF, optional): The weight function to use. Defaults to None.
            a (float, optional): The degree of hardness (larger values are "harder"). Defaults to None.
        """
        super().__init__(in_features, out_features, n_terms, n_members, (SmoothInter(a=a), SmoothUnionOn(dim=-2, a=a)), wf)
        self._a = a

    @property
    def a(self) -> float:
        """Get the degree of hardness

        Returns:
            float: The degree of hardness
        """
        return self._a
    
    @a.setter
    def a(self, a: float) -> float:
        """Set the degree of hardness

        Args:
            a (float): The hardness (larger values are harder)

        Returns:
            float: the hardness
        """
        self._a = a
        self._f = OrF(
            SmoothInter(a=a), SmoothUnionOn(dim=-2, a=a)
        )
        return a


class SmoothMinMax(And):
    """A smooth And function that uses the softmin and softmax functions
    """

    def __init__(
        self, in_features: int, out_features: int, 
        n_terms: int = None, n_members: int=None, wf: 'WeightF'=None,
        a: float=None, sub1: bool=False
    ) -> None:
        """Create a smooth And neuron

        Args:
            in_features (int): The number of input features
            out_features (int): The number of output features
            n_terms (int, optional): The number of terms. Defaults to None.
            n_members (int, optional): The number of individuals to use. Defaults to None.
            wf (WeightF, optional): The weight function to use. Defaults to None.
            a (float, optional): The degree of hardness (larger values are "harder"). Defaults to None.
        """
        super().__init__(
            in_features, out_features, 
            n_terms, n_members, (SmoothUnion(a=a), SmoothInterOn(dim=-2, a=a)), 
            wf, sub1
        )
        self._a = a

    @property
    def a(self) -> float:
        """Get the degree of hardness

        Returns:
            float: The degree of hardness
        """
        return self._a
    
    @a.setter
    def a(self, a: float) -> float:
        """Set the degree of hardness

        Args:
            a (float): The hardness (larger values are harder)

        Returns:
            float: the hardness
        """
        self._a = a
        self._f = AndF(
            SmoothUnion(a=a), SmoothInterOn(dim=-2, a=a)
        )
        return a

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Compute the And using m

        Args:
            m (torch.Tensor): The membership to get the output for

        Returns:
            torch.Tensor: The output of the membership
        """
        w = self.w()
        return self._f(m, w)


class MaxProd(Or):
    """An Or Neuron that uses the product for the inner function and max for the aggregation
    """

    def __init__(
        self, in_features: int, out_features: int, 
        n_terms: int = None, n_members: int=None, wf: 'WeightF'=None,
        g: G=None
    ) -> None:
        """Create a MaxProd neruon

        Args:
            in_features (int): The number of input features
            out_features (int): The number of output features
            n_terms (int, optional): The number of terms. Defaults to None.
            n_members (int, optional): The number of individuals to use. Defaults to None.
            wf (WeightF, optional): The weight function to use. Defaults to None.
            g (G, optional): The straight-through-estimator to use. Defaults to None.
        """
        super().__init__(
            in_features, out_features, n_terms, 
            n_members, (ProbInter(), UnionOn(dim=-2, g=g)), wf
        )


class MinMax(And):
    """An And neuron that uses max for the inner function and min for the aggregate function
    """

    def __init__(
        self, in_features: int, out_features: int, 
        n_terms: int = None, n_members: int=None, wf: 'WeightF'=None,
        g: typing.Union[typing.Tuple[G, G], G]=None, sub1: bool=False
    ) -> None:
        """Create an And neuron

        Args:
            in_features (int): The in features
            out_features (int): The out features
            n_terms (int, optional): The number of terms. Defaults to None.
            n_members (int, optional): The population size. Defaults to None.
            wf (WeightF, optional): The weight function to use. Defaults to None.
            g (typing.Union[typing.Tuple[G, G], G], optional): The g to use, if tuple, will be inner, outer. Defaults to None.
            sub1 (bool, optional): Whether to subtract by one. Defaults to False.
        """
        if isinstance(g, typing.Tuple):
            inner_g, outer_g = g
        else:
            inner_g, outer_g = g, g
        super().__init__(
            in_features, out_features, n_terms, n_members, 
            (Union(g=inner_g), InterOn(dim=-2, g=outer_g)), wf, sub1=sub1
        )


class MinSum(And):
    """And neuron that uses the probabilistic sum for the inner function and 
    """
    def __init__(
        self, in_features: int, out_features: int, 
        n_terms: int = None, n_members: int=None, wf: 'WeightF'=None,
        g: G=None, sub1: bool=False
    ) -> None:
        """Create an And neuron using min and probabilistic sum

        Args:
            in_features (int): The input features
            out_features (int): The number of output features
            n_terms (int, optional): The number of terms. Defaults to None.
            n_members (int, optional): The population size. Defaults to None.
            wf (WeightF, optional): The weight function. Defaults to None.
            g (G, optional): The straight through estimator to use. Defaults to None.
            sub1 (bool, optional): Whether to subtract the weight by one. Defaults to False.
        """
        super().__init__(
            in_features, out_features, n_terms, n_members, 
            (ProbUnion(), InterOn(dim=-2, g=g)), wf, sub1
        )


class WeightF(nn.Module):
    """Calculate the weight function
    """

    @abstractmethod
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        pass


class ClampWeightF(WeightF):
    """Weight function that clamps the weight between two values
    """

    def __init__(self, min_val: float=0.0, max_val: float=1.0, g: G=None):
        """Create a clamp weight function

        Args:
            min_val (float, optional): The min value for the clamp. Defaults to 0.0.
            max_val (float, optional): The max value for the clamp. Defaults to 1.0.
            g (G, optional): The straight through estimator. Defaults to None.
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.g = g

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Clamp the weight between two values

        Args:
            w (torch.Tensor): The weight

        Returns:
            torch.Tensor: The clamped weight
        """
        return functional.clamp(w, self.min_val, self.max_val, self.g)


class Sub1WeightF(WeightF):
    """Take the complement of the weights
    """

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Compute the complement of the weight

        Args:
            w (torch.Tensor): The weight to update

        Returns:
            torch.Tensor: The weight
        """
        return (1 - w)


class BooleanWeightF(WeightF):
    """A weight function that maps the weight to either 0 or 1
    """

    def __init__(self, g: G=None):
        """The boolean weight function
        Args:
            g (G, optional): The straight-through-estimator to use. Defaults to None.
        """
        super().__init__()
        self.g = g

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Compute the binarized weight

        Args:
            w (torch.Tensor): The weight function

        Returns:
            torch.Tensor: The weight
        """
        return functional.heaviside(w, self.g)


class SignWeightF(WeightF):
    """A weight function that maps the weight to either -1 or 1
    """

    def __init__(self, g: G=None):
        """The sign weight function
        Args:
            g (G, optional): The straight-through-estimator to use. Defaults to None.
        """
        super().__init__()
        self.g = g

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Compute the signed weight

        Args:
            w (torch.Tensor): The weight function

        Returns:
            torch.Tensor: The weight
        """
        return functional.sign(w, self.g)


class NullWeightF(WeightF):
    """A null weight function that does nothing
    """

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Does nothing as it is a null weight

        Args:
            w (torch.Tensor): the input weight

        Returns:
            torch.Tensor: the weight
        """
        return w


class SigmoidWeightF(nn.Module):
    """A sigmoid weight function that does nothing
    """

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Calculate the weight by taking the sigmoid

        Args:
            w (torch.Tensor): The weight

        Returns:
            torch.Tensor: The weight after sigmoiding
        """
        return torch.sigmoid(w)
