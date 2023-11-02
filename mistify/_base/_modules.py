# 1st party
from abc import ABC, abstractmethod
import typing

# 3rd party
import torch
from torch import nn
import pandas as pd


class Argmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        return torch.argmax(x, dim=-1)


class Sign(nn.Module):

    def __init__(self, grad: bool = True):
        super().__init__()
        self._grad = grad

    def forward(self, x: torch.Tensor):
        if self._grad:
            return SignSTE.apply(x)
        return torch.sign(x)


class Binary(nn.Module):

    def __init__(self, grad: bool = True):
        super().__init__()
        self._grad = grad

    def forward(self, x: torch.Tensor):
        if self._grad:
            return BinarySTE.apply(x)
        return torch.clamp(x, 0, 1).round()


class SignSTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of the Binary Step function.
        """
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # return grad_input.clamp(-1, 1)
        grad_input[(x < -1) | (x > 1)] = 0
        return grad_input


class BinarySTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of the Binary Step function.
        """
        ctx.save_for_backward(x)
        return torch.clamp(x, 0, 1).round()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # return grad_input.clamp(-1, 1)
        grad_input[(x < -1) | (x > 1)] = 0
        return grad_input


class Clamp(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, lower: float=0, upper: float=1, backward_lower: float=-1, backward_upper: float=1):
        """
        Forward pass of the Binary Step function.
        """
        ctx.save_for_backward(x)
        ctx.lower = lower
        ctx.upper = upper
        ctx.backward_lower = backward_lower
        ctx.backward_upper = backward_upper
        return torch.clamp(x, lower, upper)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        grad_input = grad_output.clone()
        return grad_input.clamp(ctx.backward_lower, ctx.backward_upper), None, None, None, None


def clamp(x: torch.Tensor) -> torch.Tensor:
    return Clamp.apply(x)


def binary_ste(x: torch.Tensor) -> torch.Tensor:
    return BinarySTE.apply(x)


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)


class ColProcessor(ABC, nn.Module):

    def __init__(self, module: nn.Module, columns: typing.Union[str, typing.List[str]]):
        super().__init__()
        self._columns = [columns] if isinstance(columns, str) else columns
        self._module = module

    @abstractmethod
    def prepare(self, table, dtype: torch.dtype=torch.float32, device="cpu") -> torch.Tensor:
        pass

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return self._module(x)
    
    def forward(self, table, dtype: torch.dtype, device: typing.Union[str, torch.device]) -> torch.Tensor:
        return self.process(self.prepare(table, dtype, device))


class PandasColProcessor(ColProcessor):
    """Defines a processor to convert a set of columns with pandas
    """

    def prepare(self, table: pd.DataFrame, dtype: torch.dtype=torch.float32, device="cpu") -> torch.Tensor:
        
        try:
            return torch.tensor(
                table[self._columns].values,
                dtype=dtype, device=device
            )
        except KeyError as e:
            raise KeyError('PandasColProcessor requires columns be available in table passed in') from e


class TableProcessor(nn.Module):
    """Defines a processor to convert a table to terms or categories
    """

    def __init__(self, column_processors: ColProcessor, flatten: bool=False, paddings: typing.List=None, out_dims=2, cat_dim=1):
        """Create a TableProcessor to use in  

        Args:
            column_processors (ColumnProcessor): The list of processors
            flatten (bool, optional): Whehter to flatten the processed. Defaults to False.
        """

        super().__init__()
        self._column_processors = column_processors
        self._flatten = flatten
        self._paddings = paddings
        self._out_dims = out_dims
        paddings = paddings or [0] * len(self._column_processors)
        if len(paddings) != len(column_processors):
            raise ValueError('Length of column processors must be the same as for paddings')
        paddings_full = []
        for padding in paddings:
            p = [0] * out_dims
            p[cat_dim] = padding
            paddings_full.append(p)

        self._paddings = paddings_full
        self._cat_dim = cat_dim

    def forward(self, table, dtype: torch.dtype=torch.float32, device="cpu") -> torch.Tensor:

        y = [
            column_processor(table, dtype, device) for column_processor in self._column_processors
        ]
        if self._flatten:
            return torch.cat([
                y_i.view(y_i.size(0), -1) for y_i in y
            ], dim=1)
        
        return torch.cat([
            torch.nn.functional.pad(y_i, padding_i, 'constant', 0) for y_i, padding_i in zip(y, self._paddings)
        ], dim=self._cat_dim)
