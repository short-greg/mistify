# 1st party
from abc import ABC, abstractmethod
import typing

# 3rd party
import torch
from torch import nn
import pandas as pd


class ColProcessor(ABC, nn.Module):
    """Define a processor for a column in a dataframe
    """

    def __init__(self, module: nn.Module, columns: typing.Union[str, typing.List[str]]):
        """Create a processor to handle a column

        Args:
            module (nn.Module): The module to process the columns with
            columns (typing.Union[str, typing.List[str]]): The columns to process
        """
        super().__init__()
        self._columns = [columns] if isinstance(columns, str) else columns
        self._module = module

    @abstractmethod
    def prepare(self, table, dtype: torch.dtype=torch.float32, device="cpu") -> torch.Tensor:
        """Prepare the data for processing

        Args:
            table: The table to process
            dtype (torch.dtype, optional): The dtype to change too. Defaults to torch.float32.
            device (str, optional): The device to output to. Defaults to "cpu".

        Returns:
            torch.Tensor: The prepared data
        """
        pass

    def process(self, x: torch.Tensor) -> torch.Tensor:
        """Process the column with the processor

        Args:
            x (torch.Tensor): The input to process

        Returns:
            torch.Tensor: 
        """
        return self._module(x)
    
    def forward(self, table, dtype: torch.dtype, device: typing.Union[str, torch.device]) -> torch.Tensor:
        """Process the column

        Args:
            table : The table to retrieve the column from
            dtype (torch.dtype): The type to convert to
            device (typing.Union[str, torch.device]): The output device

        Returns:
            torch.Tensor: The processed column
        """
        return self.process(self.prepare(table, dtype, device))


class PandasColProcessor(ColProcessor):
    """Defines a processor to convert a set of columns with pandas
    """

    def prepare(self, table: pd.DataFrame, dtype: torch.dtype=torch.float32, device="cpu") -> torch.Tensor:
        """Prepare the data for processing

        Args:
            table (pd.DataFrame): Pandas dataframe
            dtype (torch.dtype, optional): The dtype for the output value. Defaults to torch.float32.
            device (str, optional): The device to output. Defaults to "cpu".

        Raises:
            KeyError: If columns are not passed in

        Returns:
            torch.Tensor: The 'prepared' column
        """
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
        """Pass the table through each column processor to get the output of each column

        Args:
            table : The table to process
            dtype (torch.dtype, optional): The dtype of to use for torch. Defaults to torch.float32.
            device (str, optional): The torch device to use. Defaults to "cpu".

        Returns:
            torch.Tensor: The output tensor
        """
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
