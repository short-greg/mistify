import pytest

import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from mistify.wrap import _table as _modules


class TestPandasColProcesor(object):

    def test_col_processor_retrieves_columns(self):
        df = pd.DataFrame(
            np.random.randn(4, 3), columns=['a', 'b', 'c']
        )
        linear = nn.Linear(2, 3)
        processor = _modules.PandasColProcessor(
            linear, ['a', 'b']
        )
        result = processor(df, torch.float32, "cpu")
        assert result.shape == torch.Size([4, 3])

    def test_col_processor_raises_error_if_invalid_columns(self):
        df = pd.DataFrame(
            np.random.randn(4, 3), columns=['a', 'b', 'c']
        )
        linear = nn.Linear(2, 3)
        processor = _modules.PandasColProcessor(
            linear, ['a', 'd']
        )
        with pytest.raises(KeyError):
            processor(df, torch.float32, "cpu")
        

class TestTableProcessor(object):

    def test_table_processor_processes_all_columns(self):
        df = pd.DataFrame(
            np.random.randn(4, 3), columns=['a', 'b', 'c']
        )
        linear = nn.Linear(1, 2)
        processor1 = _modules.PandasColProcessor(
            linear, ['a']
        )
        processor2 = _modules.PandasColProcessor(
            linear, ['b']
        )
        table_processor = _modules.TableProcessor(
            [processor1, processor2], False
        )
        result = table_processor(df, torch.float32, "cpu")
        assert result.shape == torch.Size([4, 4])

    def test_table_processor_processes_all_columns_with_padding(self):
        df = pd.DataFrame(
            np.random.randn(4, 3), columns=['a', 'b', 'c']
        )
        linear = nn.Linear(2, 2)
        linear2 = nn.Linear(1, 2)
        processor1 = _modules.PandasColProcessor(
            linear, ['a', 'b']
        )
        processor2 = _modules.PandasColProcessor(
            linear2, ['b']
        )
        table_processor = _modules.TableProcessor(
            [processor1, processor2], True, [0, 0]
        )
        result = table_processor(df, torch.float32, "cpu")
        assert result.shape == torch.Size([4, 4])

    # TODO: need to finish this with processor of different size
