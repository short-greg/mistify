import torch
from mistify.infer import _noise as noise


class TestDropout:

    def test_dropout_produces_correct_output(self):
        
        torch.manual_seed(1)
        x = torch.randn(8, 2)
        dropout = noise.Dropout(0.25)
        y = dropout(x)
        assert ((y == 0.0) | (y == x)).all()

    def test_dropout_does_not_produce_all_zeros(self):
        
        torch.manual_seed(1)
        x = torch.randn(8, 2)
        dropout = noise.Dropout(0.25)
        y = dropout(x)
        assert ((y != 0.0)).any()

    def test_dropout_does_not_produces_zeros(self):
        
        torch.manual_seed(1)
        x = torch.randn(8, 2)
        dropout = noise.Dropout(0.25)
        y = dropout(x)
        assert ((y == 0.0)).any()

    def test_dropout_outputs_ones(self):
        
        torch.manual_seed(1)
        x = torch.randn(8, 2)
        dropout = noise.Dropout(0.25, 1.0)
        y = dropout(x)
        assert ((y == 1.0)).any()
