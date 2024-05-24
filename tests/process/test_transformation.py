from mistify.process import _transformation as processors
import torch
import pytest
from itertools import chain

class TestPiecewise:

    def test_creates_parameters_of_the_correct_size(self):

        torch.manual_seed(1)
        x = torch.rand(8, 4)
        x_range = processors.PieceRange.linspace(4)
        y_range = processors.PieceRange.linspace(4, lower=-0.1, upper=1.1)
        piecwise = processors.Piecewise(x_range, y_range)
        
        y = piecwise(x)
        assert y.shape == x.shape

    def test_creates_parameters_of_the_correct_size_with_reverse(self):

        torch.manual_seed(1)
        x = torch.rand(8, 4)
        x_range = processors.PieceRange.linspace(4)
        y_range = processors.PieceRange.linspace(4, lower=-0.1, upper=1.1)
        piecwise = processors.Piecewise(x_range, y_range)
        
        y = piecwise.reverse(x)
        
        assert y.shape == x.shape
        assert (y != x).any()
    
    def test_reverse_reconstructs_x(self):

        torch.manual_seed(1)
        x = torch.rand(8, 4)
        x_range = processors.PieceRange.linspace(4)
        y_range = processors.PieceRange.linspace(4, lower=-0.1, upper=1.1)
        piecwise = processors.Piecewise(x_range, y_range)
        
        y = piecwise(x)
        x_prime = piecwise.reverse(y)
        assert (torch.isclose(x_prime, x, 1e-4)).all()
    
    def test_piecwise_outputs_in_correct_range(self):

        torch.manual_seed(1)
        x = torch.rand(8, 4)
        x_range = processors.PieceRange.linspace(4)
        y_range = processors.PieceRange.linspace(4, lower=1.0, upper=2.0)
        piecwise = processors.Piecewise(x_range, y_range)
        
        y = piecwise(x)
        assert ((y >= 1.0) & (y <= 2.0)).all()

    def test_tunable_updates_parameters(self):

        torch.manual_seed(1)
        x = torch.rand(8, 4)
        x_range = processors.PieceRange.linspace(4)
        y_range = processors.PieceRange.linspace(4, lower=-0.1, upper=1.1, tunable=True)
        piecwise = processors.Piecewise(x_range, y_range)
        
        y = piecwise(x)
        t = torch.rand_like(y)
        optim = torch.optim.Adam(chain(y_range.parameters()), 1e-3)
        optim.zero_grad()
        y_before = torch.nn.utils.parameters_to_vector(y_range.parameters())
        (y - t).pow(2).mean().backward()
        optim.step()

        y_after = torch.nn.utils.parameters_to_vector(y_range.parameters())
        assert ((y_before != y_after).any())


    def test_linspace_creates_piecewise(self):

        piecwise = processors.Piecewise.linspace(4, lower_y=1.0, upper_y=2.0)
        assert isinstance(piecwise, processors.Piecewise)

    def test_not_tunable_returns_no_parameters(self):

        torch.manual_seed(1)
        x = torch.rand(8, 4)
        x_range = processors.PieceRange.linspace(4)
        
        with pytest.raises(NotImplementedError):
            torch.nn.utils.parameters_to_vector(x_range.parameters())

    def test_tunable_keeps_params_in_order(self):

        torch.manual_seed(1)
        x = torch.rand(8, 4)
        x_range = processors.PieceRange.linspace(4)
        y_range = processors.PieceRange.linspace(4, lower=-0.1, upper=1.1, tunable=True)
        piecwise = processors.Piecewise(x_range, y_range)
        
        optim = torch.optim.Adam(chain(y_range.parameters()), 1e0)
        optim.zero_grad()
        y = piecwise(x)
        t = torch.rand_like(y)
        (y - t).pow(2).mean().backward()
        optim.step()
        optim.zero_grad()
        y = piecwise(x)
        (y - t).pow(2).mean().backward()
        optim.step()
        optim.zero_grad()
        y = piecwise(x)
        (y - t).pow(2).mean().backward()
        optim.step()

        pieces = y_range.pieces()
        assert (pieces[:,:,:-1] < pieces[:,:,1:]).all()
