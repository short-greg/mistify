import torch
from mistify import _core

class TestAda:

    def test_adamin_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert _core.adamin(x1, x2).size() == torch.Size([3, 2, 3])
    
    def test_adamax_on_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 3)
        assert _core.adamax_on(x1, dim=-2).size() == torch.Size([3, 3])
    
    def test_adamin_on_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 3)
        assert _core.adamin_on(x1, dim=-2).size() == torch.Size([3, 3])
    
    def test_adamax_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert _core.adamax(x1, x2).size() == torch.Size([3, 2, 3])


class TestSmooth:

    def test_smoothmin_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert _core.smooth_min(x1, x2, 10).size() == torch.Size([3, 2, 3])
    
    def test_smoothmax_on_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 3)
        assert _core.smooth_max_on(x1, dim=-2, a=10).size() == torch.Size([3, 3])
    
    def test_smoothmin_on_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 3)
        assert _core.smooth_min_on(x1, dim=-2, a=10).size() == torch.Size([3, 3])
    
    def test_smoothmax_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert _core.smooth_max(x1, x2, a=10).size() == torch.Size([3, 2, 3])
