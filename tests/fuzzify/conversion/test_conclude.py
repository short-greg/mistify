# # Test all conversion modules
import torch
from mistify.fuzzify import _conversion


class TestConclusion(object):

    def test_max_value_acc(self):

        value_weight = _conversion.HypoM(
            torch.rand(2, 3, 4),
            torch.rand(2, 3, 4)
        )
        
        max_value_acc = _conversion.MaxValueConc()
        assert (max_value_acc(value_weight) == value_weight.hypo.max(dim=-1)[0]).all()

    def test_max_acc(self):

        value_weight = _conversion.HypoM(
            torch.rand(2, 3, 4),
            torch.rand(2, 3, 4)
        )
        
        max_value_acc = _conversion.MaxValueConc()
        assert (max_value_acc(value_weight).shape == torch.Size([2, 3]))

    def test_max_acc(self):

        value_weight = _conversion.HypoM(
            torch.rand(2, 3, 4),
            torch.rand(2, 3, 4)
        )
        
        max_value_acc = _conversion.WeightedAverageConc()
        assert (
            max_value_acc(value_weight) == 
            (torch.sum(value_weight.hypo * value_weight.m, dim=-1) 
             / torch.sum(value_weight.m, dim=-1))
        ).all()
