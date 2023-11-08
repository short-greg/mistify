# # Test all conversion modules
import torch
from mistify.membership import _conversion


class TestAccumulator(object):

    def test_max_value_acc(self):

        value_weight = _conversion.ValueWeight(
            torch.rand(2, 3, 4),
            torch.rand(2, 3, 4)
        )
        
        max_value_acc = _conversion.MaxValueAcc()
        assert (max_value_acc(value_weight) == value_weight.value.max(dim=-1)[0]).all()

    def test_max_acc(self):

        value_weight = _conversion.ValueWeight(
            torch.rand(2, 3, 4),
            torch.rand(2, 3, 4)
        )
        
        max_value_acc = _conversion.MaxValueAcc()
        assert (max_value_acc(value_weight).shape == torch.Size([2, 3]))

    def test_max_acc(self):

        value_weight = _conversion.ValueWeight(
            torch.rand(2, 3, 4),
            torch.rand(2, 3, 4)
        )
        
        max_value_acc = _conversion.WeightedAverageAcc()
        assert (
            max_value_acc(value_weight) == 
            (torch.sum(value_weight.value * value_weight.weight, dim=-1) 
             / torch.sum(value_weight.weight, dim=-1))
        ).all()
