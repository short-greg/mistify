# # Test all conversion modules
import torch
from mistify.fuzzify import _conversion


class TestConclusion(object):

    def test_max_value_acc(self):

        value_weight = _conversion.HypoWeight(
            torch.rand(2, 3, 4),
            torch.rand(2, 3, 4)
        )
        
        max_value_acc = _conversion.MaxValueConc(4, 3)
        assert (max_value_acc(value_weight) == value_weight.hypo.max(dim=-1)[0]).all()

    def test_max_acc(self):

        value_weight = _conversion.HypoWeight(
            torch.rand(2, 3, 4),
            torch.rand(2, 3, 4)
        )
        
        max_value_acc = _conversion.MaxValueConc(4, 3)
        assert (max_value_acc(value_weight).shape == torch.Size([2, 3]))

    def test_max_acc(self):

        value_weight = _conversion.HypoWeight(
            torch.rand(2, 3, 4),
            torch.rand(2, 3, 4)
        )
        
        max_value_acc = _conversion.WeightedMAverageConc(4, 3)
        assert (
            max_value_acc(value_weight) == 
            (torch.sum(value_weight.hypo * value_weight.weight, dim=-1) 
             / torch.sum(value_weight.weight, dim=-1))
        ).all()

    def test_learned_weight_conc(self):

        value_weight = _conversion.HypoWeight(
            torch.rand(2, 3, 4),
            torch.rand(2, 3, 4)
        )
        
        weighted_conc = _conversion.WeightedPAverageConc(4, 3)
        weight = weighted_conc.layer_weightf(weighted_conc.layer_weight)
        print(
            weight, value_weight.hypo
        )
        assert torch.isclose(
            weighted_conc(value_weight),
            (torch.sum(value_weight.hypo * weight, dim=-1) 
             / torch.sum(weight, dim=-1))
        ).all()

