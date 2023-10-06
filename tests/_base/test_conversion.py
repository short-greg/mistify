# Test all conversion modules
import torch
from mistify import conversion, membership
from mistify.fuzzy import LogisticBell


class TestImplication(object):

    def test_mean_core_implication(self):

        x = torch.rand(2, 3, 4, 2)
        
        logistic = LogisticBell.from_combined(membership.ShapeParams(x))
        implication = conversion.MeanCoreImplication()
        assert (implication(logistic) == x[:,:,:,0]).all()

    def test_centroid_implication(self):

        x = torch.rand(2, 3, 4, 2)
        
        logistic = LogisticBell.from_combined(membership.ShapeParams(x))
        implication = conversion.CentroidImplication()
        assert (implication(logistic) == x[:,:,:,0]).all()

    def test_area_implication(self):

        x = torch.rand(2, 3, 4, 2)
        
        logistic = LogisticBell.from_combined(membership.ShapeParams(x))
        implication = conversion.AreaImplication()
        assert (implication(logistic).shape == torch.Size([2, 3, 4]))


class TestAccumulator(object):

    def test_max_value_acc(self):

        value_weight = conversion.ValueWeight(
            torch.rand(2, 3, 4),
            torch.rand(2, 3, 4)
        )
        
        max_value_acc = conversion.MaxValueAcc()
        assert (max_value_acc(value_weight) == value_weight.value.max(dim=-1)[0]).all()

    def test_max_acc(self):

        value_weight = conversion.ValueWeight(
            torch.rand(2, 3, 4),
            torch.rand(2, 3, 4)
        )
        
        max_value_acc = conversion.MaxValueAcc()
        assert (max_value_acc(value_weight).shape == torch.Size([2, 3]))

    def test_max_acc(self):

        value_weight = conversion.ValueWeight(
            torch.rand(2, 3, 4),
            torch.rand(2, 3, 4)
        )
        
        max_value_acc = conversion.WeightedAverageAcc()
        assert (
            max_value_acc(value_weight) == 
            (torch.sum(value_weight.value * value_weight.weight, dim=-1) 
             / torch.sum(value_weight.weight, dim=-1))
        ).all()
