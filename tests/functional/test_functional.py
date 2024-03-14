import torch
from mistify import functional as F

class TestAda:

    def test_adamin_results_in_correct_size(self):
        
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert F.adamin(x1, x2).size() == torch.Size([3, 2, 3])
    
    def test_adamax_on_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 3)
        assert F.adamax_on(x1, dim=-2).size() == torch.Size([3, 3])
    
    def test_adamin_on_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 3)
        assert F.adamin_on(x1, dim=-2).size() == torch.Size([3, 3])
    
    def test_adamax_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert F.adamax(x1, x2).size() == torch.Size([3, 2, 3])


class TestSmooth:

    def test_smoothmin_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert F.smooth_min(x1, x2, 10).size() == torch.Size([3, 2, 3])
    
    def test_smoothmax_on_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 3)
        assert F.smooth_max_on(x1, dim=-2, a=10).size() == torch.Size([3, 3])
    
    def test_smoothmin_on_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 3)
        assert F.smooth_min_on(x1, dim=-2, a=10).size() == torch.Size([3, 3])
    
    def test_smoothmin_on_results_in_correct_size_with_a_of_none(self):
        x1 = torch.rand(3, 2, 3)
        assert F.smooth_min_on(x1, dim=-2, a=None).size() == torch.Size([3, 3])

    def test_smoothmax_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert F.smooth_max(x1, x2, a=10).size() == torch.Size([3, 2, 3])

    def test_smoothmax_results_in_correct_size_with_a_of_none(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert F.smooth_max(x1, x2, a=None).size() == torch.Size([3, 2, 3])

    def test_smoothmin_results_in_correct_size_with_a_of_none(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert F.smooth_min(x1, x2, a=None).size() == torch.Size([3, 2, 3])


class TestToSigned:

    def test_to_signed_outputs_neg_one_and_one(self):
        
        x1 = (torch.randn(3, 2) > 0).float()
        signed = F.to_signed(x1)
        assert ((signed == -1) | (signed == 1)).all()

    def test_to_signed_outputs_zero_when_uncertain(self):
        
        x1 = torch.full((3, 2), 0.5)
        signed = F.to_signed(x1)
        assert ((signed == 0)).all()    


class TestToBinary:

    def test_to_binary_outputs_zero_or_one(self):
        
        x1 = torch.randn(3, 2).sign()
        binary = F.to_binary(x1)
        assert ((binary == 0) | (binary == 1)).all()

    def test_to_binary_outputs_point_five_when_uncertain(self):
        
        x1 = torch.full((3, 2), 0.0)
        binary = F.to_binary(x1)
        assert (binary == 0.5).all()


class TestBoundedMinOn:

    def test_to_bounded_min_outputs_zero_if_below_zero(self):

        x1 = torch.full((3,), 0.5)
        y = F.bounded_min_on(x1)
        assert (y == 0).all()

    def test_to_bounded_min_outputs_zero_if_all_one(self):

        x1 = torch.full((3,), 1.0)
        y = F.bounded_min_on(x1)
        assert (y == 1.0).all()


class TestBoundedMaxOn:

    def test_to_bounded_min_outputs_zero_if_below_zero(self):

        x1 = torch.full((3,), 0.5)
        y = F.bounded_max_on(x1)
        assert (y == 1.0).all()

    def test_to_bounded_min_outputs_zero_if_all_one(self):

        x1 = torch.full((3,), 0.0)
        y = F.bounded_max_on(x1)
        assert (y == 0.0).all()


class TestBoundedMin:

    def test_to_bounded_min_outputs_zero_if_below_zero(self):

        x1 = torch.full((3,), 0.5)
        x2 = torch.full((3,), 0.5)
        y = F.bounded_min(x1, x2)
        assert (y == 0).all()

    def test_bounded_min_outputs_one_if_both_one(self):

        x1 = torch.full((3,), 1.0)
        x2 = torch.full((3,), 1.0)
        y = F.bounded_min(x1, x2)
        assert (y == 1.0).all()


class TestBoundedMax:

    def test_bounded_max_outputs_one_if_point_five(self):

        x1 = torch.full((3,), 0.5)
        x2 = torch.full((3,), 0.5)
        y = F.bounded_max(x1, x2)
        assert (y == 1.0).all()

    def test_bounded_max_outputs_zero_if_both_zero(self):

        x1 = torch.full((3,), 0.0)
        x2 = torch.full((3,), 0.0)
        y = F.bounded_max(x1, x2)
        assert (y == 0.0).all()


class TestProbSum:

    def test_prob_sum_outputs_one_if_both_one(self):

        x1 = torch.full((2,), 1.0)
        x2 = torch.full((2,), 1.0)
        y = F.prob_sum(x1, x2)
        assert (y == 1.0).all()

    def test_prob_sum_outputs_point_seven_five_if_both_point_five(self):

        x1 = torch.full((3,), 0.5)
        x2 = torch.full((3,), 0.5)
        y = F.prob_sum(x1, x2)
        assert (y == 0.75).all()


class TestProdOn:

    def test_prod_on_outputs_the_product(self):

        x1 = torch.full((3,), 0.5)
        y = F.prod_on(x1)
        assert (y == 0.5 ** 3).all()

    def test_prod_on_outputs_one_if_all_one(self):

        x1 = torch.full((3,), 1.0)
        y = F.prod_on(x1)
        assert (y == 1.0).all()
