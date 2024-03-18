import torch
from mistify._functional import _join as F


class TestBoundedMinOn:

    def test_to_bounded_min_outputs_zero_if_below_zero(self):

        x1 = torch.full((3,), 0.5)
        y = F.bounded_inter_on(x1)
        assert (y == 0).all()

    def test_to_bounded_min_outputs_zero_if_all_one(self):

        x1 = torch.full((3,), 1.0)
        y = F.bounded_inter_on(x1)
        assert (y == 1.0).all()

    def test_to_bounded_inter_outputs_has_grad_for_all(self):

        x1 = torch.rand(3, 3, requires_grad=True)
        x1.retain_grad()

        y = F.bounded_inter_on(x1, g=True, clip=0.1)
        y.sum().backward()

        assert ((x1.grad > 0.0) | (x1.grad < 0.0)).all()


class TestBoundedMaxOn:

    def test_to_bounded_min_outputs_zero_if_below_zero(self):

        x1 = torch.full((3,), 0.5)
        y = F.bounded_union_on(x1)
        assert (y == 1.0).all()

    def test_to_bounded_min_outputs_zero_if_all_one(self):

        x1 = torch.full((3,), 0.0)
        y = F.bounded_union_on(x1)
        assert (y == 0.0).all()

    def test_to_bounded_union_outputs_has_grad_for_all(self):

        x1 = torch.rand(3, 3, requires_grad=True)
        x1.retain_grad()

        y = F.bounded_union_on(x1, g=True, clip=0.1)
        y.sum().backward()

        assert ((x1.grad > 0.0) | (x1.grad < 0.0)).all()


class TestBoundedMin:

    def test_to_bounded_min_outputs_zero_if_below_zero(self):

        x1 = torch.full((3,), 0.5)
        x2 = torch.full((3,), 0.5)
        y = F.bounded_inter(x1, x2)
        assert (y == 0).all()

    def test_bounded_min_outputs_one_if_both_one(self):

        x1 = torch.full((3,), 1.0)
        x2 = torch.full((3,), 1.0)
        y = F.bounded_inter(x1, x2)
        assert (y == 1.0).all()


class TestBoundedMax:

    def test_bounded_max_outputs_one_if_point_five(self):

        x1 = torch.full((3,), 0.5)
        x2 = torch.full((3,), 0.5)
        y = F.bounded_union(x1, x2)
        assert (y == 1.0).all()

    def test_bounded_max_outputs_zero_if_both_zero(self):

        x1 = torch.full((3,), 0.0)
        x2 = torch.full((3,), 0.0)
        y = F.bounded_union(x1, x2)
        assert (y == 0.0).all()


class TestProbUnion:

    def test_prob_sum_outputs_one_if_both_one(self):

        x1 = torch.full((2,), 1.0)
        x2 = torch.full((2,), 1.0)
        y = F.prob_union(x1, x2)
        assert (y == 1.0).all()

    def test_prob_sum_outputs_point_seven_five_if_both_point_five(self):

        x1 = torch.full((3,), 0.5)
        x2 = torch.full((3,), 0.5)
        y = F.prob_union(x1, x2)
        assert (y == 0.75).all()

    def test_prob_sum_outputs_one_if_both_one(self):
        torch.manual_seed(1)
        x1 = torch.tensor([1.0, 1.0, 1.0])
        y = F.prob_union_on(x1, dim=-1)
        assert (y == 1.0).all()

    def test_prob_sum_outputs_corect_value(self):
        torch.manual_seed(1)
        x1 = torch.tensor([0.5, 0.5, 0.5])
        y = F.prob_union_on(x1, dim=-1)
        assert (y == 0.875).all()


class TestProd:

    def test_prod_on_outputs_the_product(self):

        x1 = torch.full((3,), 0.5)
        y = F.prob_inter_on(x1)
        assert (y == 0.5 ** 3).all()

    def test_prod_on_outputs_one_if_all_one(self):

        x1 = torch.full((3,), 1.0)
        y = F.prob_inter_on(x1)
        assert (y == 1.0).all()

    def test_prod_on_outputs_zero_if_one_is_zero(self):

        x1 = torch.tensor([0.0, 1.0, 0.5])
        y = F.prob_inter_on(x1)
        assert (y == 0.0).all()

    def test_prod_on_outputs_keeps_dim(self):

        x1 = torch.tensor([0.0, 1.0, 0.5])
        y = F.prob_inter_on(x1, keepdim=True)
        assert y.shape == torch.Size([1])


class TestAda:

    def test_adamin_results_in_correct_size(self):
        
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert F.ada_inter(x1, x2).size() == torch.Size([3, 2, 3])
    
    def test_adamax_on_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 3)
        assert F.ada_union_on(x1, dim=-2).size() == torch.Size([3, 3])
    
    def test_adamin_on_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 3)
        assert F.ada_inter_on(x1, dim=-2).size() == torch.Size([3, 3])
    
    def test_adamax_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert F.ada_union(x1, x2).size() == torch.Size([3, 2, 3])


class TestSmooth:

    def test_smoothmin_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert F.smooth_inter(x1, x2, 10).size() == torch.Size([3, 2, 3])
    
    def test_smoothmax_on_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 3)
        assert F.smooth_union_on(x1, dim=-2, a=10).size() == torch.Size([3, 3])
    
    def test_smoothmin_on_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 3)
        assert F.smooth_inter_on(x1, dim=-2, a=10).size() == torch.Size([3, 3])
    
    def test_smoothmin_on_results_in_correct_size_with_a_of_none(self):
        x1 = torch.rand(3, 2, 3)
        assert F.smooth_inter_on(x1, dim=-2, a=None).size() == torch.Size([3, 3])

    def test_smoothmax_results_in_correct_size(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert F.smooth_union(x1, x2, a=10).size() == torch.Size([3, 2, 3])

    def test_smoothmax_results_in_correct_size_with_a_of_none(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert F.smooth_union(x1, x2, a=None).size() == torch.Size([3, 2, 3])

    def test_smoothmin_results_in_correct_size_with_a_of_none(self):
        x1 = torch.rand(3, 2, 1)
        x2 = torch.rand(1, 2, 3)
        assert F.smooth_inter(x1, x2, a=None).size() == torch.Size([3, 2, 3])


class TestUnion:

    def test_union_on_outputs_point_five_if_max_is_point_five(self):

        x1 = torch.tensor([0.4, 0.45, 0.5])
        y = F.union_on(x1)
        assert (y == 0.5).all()

    def test_to_union_on_outputs_one_if_max_is_one(self):

        x1 = torch.tensor([0.4, 1.0, 0.5])
        y = F.union_on(x1)
        assert (y == 1.0).all()

    def test_union_on_outputs_has_grad(self):

        x1 = torch.rand(3, 3, requires_grad=True)
        x1.retain_grad()

        y = F.union_on(x1, g=True)[0]
        y.sum().backward()

        assert ((x1.grad != 0.0)).any()

    def test_union_on_retrieves_idx(self):

        x1 = torch.rand(3, 3, requires_grad=True)
        x1.retain_grad()

        y = F.union_on(x1, g=True, idx=True)
        assert y[1].dtype == torch.int64

    def test_to_union_on_outputs_one_if_max_is_one(self):

        x1 = torch.tensor([0.4, 1.0, 1.0], requires_grad=True)
        x2 = torch.tensor([1.0, 1.0, 0.5], requires_grad=True)
        x1.retain_grad()
        x2.retain_grad()
        y = F.union(x1, x2)
        assert (y == 1.0).all()

    def test_union_on_outputs_has_grad(self):

        x1 = torch.tensor([0.4, 1.0, 1.0], requires_grad=True)
        x2 = torch.tensor([1.0, 1.0, 0.5], requires_grad=True)
        x1.retain_grad()
        x2.retain_grad()
        y = F.union(x1, x2)
        y.sum().backward()
        assert ((x1.grad != 0.0)).any()
        assert ((x2.grad != 0.0)).any()


class TestInter:

    def test_inter_outputs_point_five_if_min_is_point_five(self):

        x1 = torch.tensor([0.6, 0.55, 0.5])
        y = F.inter_on(x1)
        assert (y == 0.5).all()

    def test_to_inter_on_outputs_zero_if_min_is_zero(self):

        x1 = torch.tensor([0.4, 0.0, 0.5])
        y = F.inter_on(x1)
        assert (y == 0.0).all()

    def test_inter_on_outputs_has_grad(self):

        x1 = torch.rand(3, 3, requires_grad=True)
        x1.retain_grad()

        y = F.inter_on(x1, g=True)
        y.sum().backward()

        assert ((x1.grad != 0.0)).any()

    def test_inter_on_retrieves_idx(self):

        x1 = torch.rand(3, 3, requires_grad=True)
        x1.retain_grad()

        y = F.inter_on(x1, g=True, idx=True)
        assert y[1].dtype == torch.int64

    def test_to_inter_on_outputs_point_four_if_min_is_one(self):

        x1 = torch.tensor([0.4, 1.0, 1.0], requires_grad=True)
        x2 = torch.tensor([0.5, 0.4, 0.4], requires_grad=True)
        x1.retain_grad()
        x2.retain_grad()
        y = F.inter(x1, x2)
        assert (y == 0.4).all()

    def test_inter_on_outputs_has_grad(self):

        x1 = torch.tensor([0.4, 1.0, 1.0], requires_grad=True)
        x2 = torch.tensor([1.0, 1.0, 0.5], requires_grad=True)
        x1.retain_grad()
        x2.retain_grad()
        y = F.inter(x1, x2)
        y.sum().backward()
        assert ((x1.grad != 0.0)).any()
        assert ((x2.grad != 0.0)).any()
