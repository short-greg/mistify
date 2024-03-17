import torch
from mistify.functional import _join as F


class TestBoundedMinOn:

    def test_to_bounded_min_outputs_zero_if_below_zero(self):

        x1 = torch.full((3,), 0.5)
        y = F.bounded_inter_on(x1)
        assert (y == 0).all()

    def test_to_bounded_min_outputs_zero_if_all_one(self):

        x1 = torch.full((3,), 1.0)
        y = F.bounded_inter_on(x1)
        assert (y == 1.0).all()


class TestBoundedMaxOn:

    def test_to_bounded_min_outputs_zero_if_below_zero(self):

        x1 = torch.full((3,), 0.5)
        y = F.bounded_union_on(x1)
        assert (y == 1.0).all()

    def test_to_bounded_min_outputs_zero_if_all_one(self):

        x1 = torch.full((3,), 0.0)
        y = F.bounded_union_on(x1)
        assert (y == 0.0).all()


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


class TestProbSum:

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


class TestProdOn:

    def test_prod_on_outputs_the_product(self):

        x1 = torch.full((3,), 0.5)
        y = F.prob_inter_on(x1)
        assert (y == 0.5 ** 3).all()

    def test_prod_on_outputs_one_if_all_one(self):

        x1 = torch.full((3,), 1.0)
        y = F.prob_inter_on(x1)
        assert (y == 1.0).all()
