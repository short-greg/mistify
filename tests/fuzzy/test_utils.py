import torch
from mistify import fuzzy


class TestFuzzySet(object):

    def test_zeros_dim_is_1(self):
        
        zeros = fuzzy.negatives(4)
        assert zeros.dim() == 1
    
    def test_zeros_with_batch_dim_is_2(self):
        
        zeros = fuzzy.negatives(2, 4)
        assert zeros.dim() == 2

    def test_zeros_with_batch_and_variables_dim_is_4(self):
        
        zeros = fuzzy.negatives(2, 3, 2, 4)
        assert zeros.dim() == 4

    def test_ones_with_batch_and_variables_dim_is_4(self):
        
        ones = fuzzy.positives(2, 3, 2, 4)
        assert ones.dim() == 4

    def test_ones_with_batch_and_variables_is_1(self):
        
        ones = fuzzy.positives(2, 3, 2, 4)
        assert (ones == torch.tensor(1.0)).all()

    def test_rand_with_batch_and_variables_dim_is_4(self):
        
        ones = fuzzy.rand(2, 3, 2, 4)
        assert ones.dim() == 4

    def test_intersect_results_in_all_values_being_less_or_same(self):
        torch.manual_seed(1)
        c1 = fuzzy.rand(2, 3, 2, 4)
        c2 = fuzzy.rand(2, 3, 2, 4)
        c3 = fuzzy.intersect(c1, c2)
        assert (c3 <= c2).all()

    def test_intersect_is_included_in_the_tensor(self):
        
        c1 = fuzzy.rand(2, 3, 2, 4)
        c2 = fuzzy.rand(2, 3, 2, 4) * c1
        assert (fuzzy.inclusion(c1, c2) == 1).all()

    def test_union_is_excluded_in_the_tensor(self):
        
        c1 = fuzzy.rand(2, 3, 2, 4)
        c2 = fuzzy.rand(2, 3, 2, 4) + c1
        assert (fuzzy.exclusion(c1, c2).data == 1).all()
    
    def test_differ_is_greater_than_zero_for_all(self):
        
        c1 = fuzzy.rand(2, 3, 2, 4)
        c2 = fuzzy.rand(2, 3, 2, 4)
        assert ((fuzzy.differ(c1, c2)).data >= 0.0).all()

    def test_differ_is_included_in_tensor(self):
        
        c1 = fuzzy.rand(2, 3, 2, 4)
        c2 = fuzzy.rand(2, 3, 2, 4)
        assert (fuzzy.inclusion(c1, fuzzy.differ(c1, c2)).data == 1.0).all()

    def test_transpose_tranposes_dimensions_correctly(self):
        
        c1 = fuzzy.rand(2, 3, 2, 4)
        assert (c1.transpose(1, 2) == c1.transpose(1, 2)).all()

    def test_union_results_in_all_values_being_greater_or_same(self):
        
        torch.manual_seed(1)
        c1 = fuzzy.rand(2, 3, 2, 4)
        c2 = fuzzy.rand(2, 3, 2, 4)
        c3 = fuzzy.unify(c1, c2)
        assert (c3.data >= c2.data).all()
    
    def test_rand_with_batch_and_variables_is_between_one_and_zero(self):
        
        rands = fuzzy.rand(2, 3, 2, 4)
        assert ((rands <= torch.tensor(1.0)) | (rands >= torch.tensor(0.0))).all()
