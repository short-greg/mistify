import torch
from mistify.infer import boolean


class TestBoolean(object):

    def test_zeros_dim_is_1(self):
        
        zeros = boolean.negatives(4)
        assert zeros.dim() == 1
    
    def test_zeros_with_batch_dim_is_2(self):
        
        zeros = boolean.negatives(2, 4)
        assert zeros.dim() == 2

    def test_zeros_with_batch_and_variables_dim_is_4(self):
        
        zeros = boolean.negatives(2, 3, 2, 4)
        assert zeros.dim() == 4

    def test_ones_with_batch_and_variables_dim_is_4(self):
        
        ones = boolean.positives(2, 3, 2, 4)
        assert ones.dim() == 4

    def test_ones_with_batch_and_variables_is_1(self):
        
        ones = boolean.positives(2, 3, 2, 4)
        assert (ones == torch.tensor(1.0)).all()

    def test_rand_with_batch_and_variables_dim_is_4(self):
        
        ones = boolean.rand(2, 3, 2, 4)
        assert ones.dim() == 4

    def test_intersect_results_in_all_values_being_less_or_same(self):
        torch.manual_seed(1)
        c1 = boolean.rand(2, 3, 2, 4)
        c2 = boolean.rand(2, 3, 2, 4)
        c3 = boolean._functional.intersect(c1, c2)
        assert (c3 <= c2).all()

    def test_intersect_is_included_in_the_tensor(self):
        
        c1 = boolean.rand(2, 3, 2, 4)
        c2 = torch.min(boolean.rand(2, 3, 2, 4), c1)
        assert (boolean.inclusion(c2, c1) == 1).all()

    def test_union_is_excluded_in_the_tensor(self):
        
        c1 = boolean.rand(2, 3, 2, 4)
        c2 = torch.max(boolean.rand(2, 3, 2, 4), c1)
        assert (boolean.exclusion(c2, c1).data == 1).all()
    
    def test_differ_is_greater_than_zero_for_all(self):
        
        c1 = boolean.rand(2, 3, 2, 4)
        c2 = boolean.rand(2, 3, 2, 4)
        assert ((boolean.differ(c1, c2)).data >= 0.0).all()

    def test_differ_is_included_in_tensor(self):
        
        c1 = boolean.rand(2, 3, 2, 4)
        c2 = boolean.rand(2, 3, 2, 4)
        c2 = boolean.differ(c1, c2)
        assert (boolean.inclusion(c2, c1).data == 1.0).all()

    def test_union_results_in_all_values_being_greater_or_same(self):
        
        torch.manual_seed(1)
        c1 = boolean.rand(2, 3, 2, 4)
        c2 = boolean.rand(2, 3, 2, 4)
        c3 = boolean.unify(c1, c2)
        assert (c3.data >= c2.data).all()
    
    def test_rand_with_batch_and_variables_is_between_one_and_zero(self):
        
        rands = boolean.rand(2, 3, 2, 4)
        assert ((rands == torch.tensor(1.0)) | (rands == torch.tensor(0.0))).all()
    
    def test_forget_sets_values_to_point5(self):
        
        torch.manual_seed(1)
        rands = boolean.rand(2, 3, 2, 4)
        assert (boolean.forget(rands, 0.5) == 0.5).any()
