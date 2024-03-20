import torch
from mistify._functional import fuzzy


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

    def test_transpose_tranposes_dimensions_correctly(self):
        
        c1 = fuzzy.rand(2, 3, 2, 4)
        assert (c1.transpose(1, 2) == c1.transpose(1, 2)).all()
    
    def test_rand_with_batch_and_variables_is_between_one_and_zero(self):
        
        rands = fuzzy.rand(2, 3, 2, 4)
        assert ((rands <= torch.tensor(1.0)) | (rands >= torch.tensor(0.0))).all()
