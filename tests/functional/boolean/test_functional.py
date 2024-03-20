import torch
from mistify._functional import boolean


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

    
    def test_rand_with_batch_and_variables_is_between_one_and_zero(self):
        
        rands = boolean.rand(2, 3, 2, 4)
        assert ((rands == torch.tensor(1.0)) | (rands == torch.tensor(0.0))).all()
    
    def test_forget_sets_values_to_point5(self):
        
        torch.manual_seed(1)
        rands = boolean.rand(2, 3, 2, 4)
        assert (boolean.forget(rands, 0.5) == 0.5).any()
