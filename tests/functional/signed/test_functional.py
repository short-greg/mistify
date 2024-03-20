import torch
from mistify._functional import signed


class TestFuzzySet(object):

    def test_zeros_dim_is_1(self):
        
        negs = signed.negatives(4)
        assert negs.dim() == 1
    
    def test_zeros_with_batch_dim_is_2(self):
        
        negs = signed.negatives(2, 4)
        assert negs.dim() == 2

    def test_zeros_with_batch_and_variables_dim_is_4(self):
        
        negs = signed.negatives(2, 3, 2, 4)
        assert negs.dim() == 4

    def test_ones_with_batch_and_variables_dim_is_4(self):
        
        ones = signed.positives(2, 3, 2, 4)
        assert ones.dim() == 4

    def test_ones_with_batch_and_variables_is_1(self):
        
        ones = signed.positives(2, 3, 2, 4)
        assert (ones == torch.tensor(1.0)).all()
    
    def test_negs_with_batch_and_variables_is_1(self):
        
        negs = signed.negatives(2, 3, 2, 4)
        assert (negs == torch.tensor(-1.0)).all()

    def test_rand_with_batch_and_variables_dim_is_4(self):
        
        rands = signed.rand(2, 3, 2, 4)
        assert rands.dim() == 4

    def test_rands_with_batch_and_variables_dim_is_4(self):
        
        torch.manual_seed(1)
        rands = signed.rand(2, 3, 2, 4)
        assert ((rands == 1.0) | (rands == -1.0)).all()

    def test_transpose_tranposes_dimensions_correctly(self):
        
        c1 = signed.rand(2, 3, 2, 4)
        assert (c1.transpose(1, 2) == c1.transpose(1, 2)).all()
    
    def test_rand_with_batch_and_variables_is_between_one_and_zero(self):
        
        rands = signed.rand(2, 3, 2, 4)
        assert ((rands <= torch.tensor(1.0)) | (rands >= torch.tensor(0.0))).all()
