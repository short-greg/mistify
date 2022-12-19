from . import crisp
import torch


class TestCrispSet(object):

    def test_zeros_dim_is_1(self):
        
        zeros = crisp.CrispSet.zeros(4)
        assert zeros.data.dim() == 1
    
    def test_zeros_with_batch_dim_is_2(self):
        
        zeros = crisp.CrispSet.zeros(4, 2)
        assert zeros.data.dim() == 2

    def test_zeros_with_batch_and_variables_dim_is_4(self):
        
        zeros = crisp.CrispSet.zeros(4, 2, (3, 2))
        assert zeros.data.dim() == 4

    def test_ones_with_batch_and_variables_dim_is_4(self):
        
        ones = crisp.CrispSet.ones(4, 2, (3, 2))
        assert ones.data.dim() == 4

    def test_ones_with_batch_and_variables_is_1(self):
        
        ones = crisp.CrispSet.ones(4, 2, (3, 2))
        assert (ones.data == torch.tensor(1.0)).all()

    def test_rand_with_batch_and_variables_dim_is_4(self):
        
        ones = crisp.CrispSet.rand(4, 2, (3, 2))
        assert ones.data.dim() == 4

    def test_intersect_results_in_all_values_being_less_or_same(self):
        torch.manual_seed(1)
        c1 = crisp.CrispSet.rand(4, 2, (3, 2))
        c2 = crisp.CrispSet.rand(4, 2, (3, 2))
        c3 = c1 * c2
        assert (c3.data <= c2.data).all()

    def test_uion_results_in_all_values_being_greater_or_same(self):
        
        torch.manual_seed(1)
        c1 = crisp.CrispSet.rand(4, 2, (3, 2))
        c2 = crisp.CrispSet.rand(4, 2, (3, 2))
        c3 = c1 + c2
        assert (c3.data >= c2.data).all()
    
    def test_ones_with_batch_and_variables_is_1_or_zero(self):
        
        ones = crisp.CrispSet.rand(4, 2, (3, 2))
        assert ((ones.data == torch.tensor(1.0)) | (ones.data == torch.tensor(0.0))).all()


class TestCrispComposition(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = crisp.CrispComposition(2, 4)
        crisp_set = crisp.CrispSet.rand(2, batch_size=4)
        assert composition.forward(crisp_set)
    
    def test_forward_outputs_correct_size_with_complement(self):
        composition = crisp.CrispComposition(2, 4, True)
        crisp_set = crisp.CrispSet.rand(2, batch_size=4)
        assert composition.forward(crisp_set)

    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = crisp.CrispComposition(2, 4, True, in_variables=2)
        crisp_set = crisp.CrispSet.rand(2, batch_size=4, variable_size=(2,))
        assert composition.forward(crisp_set)

    def test_forward_outputs_all_ones_or_zeros(self):
        composition = crisp.CrispComposition(2, 4, True, in_variables=2)
        crisp_set = crisp.CrispSet.rand(2, batch_size=4, variable_size=(2,))
        result = composition.forward(crisp_set)
        assert ((result.data == torch.tensor(1.0)) | (result.data == torch.tensor(0.0))).all()