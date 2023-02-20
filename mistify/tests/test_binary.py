from .. import binary
import torch


class TestCrispComposition(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = binary.BinaryComposition(2, 4)
        crisp_set = binary.rand(4, 2)
        assert composition.forward(crisp_set).size() == torch.Size([4, 4])
    
    def test_forward_outputs_correct_size_with_complement(self):
        composition = binary.BinaryComposition(2, 4, True)
        crisp_set = binary.rand(4, 2)
        assert composition.forward(crisp_set).size() == torch.Size([4, 4])

    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = binary.BinaryComposition(2, 4, True, in_variables=2)
        crisp_set = binary.rand(4, 2, 2)
        assert composition.forward(crisp_set).size() == torch.Size([4, 2, 4])

    # def test_forward_outputs_all_ones_or_zeros(self):
    #     composition = crisp.CrispComposition(2, 4, True, in_variables=2)
    #     crisp_set = crisp.CrispSet.rand(4, 2, 2)
    #     result = composition.forward(crisp_set)
    #     assert ((result.data == torch.tensor(1.0)) | (result.data == torch.tensor(0.0))).all()

    # def test_forward_outputs_correct_size(self):
    #     composition = crisp.CrispComposition(2, 4, True, in_variables=2)
    #     crisp_set = crisp.CrispSet.rand(4, 2, 2)
    #     result = composition.forward(crisp_set)
    #     assert result.data.size() == torch.Size([4, 2, 4])

# class TestCrispSet(object):

#     def test_zeros_dim_is_1(self):
        
#         zeros = crisp.BinarySet.negatives(4)
#         assert zeros.data.dim() == 1
    
#     def test_zeros_with_batch_dim_is_2(self):
        
#         zeros = crisp.BinarySet.negatives(2, 4)
#         assert zeros.data.dim() == 2

#     def test_zeros_with_batch_and_variables_dim_is_4(self):
        
#         zeros = crisp.BinarySet.negatives(2, 3, 2, 4)
#         assert zeros.data.dim() == 4

#     def test_ones_with_batch_and_variables_dim_is_4(self):
        
#         ones = crisp.BinarySet.positives(2, 3, 2, 4)
#         assert ones.data.dim() == 4

#     def test_ones_with_batch_and_variables_is_1(self):
        
#         ones = crisp.BinarySet.positives(2, 3, 2, 4)
#         assert (ones.data == torch.tensor(1.0)).all()

#     def test_rand_with_batch_and_variables_dim_is_4(self):
        
#         ones = crisp.BinarySet.rand(2, 3, 2, 4)
#         assert ones.data.dim() == 4

#     def test_intersect_results_in_all_values_being_less_or_same(self):
#         torch.manual_seed(1)
#         c1 = crisp.BinarySet.rand(2, 3, 2, 4)
#         c2 = crisp.BinarySet.rand(2, 3, 2, 4)
#         c3 = c1 * c2
#         assert (c3.data <= c2.data).all()

#     def test_intersect_is_included_in_the_tensor(self):
        
#         c1 = crisp.BinarySet.rand(2, 3, 2, 4)
#         c2 = crisp.BinarySet.rand(2, 3, 2, 4) * c1
#         assert (c1.inclusion(c2).data == 1).all()

#     def test_union_is_excluded_in_the_tensor(self):
        
#         c1 = crisp.BinarySet.rand(2, 3, 2, 4)
#         c2 = crisp.BinarySet.rand(2, 3, 2, 4) + c1
#         assert (c1.exclusion(c2).data == 1).all()
    
#     def test_differ_is_greater_than_zero_for_all(self):
        
#         c1 = crisp.BinarySet.rand(2, 3, 2, 4)
#         c2 = crisp.BinarySet.rand(2, 3, 2, 4)
#         assert ((c1 - c2).data >= 0.0).all()

#     def test_differ_is_included_in_tensor(self):
        
#         c1 = crisp.BinarySet.rand(2, 3, 2, 4)
#         c2 = crisp.BinarySet.rand(2, 3, 2, 4)
#         assert (c1.inclusion(c1 - c2).data == 1.0).all()

#     def test_transpose_tranposes_dimensions_correctly(self):
        
#         c1 = crisp.BinarySet.rand(2, 3, 2, 4)
#         assert (c1.transpose(1, 2).data == c1.data.transpose(1, 2)).all()

#     def test_uion_results_in_all_values_being_greater_or_same(self):
        
#         torch.manual_seed(1)
#         c1 = crisp.BinarySet.rand(2, 3, 2, 4)
#         c2 = crisp.BinarySet.rand(2, 3, 2, 4)
#         c3 = c1 + c2
#         assert (c3.data >= c2.data).all()
    
#     def test_rand_with_batch_and_variables_is_1_or_zero(self):
        
#         rands = crisp.BinarySet.rand(2, 3, 2, 4)
#         assert ((rands.data == torch.tensor(1.0)) | (rands.data == torch.tensor(0.0))).all()
