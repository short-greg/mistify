from mistify import binary
import torch


class TestBinaryComplement:

    def test_binary_complement_outputs_complement(self):

        complement = binary.BinaryComplement()
        x = torch.rand(2, 3).round()
        assert ((1 - x) == complement(x)).all()


class TestCrispComposition(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = binary.BinaryOr(2, 4)
        crisp_set = binary.rand(4, 2)
        assert composition.forward(crisp_set).size() == torch.Size([4, 4])
    

    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = binary.BinaryOr(2, 4, n_terms=2)
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
