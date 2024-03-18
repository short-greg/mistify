import torch
from mistify._functional import _m as F


class TestToSigned:

    def test_to_signed_outputs_neg_one_and_one(self):
        
        x1 = (torch.randn(3, 2) > 0).float()
        signed = F.to_signed(x1)
        assert ((signed == -1) | (signed == 1)).all()

    def test_to_signed_outputs_zero_when_uncertain(self):
        
        x1 = torch.full((3, 2), 0.5)
        signed = F.to_signed(x1)
        assert ((signed == 0)).all()    


class TestToBinary:

    def test_to_binary_outputs_zero_or_one(self):
        
        x1 = torch.randn(3, 2).sign()
        binary = F.to_boolean(x1)
        assert ((binary == 0) | (binary == 1)).all()

    def test_to_binary_outputs_point_five_when_uncertain(self):
        
        x1 = torch.full((3, 2), 0.0)
        binary = F.to_boolean(x1)
        assert (binary == 0.5).all()
