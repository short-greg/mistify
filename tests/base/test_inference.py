from mistify._base import inference
import torch.nn as nn
import torch


class TestCat():

    def test_cat_concatenates_on_correct_dimension_when_dimension_is_neg1(self):

        linear = nn.Linear(2, 2)
        cat = inference.CatOp(linear)
        x = torch.rand(2, 2)
        result = cat(x)
        assert (result[:,:2] == x).all()
        assert (result[:,2:] == linear(x)).all()

    def test_cat_concatenates_on_correct_dimension_when_dimension_is_0(self):

        linear = nn.Linear(2, 2)
        cat = inference.CatOp(linear, dim=0)
        x = torch.rand(2, 2)
        result = cat(x)
        assert (result[:2] == x).all()
        assert (result[2:] == linear(x)).all()
