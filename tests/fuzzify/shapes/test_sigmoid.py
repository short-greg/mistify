import torch
from mistify.fuzzify._shapes import _sigmoid, ShapeParams


class TestSigmoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = torch.rand(3, 4, 1)
        s = torch.rand(3, 4, 1)
        x = torch.rand(2, 3)
        sigmoid = _sigmoid.Sigmoid(
            ShapeParams(b), ShapeParams(s)
        )
        m = sigmoid.join(x)
        assert m.size() == torch.Size([2, 3, 4])

    def test_min_core_returns_tensor_with_correct_size(self):

        b = torch.rand(3, 4, 1)
        s = torch.rand(3, 4, 1)
        m = torch.rand(2, 3, 4)
        sigmoid = _sigmoid.Sigmoid(
            ShapeParams(b), ShapeParams(s)
        )
        truncated = sigmoid.truncate(m)
        assert truncated.min_cores.size() == torch.Size([2, 3, 4])

    def test_min_core_returns_tensor_with_correct_size_with_two_truncates(self):

        b = torch.rand(3, 4, 1)
        s = torch.rand(3, 4, 1)
        m = torch.rand(2, 3, 4)
        m2 = torch.rand(2, 3, 4)
        sigmoid = _sigmoid.Sigmoid(
            ShapeParams(b), ShapeParams(s)
        )
        truncated = sigmoid.truncate(m)
        truncated = truncated.truncate(m2)
        assert truncated.min_cores.size() == torch.Size([2, 3, 4])
