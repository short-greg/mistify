import torch
from mistify.fuzzify._shapes import _step, ShapeParams


class TestStep(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = torch.rand(3, 4, 1)
        x = torch.rand(2, 3)
        step = _step.Step(
            ShapeParams(b)
        )
        m = step.join(x)
        assert m.size() == torch.Size([2, 3, 4])

    def test_min_core_returns_tensor_with_correct_size(self):

        b = torch.rand(3, 4, 1)
        m = torch.rand(2, 3, 4)
        step = _step.Step(
            ShapeParams(b)
        )
        truncated = step.truncate(m)
        assert truncated.min_cores.size() == torch.Size([2, 3, 4])

    def test_min_core_returns_tensor_with_correct_size_with_two_truncates(self):

        b = torch.rand(3, 4, 1)
        s = torch.rand(3, 4, 1)
        m = torch.rand(2, 3, 4)
        m2 = torch.rand(2, 3, 4)
        step = _step.Step(
            ShapeParams(b)
        )
        truncated = step.truncate(m)
        truncated = truncated.truncate(m2)
        assert truncated.min_cores.size() == torch.Size([2, 3, 4])