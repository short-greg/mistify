import torch
from mistify.fuzzify._shapes import _ramp, ShapeParams


class TestRamp(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        ramp = _ramp.Ramp(
            ShapeParams(p)
        )
        m = ramp.join(x)
        assert m.size() == torch.Size([2, 3, 4])

    def test_min_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        ramp = _ramp.Ramp(
            ShapeParams(p)
        )
        truncated = ramp.min_cores(m)
        assert truncated.size() == torch.Size([2, 3, 4])

    # def test_min_core_returns_tensor_with_correct_size_after_2_truncates(self):

    #     p = torch.rand(3, 4, 2).cumsum(2)
    #     m = torch.rand(2, 3, 4)
    #     m2 = torch.rand(2, 3, 4)
    #     ramp = _ramp.Ramp(ShapeParams(p))
    #     scaled = ramp.min_cores(m, False)
    #     # truncated = ramp.truncate(m)
    #     # truncated = truncated.truncate(m2)
    #     assert scaled.size() == torch.Size([2, 3, 4])
