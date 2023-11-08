import torch

from mistify.membership._shapes._utils import calc_m_linear_increasing, calc_m_linear_decreasing


class TestCalculations:

    def test_calc_m_linear_increasing_is_equal_to_m_for_all(self):

        x = torch.ones(2, 2, 3)
        pt1 = torch.zeros(2, 2, 3)
        pt2 = torch.ones(2,2, 3)
        m = torch.ones(2, 2, 3) * 0.5

        assert (calc_m_linear_increasing(x, pt1, pt2, m) == m).all()

    def test_calc_m_linear_increasing_is_half_m_for_all(self):

        x = torch.ones(2, 2, 3)
        pt1 = torch.zeros(2, 2, 3)
        pt2 = torch.ones(2,2, 3) * 2
        m = torch.ones(2, 2, 3) * 0.5

        assert (calc_m_linear_increasing(x, pt1, pt2, m) == (m / 2)).all()

    def test_calc_m_linear_decreasing_is_half_m_for_all(self):

        x = torch.zeros(2, 2, 3)
        pt1 = torch.zeros(2, 2, 3) - 1
        pt2 = torch.ones(2,2, 3)
        m = torch.ones(2, 2, 3) * 0.5

        assert (calc_m_linear_decreasing(x, pt1, pt2, m) == (m / 2)).all()

