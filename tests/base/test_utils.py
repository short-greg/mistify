# Test all utils
import torch
import torch.nn as nn

from mistify._base.utils import (
    ToOptim, calc_m_linear_increasing,
    unsqueeze, calc_area_logistic, calc_m_linear_decreasing,
    calc_m_linear_decreasing, calc_area_logistic_one_side,
    calc_x_linear_decreasing, calc_m_logistic,
    calc_x_logistic, calc_dx_logistic,
    calc_area_logistic, resize_to,
    check_contains, maxmin, minmax,
    maxprod



)

class TestToOptim:

    def test_x_returns_false(self):

        theta = ToOptim.THETA
        assert theta.x() is False

    def test_theta_returns_true(self):

        theta = ToOptim.THETA
        assert theta.theta() is True
    
    def test_theta_returns_true_when_both_set(self):

        theta = ToOptim.BOTH
        assert theta.theta() is True


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


def test_check_contains_is_true_if_inside():

    x = torch.zeros(2, 2, 3)
    y = torch.ones(2, 2, 3)
    pt = torch.ones(2, 2, 3) - 0.5
    assert (check_contains(pt, x, y) == True).all()


def test_check_contains_is_false_if_outside():
    x = torch.zeros(2, 2, 3)
    y = torch.ones(2, 2, 3)
    pt = torch.ones(2, 2, 3) + 0.5
    assert (check_contains(pt, x, y) == False).all()


def test_check_contains_is_true_if_on_boundary():
    x = torch.zeros(2, 2, 3)
    y = torch.ones(2, 2, 3)
    pt = torch.ones(2, 2, 3)
    assert (check_contains(pt, x, y) == True).all()