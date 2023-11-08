# Test all utils
import torch
import torch.nn as nn

from mistify._base import (
    ToOptim
)
from mistify._base.utils import unsqueeze

from mistify.utils import (
    check_contains
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
