# Test all utils
import torch


from mistify.utils import (
    check_contains
)


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
