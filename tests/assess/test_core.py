
from mistify.assess._core import ToOptim


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

