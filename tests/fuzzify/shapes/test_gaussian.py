import torch

from mistify.fuzzify import ShapeParams
from mistify.fuzzify._shapes import _gaussian
from pprint import pprint


class TestGaussian(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        x = torch.rand(2, 3)
        gaussian = _gaussian.GaussianBell(
            b, s
        )
        m = gaussian.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.GaussianBell(
            b, s
        )
        shape = gaussian.scale(m)
        assert isinstance(shape, _gaussian.GaussianBell)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.ones(2, 3, 4)
        gaussian = _gaussian.GaussianBell(
            b, s
        )
        shape = gaussian.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.GaussianBell(
            b, s
        )
        shape = gaussian.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.GaussianBell(
            b, s
        )
        shape = gaussian.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.GaussianBell(
            b, s
        )
        shape = gaussian.truncate(m)
        assert isinstance(shape, _gaussian.GaussianTrapezoid)


class TestGaussianTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        x = torch.rand(2, 3)
        gaussian = _gaussian.GaussianTrapezoid(
            b, s, truncated_m
        )
        m = gaussian.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.GaussianTrapezoid(
            b, s, truncated_m
        )
        shape = gaussian.scale(m)
        assert isinstance(shape, _gaussian.GaussianTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        gaussian = _gaussian.GaussianTrapezoid(
            b, s, truncated_m
        )
        shape = gaussian.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.GaussianTrapezoid(
            b, s, truncated_m
        )
        shape = gaussian.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.GaussianTrapezoid(
            b, s, truncated_m
        )
        shape = gaussian.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        gaussian = _gaussian.GaussianTrapezoid(
            b, s, truncated_m
        )
        shape = gaussian.truncate(m)
        assert isinstance(shape, _gaussian.GaussianTrapezoid)


class TestRightLogistic(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        x = torch.rand(2, 3)
        gaussian = _gaussian.RightGaussian(
            b, s, True, truncated_m
        )
        m = gaussian.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.RightGaussian(
            b, s, True, truncated_m
        )
        shape = gaussian.scale(m)
        assert isinstance(shape, _gaussian.RightGaussian)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        gaussian = _gaussian.RightGaussian(
            b, s, True, truncated_m
        )
        shape = gaussian.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.RightGaussian(
            b, s, True, truncated_m
        )
        shape = gaussian.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.RightGaussian(
            b, s, True, truncated_m
        )
        shape = gaussian.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        gaussian = _gaussian.RightGaussian(
            b, s, True, truncated_m
        )
        shape = gaussian.truncate(m)
        assert isinstance(shape, _gaussian.RightGaussianTrapezoid)


class TestRightLogisticTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        x = torch.rand(2, 3)
        gaussian = _gaussian.RightGaussianTrapezoid(
            b, s, True, truncated_m
        )
        m = gaussian.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.RightGaussianTrapezoid(
            b, s, True, truncated_m
        )
        shape = gaussian.scale(m)
        assert isinstance(shape, _gaussian.RightGaussianTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        gaussian = _gaussian.RightGaussianTrapezoid(
            b, s, True, truncated_m
        )
        shape = gaussian.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

#         b = ShapeParams(torch.rand(3, 4, 1))
#         s = ShapeParams(torch.rand(3, 4, 1))
#         truncated_m = torch.rand(2, 3, 4)
#         m = torch.rand(2, 3, 4)
#         logistic = _logistic.RightLogisticTrapezoid(
#             b, s, True, truncated_m
#         )
#         shape = logistic.scale(m)
#         assert shape.mean_cores.shape == torch.Size([2, 3, 4])


    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.RightGaussianTrapezoid(
            b, s, True, truncated_m
        )
        shape = gaussian.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.RightGaussianTrapezoid(
            b, s, True, truncated_m
        )
        shape = gaussian.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        gaussian = _gaussian.RightGaussianTrapezoid(
            b, s, True, truncated_m
        )
        shape = gaussian.truncate(m)
        assert isinstance(shape, _gaussian.RightGaussianTrapezoid)
