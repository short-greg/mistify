import pytest
import torch
from mistify import fuzzy
from mistify.base import ShapeParams


class TestIncreasingRightTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        right_triangle = fuzzy.IncreasingRightTriangle(
            ShapeParams(p)
        )
        m = right_triangle.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.IncreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert isinstance(shape, fuzzy.IncreasingRightTriangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.IncreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.IncreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.IncreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_right_trapezoid(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.IncreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.truncate(m)
        assert isinstance(shape, fuzzy.IncreasingRightTrapezoid)


class TestIncreasingRightTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        x = torch.rand(2, 3)
        right_trapezoid = fuzzy.IncreasingRightTrapezoid(
            ShapeParams(p)
        )
        m = right_trapezoid.join(x)
        assert m.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = fuzzy.IncreasingRightTrapezoid(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert isinstance(shape, fuzzy.IncreasingRightTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = fuzzy.IncreasingRightTrapezoid(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = fuzzy.IncreasingRightTrapezoid(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = fuzzy.IncreasingRightTrapezoid(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_right_trapezoid(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = fuzzy.IncreasingRightTrapezoid(
            ShapeParams(p)
        )
        shape = right_trapezoid.truncate(m)
        assert isinstance(shape, fuzzy.IncreasingRightTrapezoid)


class TestDecreasingRightTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        right_triangle = fuzzy.DecreasingRightTriangle(
            ShapeParams(p)
        )
        m = right_triangle.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.DecreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert isinstance(shape, fuzzy.DecreasingRightTriangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.DecreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.DecreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.DecreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_right_trapezoid(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.DecreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.truncate(m)
        assert isinstance(shape, fuzzy.DecreasingRightTrapezoid)


class TestTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        x = torch.rand(2, 3)
        right_triangle = fuzzy.Triangle(
            ShapeParams(p)
        )
        m = right_triangle.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])


    def test_join_returns_fuzzy_set_with_correct_size_and_5_inputs(self):

        p = torch.rand(5, 4, 3).cumsum(2)
        x = torch.rand(2, 5)
        right_triangle = fuzzy.Triangle(
            ShapeParams(p)
        )
        m = right_triangle.join(x)
        assert m.data.size() == torch.Size([2, 5, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = fuzzy.Triangle(
            ShapeParams(p)
        )
        shape = triangle.scale(m)
        assert isinstance(shape, fuzzy.Triangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.Triangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.Triangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.Triangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = fuzzy.Triangle(
            ShapeParams(p)
        )
        shape = right_triangle.truncate(m)
        assert isinstance(shape, fuzzy.Trapezoid)


class TestTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        x = torch.rand(2, 3)
        trapezoid = fuzzy.Trapezoid(
            ShapeParams(p)
        )
        m = trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = fuzzy.Trapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert isinstance(shape, fuzzy.Trapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = fuzzy.Trapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = fuzzy.Trapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = fuzzy.Trapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = fuzzy.Trapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.truncate(m)
        assert isinstance(shape, fuzzy.Trapezoid)


class TestIsocelesTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        right_trapezoid = fuzzy.IsoscelesTriangle(
            ShapeParams(p)
        )
        m = right_trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = fuzzy.IsoscelesTriangle(
            ShapeParams(p)
        )
        shape = triangle.scale(m)
        assert isinstance(shape, fuzzy.IsoscelesTriangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = fuzzy.IsoscelesTriangle(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = fuzzy.IsoscelesTriangle(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = fuzzy.IsoscelesTriangle(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = fuzzy.IsoscelesTriangle(
            ShapeParams(p)
        )
        shape = right_trapezoid.truncate(m)
        assert isinstance(shape, fuzzy.IsoscelesTrapezoid)


class TestIsoscelesTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        x = torch.rand(2, 3)
        trapezoid = fuzzy.IsoscelesTrapezoid(
            ShapeParams(p)
        )
        m = trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = fuzzy.IsoscelesTrapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert isinstance(shape, fuzzy.IsoscelesTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = fuzzy.IsoscelesTrapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = fuzzy.IsoscelesTrapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = fuzzy.IsoscelesTrapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = fuzzy.IsoscelesTrapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.truncate(m)
        assert isinstance(shape, fuzzy.IsoscelesTrapezoid)


class TestLogistic(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        x = torch.rand(2, 3)
        logistic = fuzzy.LogisticBell(
            b, s
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.LogisticBell(
            b, s
        )
        shape = logistic.scale(m)
        assert isinstance(shape, fuzzy.LogisticBell)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.LogisticBell(
            b, s
        )
        shape = logistic.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.LogisticBell(
            b, s
        )
        shape = logistic.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.LogisticBell(
            b, s
        )
        shape = logistic.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.LogisticBell(
            b, s
        )
        shape = logistic.truncate(m)
        assert isinstance(shape, fuzzy.LogisticTrapezoid)


class TestLogisticTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        x = torch.rand(2, 3)
        logistic = fuzzy.LogisticTrapezoid(
            b, s, truncated_m
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.scale(m)
        assert isinstance(shape, fuzzy.LogisticTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        logistic = fuzzy.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        logistic = fuzzy.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.truncate(m)
        assert isinstance(shape, fuzzy.LogisticTrapezoid)


class TestRightLogistic(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        x = torch.rand(2, 3)
        logistic = fuzzy.RightLogistic(
            b, s, True, truncated_m
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert isinstance(shape, fuzzy.RightLogistic)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        logistic = fuzzy.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        logistic = fuzzy.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.truncate(m)
        assert isinstance(shape, fuzzy.RightLogisticTrapezoid)


class TestRightLogisticTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        x = torch.rand(2, 3)
        logistic = fuzzy.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert isinstance(shape, fuzzy.RightLogisticTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        logistic = fuzzy.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = fuzzy.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        logistic = fuzzy.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.truncate(m)
        assert isinstance(shape, fuzzy.RightLogisticTrapezoid)
