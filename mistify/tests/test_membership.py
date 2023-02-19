from .. import membership
import torch
import pytest
from ..fuzzy import FuzzySet

class TestShapeParams:

    def test_x_property_equals_x(self):

        x = torch.rand(2, 3, 4, 5)
        params = membership.ShapeParams(x)
        assert (params.x == x).all()

    def test_x_property_unsqueezes_x_when_dim_is_3(self):

        x = torch.rand(3, 4, 5)
        params = membership.ShapeParams(x)
        assert (params.x == x[None]).all()

    def test_n_variables_is_correct_value(self):
        x = torch.rand(2, 3, 4, 5)
        params = membership.ShapeParams(x)
        assert params.n_variables == 3

    def test_n_terms_is_correct_value(self):
        x = torch.rand(2, 3, 4, 5)
        params = membership.ShapeParams(x)
        assert params.n_terms == 4
    
    def test_batch_size_is_correct_value(self):
        x = torch.rand(2, 3, 4, 5)
        params = membership.ShapeParams(x)
        assert params.batch_size == 2

    def test_n_points_is_correct_value(self):
        x = torch.rand(2, 3, 4, 5)
        params = membership.ShapeParams(x)
        assert params.n_points == 5
    
    def test_contains_returns_true_if_inside(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x2 = x[:,:,:,2] * 0.5 + x[:,:,:,3] * 0.5
        params = membership.ShapeParams(x)
        assert params.contains(x2, 2, 3).all()

    def test_contains_returns_false_if_notinside(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x2 = x[:,:,:,3] * 0.5 + x[:,:,:,4] * 0.5
        params = membership.ShapeParams(x)
        assert not params.contains(x2, 2, 3).any()

    def test_insert_inserts_into_params(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x1 = x[:,:,:,:4]
        x2 = x[:,:,:,4]
        params = membership.ShapeParams(x1)
        inserted = params.insert(x2, 4, to_unsqueeze=True)
        assert (inserted.x[:,:,:,4] == x2).all()

    def test_insert_inserts_into_start_of_params(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x1 = x[:,:,:,:4]
        x2 = x[:,:,:,4]
        params = membership.ShapeParams(x1)
        inserted = params.insert(x2, 0, to_unsqueeze=True)
        assert (inserted.x[:,:,:,0] == x2).all()

    def test_insert_inserts_into_end_of_params(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x1 = x[:,:,:,:4]
        x2 = x[:,:,:,4]
        params = membership.ShapeParams(x1)
        inserted = params.insert(x2, 2, to_unsqueeze=True)
        assert (inserted.x[:,:,:,2] == x2).all()

    def test_replace_inserts_into_start_of_params(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x1 = x[:,:,:,:4]
        x2 = x[:,:,:,4]
        params = membership.ShapeParams(x1)
        inserted = params.replace(x2, 0, to_unsqueeze=True)
        assert (inserted.x[:,:,:,0] == x2).all() and inserted.x.size(3) == 4

    def test_replace_inserts_into_end_of_params(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x1 = x[:,:,:,:4]
        x2 = x[:,:,:,4]
        params = membership.ShapeParams(x1)
        inserted = params.replace(x2, 3, to_unsqueeze=True)
        assert (inserted.x[:,:,:,3] == x2).all() and inserted.x.size(3) == 4

    def test_replace_raises_error_if_outside_bounds(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x1 = x[:,:,:,:4]
        x2 = x[:,:,:,4]
        params = membership.ShapeParams(x1)
        with pytest.raises(ValueError):
            params.replace(x2, 4, to_unsqueeze=True)


class TestIncreasingRightTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        right_triangle = membership.IncreasingRightTriangle(
            membership.ShapeParams(p)
        )
        m = right_triangle.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_triangle = membership.IncreasingRightTriangle(
            membership.ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert isinstance(shape, membership.IncreasingRightTriangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_triangle = membership.IncreasingRightTriangle(
            membership.ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_triangle = membership.IncreasingRightTriangle(
            membership.ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_triangle = membership.IncreasingRightTriangle(
            membership.ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_right_trapezoid(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_triangle = membership.IncreasingRightTriangle(
            membership.ShapeParams(p)
        )
        shape = right_triangle.truncate(m)
        assert isinstance(shape, membership.IncreasingRightTrapezoid)


class TestIncreasingRightTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        x = torch.rand(2, 3)
        right_trapezoid = membership.IncreasingRightTrapezoid(
            membership.ShapeParams(p)
        )
        m = right_trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_trapezoid = membership.IncreasingRightTrapezoid(
            membership.ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert isinstance(shape, membership.IncreasingRightTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_trapezoid = membership.IncreasingRightTrapezoid(
            membership.ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_trapezoid = membership.IncreasingRightTrapezoid(
            membership.ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_trapezoid = membership.IncreasingRightTrapezoid(
            membership.ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_right_trapezoid(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_trapezoid = membership.IncreasingRightTrapezoid(
            membership.ShapeParams(p)
        )
        shape = right_trapezoid.truncate(m)
        assert isinstance(shape, membership.IncreasingRightTrapezoid)


class TestSquare(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        right_trapezoid = membership.Square(
            membership.ShapeParams(p)
        )
        m = right_trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        square = membership.Square(
            membership.ShapeParams(p)
        )
        square = square.scale(m)
        assert isinstance(square, membership.Square)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        square = membership.Square(
            membership.ShapeParams(p)
        )
        shape = square.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        square = membership.Square(
            membership.ShapeParams(p)
        )
        shape = square.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        square = membership.Square(
            membership.ShapeParams(p)
        )
        shape = square.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_right_trapezoid(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        square = membership.Square(
            membership.ShapeParams(p)
        )
        shape = square.truncate(m)
        assert isinstance(shape, membership.Square)


class TestDecreasingRightTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        right_triangle = membership.DecreasingRightTriangle(
            membership.ShapeParams(p)
        )
        m = right_triangle.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_triangle = membership.DecreasingRightTriangle(
            membership.ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert isinstance(shape, membership.DecreasingRightTriangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_triangle = membership.DecreasingRightTriangle(
            membership.ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_triangle = membership.DecreasingRightTriangle(
            membership.ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_triangle = membership.DecreasingRightTriangle(
            membership.ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_right_trapezoid(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_triangle = membership.DecreasingRightTriangle(
            membership.ShapeParams(p)
        )
        shape = right_triangle.truncate(m)
        assert isinstance(shape, membership.DecreasingRightTrapezoid)


class TestTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        x = torch.rand(2, 3)
        right_trapezoid = membership.Triangle(
            membership.ShapeParams(p)
        )
        m = right_trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        triangle = membership.Triangle(
            membership.ShapeParams(p)
        )
        shape = triangle.scale(m)
        assert isinstance(shape, membership.Triangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_trapezoid = membership.Triangle(
            membership.ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_trapezoid = membership.Triangle(
            membership.ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_trapezoid = membership.Triangle(
            membership.ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_trapezoid = membership.Triangle(
            membership.ShapeParams(p)
        )
        shape = right_trapezoid.truncate(m)
        assert isinstance(shape, membership.Trapezoid)


class TestTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        x = torch.rand(2, 3)
        trapezoid = membership.Trapezoid(
            membership.ShapeParams(p)
        )
        m = trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        trapezoid = membership.Trapezoid(
            membership.ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert isinstance(shape, membership.Trapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        trapezoid = membership.Trapezoid(
            membership.ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        trapezoid = membership.Trapezoid(
            membership.ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        trapezoid = membership.Trapezoid(
            membership.ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        trapezoid = membership.Trapezoid(
            membership.ShapeParams(p)
        )
        shape = trapezoid.truncate(m)
        assert isinstance(shape, membership.Trapezoid)


class TestIsocelesTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        right_trapezoid = membership.IsoscelesTriangle(
            membership.ShapeParams(p)
        )
        m = right_trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        triangle = membership.IsoscelesTriangle(
            membership.ShapeParams(p)
        )
        shape = triangle.scale(m)
        assert isinstance(shape, membership.IsoscelesTriangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_trapezoid = membership.IsoscelesTriangle(
            membership.ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_trapezoid = membership.IsoscelesTriangle(
            membership.ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_trapezoid = membership.IsoscelesTriangle(
            membership.ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        right_trapezoid = membership.IsoscelesTriangle(
            membership.ShapeParams(p)
        )
        shape = right_trapezoid.truncate(m)
        assert isinstance(shape, membership.IsoscelesTrapezoid)


class TestIsoscelesTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        x = torch.rand(2, 3)
        trapezoid = membership.IsoscelesTrapezoid(
            membership.ShapeParams(p)
        )
        m = trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        trapezoid = membership.IsoscelesTrapezoid(
            membership.ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert isinstance(shape, membership.IsoscelesTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        trapezoid = membership.IsoscelesTrapezoid(
            membership.ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        trapezoid = membership.IsoscelesTrapezoid(
            membership.ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        trapezoid = membership.IsoscelesTrapezoid(
            membership.ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        trapezoid = membership.IsoscelesTrapezoid(
            membership.ShapeParams(p)
        )
        shape = trapezoid.truncate(m)
        assert isinstance(shape, membership.IsoscelesTrapezoid)


class TestLogistic(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        x = torch.rand(2, 3)
        logistic = membership.LogisticBell(
            b, s
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.LogisticBell(
            b, s
        )
        shape = logistic.scale(m)
        assert isinstance(shape, membership.LogisticBell)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.LogisticBell(
            b, s
        )
        shape = logistic.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.LogisticBell(
            b, s
        )
        shape = logistic.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.LogisticBell(
            b, s
        )
        shape = logistic.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.LogisticBell(
            b, s
        )
        shape = logistic.truncate(m)
        assert isinstance(shape, membership.LogisticTrapezoid)


class TestLogisticTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)
        x = torch.rand(2, 3)
        logistic = membership.LogisticTrapezoid(
            b, s, truncated_m
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.scale(m)
        assert isinstance(shape, membership.LogisticTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        m = FuzzySet(torch.rand(2, 3, 4), True)
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)

        logistic = membership.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        m = FuzzySet(torch.rand(2, 3, 4), True)
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)

        logistic = membership.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.truncate(m)
        assert isinstance(shape, membership.LogisticTrapezoid)


class TestRightLogistic(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)
        x = torch.rand(2, 3)
        logistic = membership.RightLogistic(
            b, s, True, truncated_m
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert isinstance(shape, membership.RightLogistic)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        m = FuzzySet(torch.rand(2, 3, 4), True)
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)

        logistic = membership.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        m = FuzzySet(torch.rand(2, 3, 4), True)
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)

        logistic = membership.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.truncate(m)
        assert isinstance(shape, membership.RightLogisticTrapezoid)







class TestRightLogisticTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)
        x = torch.rand(2, 3)
        logistic = membership.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert isinstance(shape, membership.RightLogisticTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        m = FuzzySet(torch.rand(2, 3, 4), True)
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)

        logistic = membership.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)
        m = FuzzySet(torch.rand(2, 3, 4), True)
        logistic = membership.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = membership.ShapeParams(torch.rand(3, 4, 1))
        s = membership.ShapeParams(torch.rand(3, 4, 1))
        m = FuzzySet(torch.rand(2, 3, 4), True)
        truncated_m = FuzzySet(torch.rand(2, 3, 4), True)

        logistic = membership.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.truncate(m)
        assert isinstance(shape, membership.RightLogisticTrapezoid)
