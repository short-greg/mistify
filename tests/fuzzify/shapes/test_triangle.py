import torch
from mistify.fuzzify import ShapeParams
from mistify.fuzzify._shapes import _trapezoid, _triangle


class TestIncreasingRightTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        right_triangle = _triangle.IncreasingRightTriangle(
            ShapeParams(p)
        )
        m = right_triangle.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.IncreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert isinstance(shape, _triangle.IncreasingRightTriangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.IncreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.IncreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.IncreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_right_trapezoid(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.IncreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.truncate(m)
        assert isinstance(shape, _triangle.IncreasingRightTrapezoid)



class TestDecreasingRightTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        right_triangle = _triangle.DecreasingRightTriangle(
            ShapeParams(p)
        )
        m = right_triangle.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.DecreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert isinstance(shape, _triangle.DecreasingRightTriangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.DecreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.DecreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.DecreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_right_trapezoid(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.DecreasingRightTriangle(
            ShapeParams(p)
        )
        shape = right_triangle.truncate(m)
        assert isinstance(shape, _triangle.DecreasingRightTrapezoid)


class TestTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        x = torch.rand(2, 3)
        right_triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        m = right_triangle.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])


    def test_join_returns_fuzzy_set_with_correct_size_and_5_inputs(self):

        p = torch.rand(5, 4, 3).cumsum(2)
        x = torch.rand(2, 5)
        right_triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        m = right_triangle.join(x)
        assert m.data.size() == torch.Size([2, 5, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        shape = triangle.scale(m)
        assert isinstance(shape, _triangle.Triangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        shape = right_triangle.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        shape = right_triangle.truncate(m)
        assert isinstance(shape, _trapezoid.Trapezoid)


class TestIsocelesTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        right_trapezoid = _triangle.IsoscelesTriangle(
            ShapeParams(p)
        )
        m = right_trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = _triangle.IsoscelesTriangle(
            ShapeParams(p)
        )
        shape = triangle.scale(m)
        assert isinstance(shape, _triangle.IsoscelesTriangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = _triangle.IsoscelesTriangle(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = _triangle.IsoscelesTriangle(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = _triangle.IsoscelesTriangle(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = _triangle.IsoscelesTriangle(
            ShapeParams(p)
        )
        shape = right_trapezoid.truncate(m)
        assert isinstance(shape, _trapezoid.IsoscelesTrapezoid)
