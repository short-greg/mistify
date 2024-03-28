import torch
from mistify.fuzzify import ShapeParams, Square


class TestSquare(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)

        square = Square(
            ShapeParams(p)
        )
        m = square.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    # def test_scale_returns_shape_with_correct_size(self):

    #     p = torch.rand(3, 4, 2).cumsum(2)
    #     m = torch.rand(2, 3, 4)
    #     square = Square(
    #         ShapeParams(p)
    #     )
    #     square = square.(m)
    #     assert isinstance(square, Square)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        square = Square(
            ShapeParams(p)
        )
        mean_cores = square.mean_cores(m, False)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_mean_core_returns_tensor_with_correct_size_with_truncate(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        square = Square(
            ShapeParams(p)
        )
        mean_cores = square.mean_cores(m, True)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        square = Square(
            ShapeParams(p)
        )
        centroids = square.centroids(m, True)
        assert centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        square = Square(
            ShapeParams(p)
        )
        areas = square.areas(m)
        assert areas.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size_with_truncate(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        square = Square(
            ShapeParams(p)
        )
        areas = square.areas(m, True)
        assert areas.shape == torch.Size([2, 3, 4])

    # def test_truncate_returns_right_trapezoid(self):

    #     p = torch.rand(3, 4, 2).cumsum(2)
    #     m = torch.rand(2, 3, 4)
    #     square = Square(
    #         ShapeParams(p)
    #     )
    #     areas = square.areas(m, True)
    #     assert isinstance(shape, Square)
