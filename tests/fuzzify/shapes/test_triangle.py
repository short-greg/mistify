import torch
from mistify.fuzzify import ShapeParams
from mistify.fuzzify._shapes import _trapezoid, _triangle


class TestRightTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        right_triangle = _triangle.RightTriangle(
            ShapeParams(p)
        )
        m = right_triangle.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.RightTriangle(
            ShapeParams(p)
        )
        scaled = right_triangle.centroids(m)
        assert scaled.shape == torch.Size([2, 3, 4])

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.RightTriangle(
            ShapeParams(p), True
        )
        # shape = right_triangle.scale(m)
        mean_cores = right_triangle.mean_cores(m)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_truncated_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.RightTriangle(
            ShapeParams(p)
        )
        # shape = right_triangle.scale(m)
        centroids = right_triangle.centroids(m, True)
        assert centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.RightTriangle(
            ShapeParams(p), False
        )
        areas = right_triangle.areas(m, True)
        assert areas.shape == torch.Size([2, 3, 4])

    # def test_truncate_returns_right_trapezoid(self):

    #     p = torch.rand(3, 4, 2).cumsum(2)
    #     m = torch.rand(2, 3, 4)
    #     right_triangle = _triangle.RightTriangle(
    #         ShapeParams(p)
    #     )
    #     # shape = right_triangle.truncate(m)
    #     assert isinstance(shape, _triangle.IncreasingRightTrapezoid)




    def test_scale_returns_shape_with_correct_size_with_decreasing(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.RightTriangle(
            ShapeParams(p), False
        )
        mean_cores = right_triangle.mean_cores(m)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_mean_cores_with_truncation_returns_shape_with_correct_size_with_decreasing(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.RightTriangle(
            ShapeParams(p), False
        )
        mean_cores = right_triangle.mean_cores(m, True)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_with_truncation_returns_shape_with_correct_size_with_decreasing(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.RightTriangle(
            ShapeParams(p), False
        )
        centroids = right_triangle.centroids(m, True)
        assert centroids.shape == torch.Size([2, 3, 4])

    def test_centroids_with_scaling_returns_shape_with_correct_size_with_decreasing(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.RightTriangle(
            ShapeParams(p), False
        )
        centroids = right_triangle.centroids(m, False)
        assert centroids.shape == torch.Size([2, 3, 4])


    def test_areas_with_truncation_returns_shape_with_correct_size_with_decreasing(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.RightTriangle(
            ShapeParams(p), False
        )
        areas = right_triangle.areas(m, True)
        assert areas.shape == torch.Size([2, 3, 4])

    def test_araes_with_scaling_returns_shape_with_correct_size_with_decreasing(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_triangle = _triangle.RightTriangle(
            ShapeParams(p), False
        )
        areas = right_triangle.areas(m, False)
        assert areas.shape == torch.Size([2, 3, 4])



class TestTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        x = torch.rand(2, 3)
        triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        m = triangle.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_order_presevered_after_updating(self):

        torch.manual_seed(1)
        p = torch.rand(5, 4, 3).cumsum(2)
        x = torch.rand(2, 5)
        triangle = _triangle.Triangle(
            ShapeParams(p, True)
        )
        optim = torch.optim.Adam(triangle.parameters(), lr=1e0)
        m = triangle.join(x)
        t = torch.rand_like(m)
        optim.zero_grad()
        (m - t).pow(2).sum().backward()
        optim.step()

        p = triangle.params()
        assert (triangle._params.x[:,:,:,:-1] >= triangle._params.x[:,:,:,1:]).any()
        assert (p.x[:,:,:,:-1] < p.x[:,:,:,1:]).all()

#     def test_scale_returns_shape_with_correct_size(self):

#         p = torch.rand(3, 4, 3).cumsum(2)
#         m = torch.rand(2, 3, 4)
#         triangle = _triangle.Triangle(
#             ShapeParams(p)
#         )
#         shape = triangle.scale(m)
#         assert isinstance(shape, _triangle.Triangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        mean_cores = triangle.mean_cores(m)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        centroids = triangle.centroids(m)
        assert centroids.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size_using_truncation(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        centroids = triangle.centroids(m, True)
        assert centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        areas = triangle.areas(m)
        assert areas.shape == torch.Size([2, 3, 4])

    def test_truncate_areas_returns_truncate(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        areas = triangle.areas(m, True)
        assert areas.shape == torch.Size([2, 3, 4])

    def test_scaled_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = _triangle.Triangle(
            ShapeParams(p)
        )
        areas = triangle.areas(m)
        assert areas.shape == torch.Size([2, 3, 4])

class TestIsocelesTriangle(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        right_trapezoid = _triangle.IsoscelesTriangle(
            ShapeParams(p)
        )
        m = right_trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    # def test_scale_returns_shape_with_correct_size(self):

    #     p = torch.rand(3, 4, 2).cumsum(2)
    #     m = torch.rand(2, 3, 4)
    #     triangle = _triangle.IsoscelesTriangle(
    #         ShapeParams(p)
    #     )
    #     shape = triangle.scale(m)
    #     assert isinstance(shape, _triangle.IsoscelesTriangle)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = _triangle.IsoscelesTriangle(
            ShapeParams(p)
        )
        mean_cores = triangle.mean_cores(m)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = _triangle.IsoscelesTriangle(
            ShapeParams(p)
        )
        centroids = right_trapezoid.centroids(m)
        assert centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = _triangle.IsoscelesTriangle(
            ShapeParams(p)
        )
        areas = triangle.areas(m)
        assert areas.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size_with_truncate(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = _triangle.IsoscelesTriangle(
            ShapeParams(p)
        )
        areas = triangle.areas(m, True)
        assert areas.shape == torch.Size([2, 3, 4])

    def test_mean_cores_returns_tensor_with_correct_size_with_truncate(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        triangle = _triangle.IsoscelesTriangle(
            ShapeParams(p)
        )
        mean_cores = triangle.mean_cores(m, True)
        assert mean_cores.shape == torch.Size([2, 3, 4])
