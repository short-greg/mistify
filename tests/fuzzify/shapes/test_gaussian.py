import torch

from mistify.fuzzify import ShapeParams
from mistify.fuzzify._shapes import _gaussian


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

    def test_areas_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.GaussianBell(
            b, s
        )
        areas = gaussian.areas(m)
        assert areas.shape == torch.Size([2, 3, 4])

    def test_areas_returns_shape_with_correct_size_with_truncate(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.GaussianBell(
            b, s
        )
        areas = gaussian.areas(m, True)
        assert areas.shape == torch.Size([2, 3, 4])


    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.ones(2, 3, 4)
        gaussian = _gaussian.GaussianBell(
            b, s
        )
        mean_cores = gaussian.mean_cores(m)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_mean_core_returns_tensor_with_correct_size_truncate(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.ones(2, 3, 4)
        gaussian = _gaussian.GaussianBell(
            b, s
        )
        mean_cores = gaussian.mean_cores(m, True)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.GaussianBell(
            b, s
        )
        centroids = gaussian.centroids(m)
        assert centroids.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size_with_truncate(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.GaussianBell(
            b, s
        )
        centroids = gaussian.centroids(m, True)
        assert centroids.shape == torch.Size([2, 3, 4])


class TestRightGaussian(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        x = torch.rand(2, 3)
        gaussian = _gaussian.HalfGaussianBell(
            b, s, True
        )
        m = gaussian.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])


    def test_join_returns_fuzzy_set_with_correct_size_with_decreasing(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        x = torch.rand(2, 3)
        gaussian = _gaussian.HalfGaussianBell(
            b, s, False
        )
        m = gaussian.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)

        gaussian = _gaussian.HalfGaussianBell(
            b, s, True
        )
        mean_cores = gaussian.mean_cores(m)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.HalfGaussianBell(
            b, s, False
        )
        centroids = gaussian.centroids(m)
        assert centroids.shape == torch.Size([2, 3, 4])
    
    def test_centroids_returns_tensor_with_correct_size_truncated(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.HalfGaussianBell(
            b, s, False
        )
        centroids = gaussian.centroids(m, True)
        assert centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.HalfGaussianBell(
            b, s, True
        )
        areas = gaussian.areas(m)
        assert areas.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size_truncated(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        gaussian = _gaussian.HalfGaussianBell(
            b, s, True
        )
        areas = gaussian.areas(m, True)
        assert areas.shape == torch.Size([2, 3, 4])



class TestGaussianFunctions:

    def test_gaussian_area_gives_correct_value(self):

        areas = _gaussian.gaussian_area(torch.tensor([2, 0.5]))
        assert torch.isclose(areas[0], torch.tensor(5.01325654), atol=1e-4)
        assert torch.isclose(areas[1], torch.tensor(1.2533), atol=1e-4)

    def test_gaussian_invert_inverts_the_value(self):

        x = torch.tensor([-1., 0.1])
        bias = torch.tensor([0.0, 0.0])
        scale = torch.tensor([1.0, 1.0])
        y = _gaussian.gaussian(x, bias, scale)
        lhs, rhs = _gaussian.gaussian_invert(y, bias, scale)
        assert torch.isclose(lhs[0], x[0], 1e-4).all()
        assert torch.isclose(rhs[1], x[1], 1e-4).all()

    def test_gaussian_invert_inverts_the_value_with_scale_and_bias(self):

        x = torch.tensor([0.1, 0.1])
        bias = torch.tensor([0.2, -0.1])
        scale = torch.tensor([2.0, 0.5])
        y = _gaussian.gaussian(x, bias, scale)
        lhs, rhs = _gaussian.gaussian_invert(y, bias, scale)
        assert torch.isclose(lhs[0], x[0], 1e-4).all()
        assert torch.isclose(rhs[1], x[1], 1e-4).all()

    def test_gaussian_gives_value_at_one(self):

        x = torch.tensor([0.2, 4.0])
        bias = torch.tensor([0.2, 0.0])
        scale = torch.tensor([1.0, 2.0])

        m = _gaussian.gaussian(x, bias, scale)
        assert torch.isclose(m[0], torch.tensor(1.0), atol=1e-4).all()
        assert torch.isclose(m[1], torch.tensor(0.1353), atol=1e-4).all()

    def test_gaussian_area_up_to(self):

        x = torch.tensor([0.2, -2.0])
        bias = torch.tensor([0.2, 0.0])
        scale = torch.tensor([1.0, 2.0])

        full = _gaussian.gaussian_area(scale)
        t2 = (0.5 - 0.6826895 / 2.0) * full[1]
        # print(full[1] * 0.16)
        m = _gaussian.gaussian_area_up_to(x, bias, scale)
        assert torch.isclose(m[0], full[0] / 2., atol=1e-2).all()
        assert torch.isclose(m[1], t2, atol=1e-4).all()

    def test_gaussian_area_up_to_inv(self):

        bias = torch.tensor([0.2, 0.0])
        scale = torch.tensor([1.0, 2.0])
        area = _gaussian.gaussian_area(scale)
        area[0] *= 0.5
        area[1] *= (0.5 - 0.6826895 / 2.0)

        m = _gaussian.gaussian_area_up_to_inv(area, bias, scale)
        assert torch.isclose(m[0], torch.tensor(0.2), 1e-4).all()
        assert torch.isclose(m[1], torch.tensor(-2.0), 1e-4).all()
