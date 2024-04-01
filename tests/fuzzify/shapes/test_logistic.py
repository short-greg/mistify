import torch

from mistify.fuzzify import ShapeParams
from mistify.fuzzify._shapes import _logistic
from pprint import pprint


class TestLogistic(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        x = torch.rand(2, 3)
        logistic = _logistic.LogisticBell(
            b, s
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_areas_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        areas = logistic.areas(m)
        assert areas.shape == torch.Size([2, 3, 4])

    def test_areas_returns_shape_with_correct_size_with_truncate(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        areas = logistic.areas(m, True)
        assert areas.shape == torch.Size([2, 3, 4])

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.ones(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        mean_cores = logistic.mean_cores(m)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_mean_core_returns_tensor_with_correct_size_truncate(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.ones(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        mean_cores = logistic.mean_cores(m, True)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        centroids = logistic.centroids(m)
        assert centroids.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size_with_truncate(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        centroids = logistic.centroids(m, True)
        assert centroids.shape == torch.Size([2, 3, 4])


class TestRightLogistic(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        x = torch.rand(2, 3)
        logistic = _logistic.HalfLogisticBell(
            b, s, True
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_join_returns_fuzzy_set_with_correct_size_with_decreasing(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        x = torch.rand(2, 3)
        logistic = _logistic.HalfLogisticBell(
            b, s, False
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)

        logistic = _logistic.HalfLogisticBell(
            b, s, True
        )
        mean_cores = logistic.mean_cores(m)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.HalfLogisticBell(
            b, s, False
        )
        centroids = logistic.centroids(m)
        assert centroids.shape == torch.Size([2, 3, 4])
    
    def test_centroids_returns_tensor_with_correct_size_truncated(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.HalfLogisticBell(
            b, s, False
        )
        centroids = logistic.centroids(m, True)
        assert centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.HalfLogisticBell(
            b, s, True
        )
        areas = logistic.areas(m)
        assert areas.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size_truncated(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.HalfLogisticBell(
            b, s, True
        )
        areas = logistic.areas(m, True)
        assert areas.shape == torch.Size([2, 3, 4])


class TestLogisticFunctions:

    def test_logistic_area_gives_correct_value(self):

        areas = _logistic.logistic_area(torch.tensor([2, 0.5]))
        assert areas[0].item() == 8
        assert areas[1].item() == 2

    def test_logistic_invert_inverts_the_value(self):

        x = torch.tensor([-1., 0.1])
        bias = torch.tensor([0.0, 0.0])
        scale = torch.tensor([1.0, 1.0])
        y = _logistic.logistic(x, bias, scale)
        lhs, rhs = _logistic.logistic_invert(y, bias, scale)
        assert torch.isclose(lhs[0], x[0], 1e-4).all()
        assert torch.isclose(rhs[1], x[1], 1e-4).all()

    def test_logistic_invert_inverts_the_value_with_scale_and_bias(self):

        x = torch.tensor([0.1, 0.1])
        bias = torch.tensor([0.2, -0.1])
        scale = torch.tensor([2.0, 0.5])
        y = _logistic.logistic(x, bias, scale)
        lhs, rhs = _logistic.logistic_invert(y, bias, scale)
        assert torch.isclose(lhs[0], x[0], 1e-4).all()
        assert torch.isclose(rhs[1], x[1], 1e-4).all()

    def test_logistic_gives_value_at_one(self):

        x = torch.tensor([0.2, 4.0])
        bias = torch.tensor([0.2, 0.0])
        scale = torch.tensor([1.0, 2.0])

        m = _logistic.logistic(x, bias, scale)
        assert torch.isclose(m[0], torch.tensor(1.0), 1e-4).all()
        assert torch.isclose(m[1], torch.tensor(0.42), 1e-4).all()

    def test_logistic_area_up_to(self):

        x = torch.tensor([0.2, 1.0])
        bias = torch.tensor([0.2, 0.0])
        scale = torch.tensor([1.0, 2.0])

        m = _logistic.logistic_area_up_to(x, bias, scale)
        assert torch.isclose(m[0], torch.tensor(2.0), 1e-4).all()
        assert torch.isclose(m[1], torch.tensor(1.245 * 4), 1e-4).all()

    def test_logistic_area_up_to_inv(self):

        y = torch.tensor([0.5, 1.0])
        bias = torch.tensor([0.2, 0.0])
        scale = torch.tensor([1.0, 2.0])

        m = _logistic.logistic_area_up_to_inv(y, bias, scale)
        assert torch.isclose(m[0], torch.tensor(0.2), 1e-4).all()
        assert torch.isclose(m[1], torch.tensor(0.0), 1e-4).all()

    def test_truncated_logistic_area(self):

        height = torch.tensor([0.0, 1.0])
        bias = torch.tensor([0.2, 0.0])
        scale = torch.tensor([1.0, 2.0])

        m = _logistic.truncated_logistic_area(bias, scale, height)
        assert torch.isclose(m[0], torch.tensor(0.0), atol=1e-5).all()
        assert torch.isclose(m[1], torch.tensor(8.0), atol=1e-4).all()

    def test_truncated_logistic_mean_core(self):

        height = torch.tensor([0.0, 1.0])
        bias = torch.tensor([0.2, 0.0])
        scale = torch.tensor([1.0, 2.0])

        m = _logistic.truncated_logistic_mean_core(bias, scale, height)
        assert torch.isclose(m[0], torch.tensor(0.2), atol=1e-5).all()
        assert torch.isclose(m[1], torch.tensor(0.0), atol=1e-4).all()

    def test_half_logistic_area(self):

        scale = torch.tensor([1.0, 2.0])

        m = _logistic.half_logistic_area(scale)
        assert torch.isclose(m[0], torch.tensor(2.0), atol=1e-4).all()
        assert torch.isclose(m[1], torch.tensor(4.0), atol=1e-4).all()

    def test_truncated_half_logistic_area(self):

        height = torch.tensor([0.0, 1.0])
        bias = torch.tensor([0.2, 0.0])
        scale = torch.tensor([1.0, 2.0])

        m = _logistic.truncated_half_logistic_area(bias, scale, height)
        assert torch.isclose(m[0], torch.tensor(0.0), atol=1e-4).all()
        assert torch.isclose(m[1], torch.tensor(4.0), atol=1e-4).all()



# class TestLogistic(object):

#     def test_join_returns_fuzzy_set_with_correct_size(self):

#         b = ShapeParams(torch.rand(3, 4, 1))
#         s = ShapeParams(torch.rand(3, 4, 1))
#         x = torch.rand(2, 3)
#         logistic = _logistic.LogisticBell(
#             b, s
#         )
#         m = logistic.join(x)
#         assert m.data.size() == torch.Size([2, 3, 4])

    # def test_scale_returns_shape_with_correct_size(self):

    #     b = ShapeParams(torch.rand(3, 4, 1))
    #     s = ShapeParams(torch.rand(3, 4, 1))
    #     m = torch.rand(2, 3, 4)
    #     logistic = _logistic.LogisticBell(
    #         b, s
    #     )
    #     # shape = logistic.scale(m)
    #     assert isinstance(shape, _logistic.LogisticBell)

#     def test_mean_core_returns_tensor_with_correct_size(self):

#         b = ShapeParams(torch.rand(3, 4, 1))
#         s = ShapeParams(torch.rand(3, 4, 1))
#         m = torch.ones(2, 3, 4)
#         logistic = _logistic.LogisticBell(
#             b, s
#         )
#         mean_cores = logistic.mean_cores(m, False)
#         # shape = logistic.scale(m)
#         assert mean_cores.shape == torch.Size([2, 3, 4])

#     def test_centroids_returns_tensor_with_correct_size(self):
#         b = ShapeParams(torch.rand(3, 4, 1))
#         s = ShapeParams(torch.rand(3, 4, 1))
#         m = torch.rand(2, 3, 4)
#         logistic = _logistic.LogisticBell(
#             b, s
#         )
#         # shape = logistic.scale(m)
#         centroids = logistic.centroids(m, True)
#         assert centroids.shape == torch.Size([2, 3, 4])

#     def test_areas_returns_tensor_with_correct_size(self):
#         b = ShapeParams(torch.rand(3, 4, 1))
#         s = ShapeParams(torch.rand(3, 4, 1))
#         m = torch.rand(2, 3, 4)
#         logistic = _logistic.LogisticBell(
#             b, s
#         )
#         # shape = logistic.scale(m)
#         areas = logistic.areas(m, False)
#         assert areas.shape == torch.Size([2, 3, 4])

#     def test_truncate_returns_trapezoid(self):
#         b = ShapeParams(torch.rand(3, 4, 1))
#         s = ShapeParams(torch.rand(3, 4, 1))
#         m = torch.rand(2, 3, 4)
#         logistic = _logistic.LogisticBell(
#             b, s
#         )
#         # shape = logistic.truncate(m)
#         areas = logistic.areas(m, True)
#         assert areas.shape == torch.Size([2, 3, 4])
#         # assert isinstance(shape, _logistic.LogisticTrapezoid)


# # class TestLogisticTrapezoid(object):

# #     def test_join_returns_fuzzy_set_with_correct_size(self):

# #         b = ShapeParams(torch.rand(3, 4, 1))
# #         s = ShapeParams(torch.rand(3, 4, 1))
# #         truncated_m = torch.rand(2, 3, 4)
# #         x = torch.rand(2, 3)
# #         logistic = _logistic.LogisticTrapezoid(
# #             b, s, truncated_m
# #         )
# #         # m = logistic.join(x)
# #         assert m.data.size() == torch.Size([2, 3, 4])

# #     def test_scale_returns_shape_with_correct_size(self):

# #         b = ShapeParams(torch.rand(3, 4, 1))
# #         s = ShapeParams(torch.rand(3, 4, 1))
# #         truncated_m = torch.rand(2, 3, 4)
# #         m = torch.rand(2, 3, 4)
# #         logistic = _logistic.LogisticTrapezoid(
# #             b, s, truncated_m
# #         )
# #         # shape = logistic.scale(m)
# #         assert isinstance(shape, _logistic.LogisticTrapezoid)

# #     def test_mean_core_returns_tensor_with_correct_size(self):

# #         b = ShapeParams(torch.rand(3, 4, 1))
# #         s = ShapeParams(torch.rand(3, 4, 1))
# #         m = torch.rand(2, 3, 4)
# #         truncated_m = torch.rand(2, 3, 4)

# #         logistic = _logistic.LogisticTrapezoid(
# #             b, s, truncated_m
# #         )
# #         shape = logistic.scale(m)
# #         assert shape.mean_cores.shape == torch.Size([2, 3, 4])

# #     def test_centroids_returns_tensor_with_correct_size(self):
# #         b = ShapeParams(torch.rand(3, 4, 1))
# #         s = ShapeParams(torch.rand(3, 4, 1))
# #         truncated_m = torch.rand(2, 3, 4)
# #         m = torch.rand(2, 3, 4)
# #         logistic = _logistic.LogisticTrapezoid(
# #             b, s, truncated_m
# #         )
# #         shape = logistic.scale(m)
# #         assert shape.centroids.shape == torch.Size([2, 3, 4])

# #     def test_areas_returns_tensor_with_correct_size(self):
# #         b = ShapeParams(torch.rand(3, 4, 1))
# #         s = ShapeParams(torch.rand(3, 4, 1))
# #         truncated_m = torch.rand(2, 3, 4)
# #         m = torch.rand(2, 3, 4)
# #         logistic = _logistic.LogisticTrapezoid(
# #             b, s, truncated_m
# #         )
# #         shape = logistic.scale(m)
# #         assert shape.areas.shape == torch.Size([2, 3, 4])

# #     def test_truncate_returns_trapezoid(self):
# #         b = ShapeParams(torch.rand(3, 4, 1))
# #         s = ShapeParams(torch.rand(3, 4, 1))
# #         m = torch.rand(2, 3, 4)
# #         truncated_m = torch.rand(2, 3, 4)

# #         logistic = _logistic.LogisticTrapezoid(
# #             b, s, truncated_m
# #         )
# #         shape = logistic.truncate(m)
# #         assert isinstance(shape, _logistic.LogisticTrapezoid)


# class TestHalfLogistic(object):

#     def test_join_returns_fuzzy_set_with_correct_size(self):

#         b = ShapeParams(torch.rand(3, 4, 1))
#         s = ShapeParams(torch.rand(3, 4, 1))
#         x = torch.rand(2, 3)
#         logistic = _logistic.HalfLogisticBell(
#             b, s, True
#         )
#         m = logistic.join(x)
#         assert m.data.size() == torch.Size([2, 3, 4])

#     # def test_scale_returns_shape_with_correct_size(self):

#     #     b = ShapeParams(torch.rand(3, 4, 1))
#     #     s = ShapeParams(torch.rand(3, 4, 1))
#     #     m = torch.rand(2, 3, 4)
#     #     logistic = _logistic.HalfLogisticBell(
#     #         b, s, True
#     #     )
#     #     areas = logistic.areas(m)
#     #     assert isinstance(areas, _logistic.RightLogistic)

#     def test_mean_core_returns_tensor_with_correct_size(self):

#         b = ShapeParams(torch.rand(3, 4, 1))
#         s = ShapeParams(torch.rand(3, 4, 1))
#         m = torch.rand(2, 3, 4)
#         truncated_m = torch.rand(2, 3, 4)

#         logistic = _logistic.HalfLogisticBell(
#             b, s, True
#         )
#         # shape = logistic.scale(m)

#         mean_cores = logistic.mean_cores(m, False)
#         assert mean_cores.shape == torch.Size([2, 3, 4])

#     def test_centroids_returns_tensor_with_correct_size_scaled(self):
#         b = ShapeParams(torch.rand(3, 4, 1))
#         s = ShapeParams(torch.rand(3, 4, 1))
#         truncated_m = torch.rand(2, 3, 4)
#         m = torch.rand(2, 3, 4)
#         logistic = _logistic.HalfLogisticBell(
#             b, s, True
#         )
#         centroids = logistic.centroids(m, False)
#         # shape = logistic.scale(m)
#         assert centroids.shape == torch.Size([2, 3, 4])

#     def test_centroids_returns_tensor_with_correct_size_truncated(self):
#         b = ShapeParams(torch.rand(3, 4, 1))
#         s = ShapeParams(torch.rand(3, 4, 1))
#         truncated_m = torch.rand(2, 3, 4)
#         m = torch.rand(2, 3, 4)
#         logistic = _logistic.HalfLogisticBell(
#             b, s, True
#         )
#         centroids = logistic.centroids(m, True)
#         # shape = logistic.scale(m)
#         assert centroids.shape == torch.Size([2, 3, 4])

#     def test_areas_returns_tensor_with_correct_size_truncated(self):
#         b = ShapeParams(torch.rand(3, 4, 1))
#         s = ShapeParams(torch.rand(3, 4, 1))
#         truncated_m = torch.rand(2, 3, 4)
#         m = torch.rand(2, 3, 4)
#         logistic = _logistic.HalfLogisticBell(
#             b, s, True
#         )
#         # shape = logistic.scale(m)
#         areas = logistic.areas(m, True)
#         assert areas.shape == torch.Size([2, 3, 4])

#     def test_areas_returns_tensor_with_correct_size_scaled(self):
#         b = ShapeParams(torch.rand(3, 4, 1))
#         s = ShapeParams(torch.rand(3, 4, 1))
#         truncated_m = torch.rand(2, 3, 4)
#         m = torch.rand(2, 3, 4)
#         logistic = _logistic.HalfLogisticBell(
#             b, s, True
#         )
#         # shape = logistic.scale(m)
#         areas = logistic.areas(m, False)
#         assert areas.shape == torch.Size([2, 3, 4])
# #     def test_truncate_returns_trapezoid(self):
# #         b = ShapeParams(torch.rand(3, 4, 1))
# #         s = ShapeParams(torch.rand(3, 4, 1))
# #         m = torch.rand(2, 3, 4)
# #         truncated_m = torch.rand(2, 3, 4)

# #         logistic = _logistic.RightLogistic(
# #             b, s, True, truncated_m
# #         )
# #         shape = logistic.truncate(m)
# #         assert isinstance(shape, _logistic.RightLogisticTrapezoid)


# # class TestRightLogisticTrapezoid(object):

# #     def test_join_returns_fuzzy_set_with_correct_size(self):

# #         b = ShapeParams(torch.rand(3, 4, 1))
# #         s = ShapeParams(torch.rand(3, 4, 1))
# #         truncated_m = torch.rand(2, 3, 4)
# #         x = torch.rand(2, 3)
# #         logistic = _logistic.RightLogisticTrapezoid(
# #             b, s, True, truncated_m
# #         )
# #         m = logistic.join(x)
# #         assert m.data.size() == torch.Size([2, 3, 4])

# #     def test_scale_returns_shape_with_correct_size(self):

# #         b = ShapeParams(torch.rand(3, 4, 1))
# #         s = ShapeParams(torch.rand(3, 4, 1))
# #         truncated_m = torch.rand(2, 3, 4)
# #         m = torch.rand(2, 3, 4)
# #         logistic = _logistic.RightLogisticTrapezoid(
# #             b, s, True, truncated_m
# #         )
# #         shape = logistic.scale(m)
# #         assert isinstance(shape, _logistic.RightLogisticTrapezoid)

# #     def test_mean_core_returns_tensor_with_correct_size(self):

# #         # b = ShapeParams(torch.rand(3, 4, 1))
# #         # s = ShapeParams(torch.rand(3, 4, 1))
# #         # m = torch.rand(2, 3, 4)
# #         # truncated_m = torch.rand(2, 3, 4)

# #         # logistic = _logistic.RightLogisticTrapezoid(
# #         #     b, s, True, truncated_m
# #         # )
# #         # pprint(type(logistic))
# #         # shape = logistic.scale(m)
# #         # assert shape.mean_cores.shape == torch.Size([2, 3, 4])

# #         b = ShapeParams(torch.rand(3, 4, 1))
# #         s = ShapeParams(torch.rand(3, 4, 1))
# #         truncated_m = torch.rand(2, 3, 4)
# #         m = torch.rand(2, 3, 4)
# #         logistic = _logistic.RightLogisticTrapezoid(
# #             b, s, True, truncated_m
# #         )
# #         shape = logistic.scale(m)
# #         assert shape.mean_cores.shape == torch.Size([2, 3, 4])


# #     def test_centroids_returns_tensor_with_correct_size(self):
# #         b = ShapeParams(torch.rand(3, 4, 1))
# #         s = ShapeParams(torch.rand(3, 4, 1))
# #         truncated_m = torch.rand(2, 3, 4)
# #         m = torch.rand(2, 3, 4)
# #         logistic = _logistic.RightLogisticTrapezoid(
# #             b, s, True, truncated_m
# #         )
# #         shape = logistic.scale(m)
# #         assert shape.centroids.shape == torch.Size([2, 3, 4])

# #     def test_areas_returns_tensor_with_correct_size(self):
# #         b = ShapeParams(torch.rand(3, 4, 1))
# #         s = ShapeParams(torch.rand(3, 4, 1))
# #         truncated_m = torch.rand(2, 3, 4)
# #         m = torch.rand(2, 3, 4)
# #         logistic = _logistic.RightLogisticTrapezoid(
# #             b, s, True, truncated_m
# #         )
# #         shape = logistic.scale(m)
# #         assert shape.areas.shape == torch.Size([2, 3, 4])

# #     def test_truncate_returns_trapezoid(self):
# #         b = ShapeParams(torch.rand(3, 4, 1))
# #         s = ShapeParams(torch.rand(3, 4, 1))
# #         m = torch.rand(2, 3, 4)
# #         truncated_m = torch.rand(2, 3, 4)

# #         logistic = _logistic.RightLogisticTrapezoid(
# #             b, s, True, truncated_m
# #         )
# #         shape = logistic.truncate(m)
# #         assert isinstance(shape, _logistic.RightLogisticTrapezoid)
