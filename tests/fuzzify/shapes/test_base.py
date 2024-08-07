from mistify import fuzzify
from mistify.fuzzify._shapes._base import replace, insert
import torch


class TestShapeParams:

    def test_x_property_equals_x(self):

        x = torch.rand(2, 3, 4, 5).cumsum(-1)
        params = fuzzify.Coords(x)
        assert torch.isclose(params(), x).all()

    def test_x_property_unsqueezes_x_when_dim_is_3(self):

        x = torch.rand(3, 4, 5).cumsum(-1)
        params = fuzzify.Coords(x)
        assert torch.isclose(params(), x[None]).all()

    def test_n_variables_is_correct_value(self):
        x = torch.rand(2, 3, 4, 5).cumsum(-1)
        params = fuzzify.Coords(x)
        assert params.n_vars == 3

    def test_n_terms_is_correct_value(self):
        x = torch.rand(2, 3, 4, 5).cumsum(-1)
        params = fuzzify.Coords(x)
        assert params.n_terms == 4
    
    def test_batch_size_is_correct_value(self):
        x = torch.rand(2, 3, 4, 5).cumsum(-1)
        params = fuzzify.Coords(x)
        assert params.batch_size == 2

    def test_n_points_is_correct_value(self):
        x = torch.rand(2, 3, 4, 5).cumsum(-1)
        params = fuzzify.Coords(x)
        assert params.n_points == 5
    
    def test_contains_returns_true_if_inside(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x2 = x[:,:,:,2] * 0.5 + x[:,:,:,3] * 0.5
        params = fuzzify.Coords(x)
        assert params.contains(x2, 2, 3).all()

    def test_contains_returns_false_if_notinside(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x2 = x[:,:,:,3] * 0.5 + x[:,:,:,4] * 0.5
        params = fuzzify.Coords(x)
        assert not params.contains(x2, 2, 3).any()


class TestInsert:

    def test_insert_inserts_into_params(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x1 = x[:,:,:,:4]
        x2 = x[:,:,:,4]
        params = fuzzify.Coords(x1)
        inserted = insert(params(), x2, 4, to_unsqueeze=True)
        assert (inserted[:,:,:,4] == x2).all()

    def test_insert_inserts_into_start_of_params(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x1 = x[:,:,:,:4]
        x2 = x[:,:,:,4]
        params = fuzzify.Coords(x1)
        inserted = insert(params(), x2, 0, to_unsqueeze=True)
        assert (inserted[:,:,:,0] == x2).all()

    def test_insert_inserts_into_end_of_params(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x1 = x[:,:,:,:4]
        x2 = x[:,:,:,4]
        params = fuzzify.Coords(x1)
        inserted = insert(params(), x2, 2, to_unsqueeze=True)
        assert (inserted[:,:,:,2] == x2).all()


class TestReplace:

    def test_replace_inserts_into_start_of_params(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x1 = x[:,:,:,:4]
        x2 = x[:,:,:,4]
        params = fuzzify.Coords(x1)
        inserted = replace(params(), x2, 0, to_unsqueeze=True)
        assert (inserted[...,0] == x2).all() and inserted.size(3) == 4

    def test_replace_inserts_into_end_of_params(self):
        x = torch.rand(2, 3, 4, 5).cumsum(3)
        x1 = x[:,:,:,:4]
        x2 = x[:,:,:,4]
        params = fuzzify.Coords(x1)
        inserted = replace(params(), x2, 3, to_unsqueeze=True)
        assert (inserted[...,3] == x2).all() and inserted.size(3) == 4

    # def test_replace_raises_error_if_outside_bounds(self):
    #     x = torch.rand(2, 3, 4, 5).cumsum(3)
    #     x1 = x[:,:,:,:4]
    #     x2 = x[:,:,:,4]
    #     params = fuzzify.Coords(x1)
    #     with pytest.raises(ValueError):
    #         replace(params(), x2, 4, to_unsqueeze=True)

