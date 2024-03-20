import torch
from mistify._functional import (
    fuzzy, boolean, signed, _set_ops as set_ops
)

class TestSetOps:

    def test_differ_is_greater_than_zero_for_all(self):
        
        c1 = fuzzy.rand(2, 3, 2, 4)
        c2 = fuzzy.rand(2, 3, 2, 4)
        assert ((set_ops.differ(c1, c2)).data >= 0.0).all()

    def test_differ_is_included_in_tensor(self):
        
        c1 = fuzzy.rand(2, 3, 2, 4)
        c2 = fuzzy.rand(2, 3, 2, 4)
        assert (set_ops.inclusion(fuzzy._functional.differ(c1, c2), c1).data == 1.0).all()

    def test_differ_is_greater_than_zero_for_all(self):
        
        c1 = boolean.rand(2, 3, 2, 4)
        c2 = boolean.rand(2, 3, 2, 4)
        assert ((set_ops.differ(c1, c2)).data >= 0.0).all()

    def test_differ_is_included_in_tensor(self):
        
        c1 = boolean.rand(2, 3, 2, 4)
        c2 = boolean.rand(2, 3, 2, 4)
        c2 = boolean.differ(c1, c2)
        assert (set_ops.inclusion(c2, c1).data == 1.0).all()

    def test_differ_is_greater_than_zero_for_all(self):
        
        c1 = signed.rand(2, 3, 2, 4)
        c2 = signed.rand(2, 3, 2, 4)
        assert ((set_ops.signed_differ(c1, c2)).data >= -1.0).all()

    def test_differ_is_included_in_tensor(self):
        
        c1 = signed.rand(2, 3, 2, 4)
        c2 = signed.rand(2, 3, 2, 4)
        assert (set_ops.inclusion(set_ops.signed_differ(c1, c2), c1).data == 1.0).all()

    # def test_union_results_in_all_values_being_greater_or_same(self):
        
    #     torch.manual_seed(1)
    #     c1 = boolean.rand(2, 3, 2, 4)
    #     c2 = boolean.rand(2, 3, 2, 4)
    #     c3 = boolean.unify(c1, c2)
    #     assert (c3.data >= c2.data).all()

    # def test_union_results_in_all_values_being_greater_or_same(self):
        
    #     torch.manual_seed(1)
    #     c1 = signed.rand(2, 3, 2, 4)
    #     c2 = signed.rand(2, 3, 2, 4)
    #     c3 = signed._functional.unify(c1, c2)
    #     assert (c3.data >= c2.data).all()

    # def test_intersect_results_in_all_values_being_less_or_same(self):
    #     torch.manual_seed(1)
    #     c1 = signed.rand(2, 3, 2, 4)
    #     c2 = signed.rand(2, 3, 2, 4)
    #     c3 = signed._functional.intersect(c1, c2)
    #     assert (c3 <= c2).all()

    # def test_intersect_is_included_in_the_tensor(self):
        
    #     c1 = signed.rand(2, 3, 2, 4)
    #     c2 = signed.rand(2, 3, 2, 4) * c1
    #     assert (signed._functional.inclusion(c2, c1) == 1).all()

    # def test_union_is_excluded_in_the_tensor(self):
        
    #     c1 = signed.rand(2, 3, 2, 4)
    #     c2 = signed.rand(2, 3, 2, 4) + c1
    #     assert (signed._functional.exclusion(c2, c1).data == 1).all()
    
    # def test_intersect_results_in_all_values_being_less_or_same(self):
    #     torch.manual_seed(1)
    #     c1 = boolean.rand(2, 3, 2, 4)
    #     c2 = boolean.rand(2, 3, 2, 4)
    #     c3 = boolean._functional.intersect(c1, c2)
    #     assert (c3 <= c2).all()

    # def test_intersect_is_included_in_the_tensor(self):
        
    #     c1 = boolean.rand(2, 3, 2, 4)
    #     c2 = torch.min(boolean.rand(2, 3, 2, 4), c1)
    #     assert (boolean.inclusion(c2, c1) == 1).all()

    # def test_union_is_excluded_in_the_tensor(self):
        
    #     c1 = boolean.rand(2, 3, 2, 4)
    #     c2 = torch.max(boolean.rand(2, 3, 2, 4), c1)
    #     assert (boolean.exclusion(c2, c1).data == 1).all()
    
    # def test_intersect_results_in_all_values_being_less_or_same(self):
    #     torch.manual_seed(1)
    #     c1 = fuzzy.rand(2, 3, 2, 4)
    #     c2 = fuzzy.rand(2, 3, 2, 4)
    #     c3 = fuzzy._functional.intersect(c1, c2)
    #     assert (c3 <= c2).all()

    # def test_intersect_is_included_in_the_tensor(self):
        
    #     c1 = fuzzy.rand(2, 3, 2, 4)
    #     c2 = fuzzy.rand(2, 3, 2, 4) * c1
    #     assert (fuzzy._functional.inclusion(c2, c1) == 1).all()

    # def test_union_is_excluded_in_the_tensor(self):
        
    #     c1 = fuzzy.rand(2, 3, 2, 4)
    #     c2 = fuzzy.rand(2, 3, 2, 4) + c1
    #     assert (fuzzy._functional.exclusion(c2, c1).data == 1).all()
    
    # def test_union_results_in_all_values_being_greater_or_same(self):
        
    #     torch.manual_seed(1)
    #     c1 = fuzzy.rand(2, 3, 2, 4)
    #     c2 = fuzzy.rand(2, 3, 2, 4)
    #     c3 = set_ops.unify(c1, c2)
    #     assert (c3.data >= c2.data).all()
