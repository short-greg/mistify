#noqa

# from ._inference import (
#     FuzzyComplement, FuzzyIntersectionOn, FuzzyElse, FuzzyAnd,
#     FuzzyUnionOn, FuzzyOr
# )
from ._generate import (
    rand, positives, negatives
)
from . import _utils as utils
from ._functional import (
    differ, intersect, intersect_on,
    unify, unify_on, intersect_on, exclusion,
    else_, complement
)
