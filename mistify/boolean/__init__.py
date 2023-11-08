# from ._conversion import (
#     CrispConverter, Crispifier, StepCrispConverter,
#     ConverterCrispifier, ConverterDecrispifier
# )
# from ._membership import (
#     Square
# )
from ._inference import (
    BooleanAnd, BooleanComplement, BooleanElse, BooleanIntersectionOn,
    BooleanOr, BooleanUnionOn, binary_func
)
from ._generate import (
    rand, positives, negatives
)
from ._functional import (
    differ, unify, intersect, intersect_on, unify_on, inclusion,
    exclusion, complement, forget, else_
)
# from . import _utils