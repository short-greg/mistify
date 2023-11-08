from ._functional import (
    differ, unify, intersect, unify_on, intersect_on,
    inclusion, exclusion, complement, forget
)
from ._generate import (
    negatives, positives, unknowns, rand
)
from ._inference import (
    SignedAnd, SignedComplement, SignedElse, 
    SignedIntersectionOn, SignedOr, SignedUnionOn
)
