#noqa

from . import boolean
from . import signed
from . import fuzzy

from ._m import (
    binary, to_binary, sign, 
    to_signed, ramp
)
from ._grad import (
    SignG, BinaryG,
    MaxG, MinG, ClampG, MaxOnG, MinOnG
)
from ._join import (
    inter, inter_on, ada_inter,
    ada_inter_on, prob_inter, prob_inter_on,
    smooth_inter, smooth_inter_on, bounded_inter,
    bounded_inter_on,
    union, union_on, ada_union, ada_union_on,
    prob_union, prob_union_on, smooth_union, smooth_union_on,
    bounded_union, bounded_union_on
)

from ._logic import (
    or_, and_, 
    ada_or, ada_and,
    or_prod

)
from ._factory import (
    AndF, OrF,
    Union, UnionOn, Inter, InterOn
)
from ._shape import (
    triangle, trapezoid, isosceles, isosceles_trapezoid
)
