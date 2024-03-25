from ._neurons import (
    Or, And, LogicalNeuron, WEIGHT_FACTORY,
    BuildAnd, BuildLogical, BuildOr,
    MaxMin, MaxProd, MinMax, MinSum,
    validate_binary_weight, validate_weight_range
)
from ._noise import DropoutNoise, ExpNoise, GaussianClampNoise
from ._shape import swap, expand_term, collapse_term
from ._ops import (
    UnionOn, JunctionOn, IntersectionOn, 
    Complement, CatComplement, CatElse, 
    Else
)
from ._activate import (
    MembershipAct, DescaleAct,
    SigmoidalAct, TriangularAct, HedgeAct
)
