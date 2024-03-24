from ._neurons import (
    Or, And, LogicalNeuron, WEIGHT_FACTORY,
    BuildAnd, BuildLogical, BuildOr,
    MaxMin, MaxProd, MinMax, MinSum
)
from ._noise import DropoutNoise, ExpNoise, GaussianClampNoise
from ._shape import swap, expand_term, collapse_term
from ._ops import (
    UnionOn, JunctionOn, IntersectionOn, 
    Complement, CatComplement, CatElse, 
    Else
)
from ._activate import (
    MembershipActivation, Descale,
    Sigmoidal, Triangular, Hedge
)
