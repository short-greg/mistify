from ._neurons import (
    Or, And, LogicalNeuron, WEIGHT_FACTORY
)
from ._noise import DropoutNoise, ExpNoise, GaussianClampNoise
from ._shape import swap, expand_term, collapse_term
from ._ops import (
    UnionOn, JunctionOn, IntersectionOn, 
    Complement, CatComplement, CatElse, 
    Else
)
