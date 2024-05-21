from ._neurons import (
    Or, And, WEIGHT_FACTORY,
    MinMax, MaxMin, MinSum, MaxProd,
    SmoothMinMax, SmoothMaxMin,
    WeightF, NullWeightF, SignWeightF, Sub1WeightF, ClampWeightF,
    BooleanWeightF, 
    validate_binary_weight, validate_weight_range,
    SigmoidWeightF, LogicalNeuron
)
from ._noise import DropoutNoise, ExpNoise, GaussianClampNoise
from ._shape import swap, expand_term, collapse_term
from ._ops import (
    UnionOnBase, InterOnBase, 
    Complement, CatComplement, CatElse, 
    Else, 
    Union, UnionOn, ProbInter,
    ProbUnion, UnionBase, ProbUnionOn, SmoothUnion,
    SmoothUnionOn, BoundedUnion, BoundedUnionOn,
    Inter, InterBase, InterOn, ProbInterOn, SmoothInter,
    SmoothInterOn, BoundedInter, BoundedInterOn,
)
from ._activate import (
    MembershipAct, Descale,
    Sigmoidal, Triangular, Hedge
)
