# from ._fuzzy_assess import (
#     FuzzyAggregatorLoss, FuzzyLoss, IntersectionOnLoss, UnionOnLoss, 
#     MaxMinLoss, MaxMinLoss2, MaxMinLoss3, MaxProdLoss, MinMaxLoss2, MinMaxLoss3,
#     NeuronMSELoss
# )
# from ._core import (
#     ToOptim, MistifyLoss, MistifyLossFactory
# )
# from ._learn import (
#     PreFit, PostFit, OrLearner, AndLearner
# )
from ._infer import (
    MaxMinLoss, MaxMinPredictorLoss, MaxMinSortedPredictorLoss,
    MinMaxLoss, MinMaxPredictorLoss, MinMaxSortedPredictorLoss,
    IntersectionOnLoss, UnionOnLoss
)
