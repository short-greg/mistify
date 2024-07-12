.. _api:


API Reference
=============

mistify (core)
-------------

.. autosummary::
   :toctree: generated

   mistify.AndF
   mistify.OrF
   mistify.Union
   mistify.Inter
   mistify.InterOn
   mistify.UnionOn
   mistify.heaviside
   mistify.to_boolean
   mistify.signify
   mistify.to_signed
   mistify.clamp
   mistify.ramp
   mistify.threshold
   mistify.G
   mistify.ClipG
   mistify.AllG
   mistify.ZeroG
   mistify.SignG
   mistify.HeavisideG
   mistify.ClampG
   mistify.MaxOnG
   mistify.MinOnG
   mistify.BindG
   mistify.MulG
   mistify.inter
   mistify.inter_on
   mistify.ada_inter
   mistify.ada_inter_on
   mistify.prob_inter
   mistify.prob_inter_on
   mistify.smooth_inter
   mistify.smooth_inter_on
   mistify.bounded_inter
   mistify.bounded_inter_on
   mistify.union
   mistify.union_on
   mistify.ada_union
   mistify.ada_union_on
   mistify.prob_union
   mistify.prob_union_on
   mistify.smooth_union
   mistify.smooth_union_on
   mistify.bounded_union
   mistify.bounded_union_on
   mistify.max_min
   mistify.min_max
   mistify.ada_max_min
   mistify.ada_min_max
   mistify.max_prod
   mistify.min_sum
   mistify.shape

mistify.fuzzify
-------------

.. autosummary::
   :toctree: generated

   mistify.fuzzify.HypoWeight
   mistify.fuzzify.MaxConc
   mistify.fuzzify.Conclusion
   mistify.fuzzify.MaxValueConc
   mistify.fuzzify.WeightedMAverageConc
   mistify.fuzzify.WeightedPAverageConc
   mistify.fuzzify.ConcEnum
   mistify.fuzzify.FlattenConc
   mistify.fuzzify.var_normalize
   mistify.fuzzify.Fuzzifier
   mistify.fuzzify.EmbeddingFuzzifier
   mistify.fuzzify.Defuzzifier
   mistify.fuzzify.HypothesisEnum
   mistify.fuzzify.AreaHypothesis
   mistify.fuzzify.ShapeHypothesis
   mistify.fuzzify.CentroidHypothesis
   mistify.fuzzify.MeanCoreHypothesis
   mistify.fuzzify.MinCoreHypothesis
   mistify.fuzzify.stride_coordinates
   mistify.fuzzify.ShapeFuzzifier
   mistify.fuzzify.LogisticFuzzifier
   mistify.fuzzify.FuzzifierDecorator
   mistify.fuzzify.FuncFuzzifierDecorator
   mistify.fuzzify.TriangleFuzzifier
   mistify.fuzzify.IsoscelesFuzzifier
   mistify.fuzzify.TrapezoidFuzzifier
   mistify.fuzzify.IsoscelesTrapezoidFuzzifier
   mistify.fuzzify.SigmoidFuzzifier
   mistify.fuzzify.RampFuzzifier
   mistify.fuzzify.Coords
   mistify.fuzzify.Shape
   mistify.fuzzify.Polygon
   mistify.fuzzify.Nonmonotonic
   mistify.fuzzify.Monotonic
   mistify.fuzzify.Square
   mistify.fuzzify.Logistic
   mistify.fuzzify.LogisticBell
   mistify.fuzzify.HalfLogisticBell
   mistify.fuzzify.Gaussian
   mistify.fuzzify.GaussianBell
   mistify.fuzzify.HalfGaussianBell
   mistify.fuzzify.Trapezoid
   mistify.fuzzify.IsoscelesTrapezoid
   mistify.fuzzify.RightTrapezoid
   mistify.fuzzify.Triangle
   mistify.fuzzify.IsoscelesTriangle
   mistify.fuzzify.RightTriangle
   mistify.fuzzify.shape_utils
   mistify.fuzzify.Composite
   mistify.fuzzify.Sigmoid
   mistify.fuzzify.Ramp
   mistify.fuzzify.Step
   mistify.fuzzify.shape_utils
   mistify.fuzzify.resp
   mistify.fuzzify.resp_average


mistify.infer
-------------

.. autosummary::
   :toctree: generated

   mistify.infer.Or
   mistify.infer.And
   mistify.infer.WEIGHT_FACTORY
   mistify.infer.MinMax
   mistify.infer.MaxMin
   mistify.infer.MinSum
   mistify.infer.MaxProd
   mistify.infer.SmoothMinMax
   mistify.infer.SmoothMaxMin
   mistify.infer.WeightF
   mistify.infer.NullWeightF
   mistify.infer.SignWeightF
   mistify.infer.Sub1WeightF
   mistify.infer.ClampWeightF
   mistify.infer.BooleanWeightF
   mistify.infer.validate_binary_weight
   mistify.infer.validate_weight_range
   mistify.infer.SigmoidWeightF
   mistify.infer.LogicalNeuron
   mistify.infer.DropoutNoise
   mistify.infer.ExpNoise
   mistify.infer.GaussianClampNoise
   mistify.infer.swap
   mistify.infer.expand_term
   mistify.infer.collapse_term
   mistify.infer.UnionOnBase
   mistify.infer.InterOnBase
   mistify.infer.Complement
   mistify.infer.CatComplement
   mistify.infer.CatElse
   mistify.infer.Else
   mistify.infer.Union
   mistify.infer.UnionOn
   mistify.infer.ProbInter
   mistify.infer.ProbUnion
   mistify.infer.UnionBase
   mistify.infer.ProbUnionOn
   mistify.infer.SmoothUnion
   mistify.infer.SmoothUnionOn
   mistify.infer.BoundedUnion
   mistify.infer.BoundedUnionOn
   mistify.infer.Inter
   mistify.infer.InterBase
   mistify.infer.InterOn
   mistify.infer.ProbInterOn
   mistify.infer.SmoothInter
   mistify.infer.SmoothInterOn
   mistify.infer.BoundedInter
   mistify.infer.BoundedInterOn
   mistify.infer.MembershipAct
   mistify.infer.Descale
   mistify.infer.Sigmoidal
   mistify.infer.Triangular
   mistify.infer.Hedge
