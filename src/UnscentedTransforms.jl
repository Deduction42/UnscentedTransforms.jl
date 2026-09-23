module UnscentedTransforms
    include("_SigmaPoints.jl")
    include("_StateSpaceModel.jl")
    include("math.jl")
    include("domainlimits.jl")
    
    export StateSpaceModel, LinearPredictor, NonlinearPredictor, MvGaussian, UvGaussian, SigmaPoints, SigmaParams, SigmaWeights, ArgLimits
    export kalman_filter!, predict, predict!, update, update!, gaussian, ±, scale_spread, correlated
end
