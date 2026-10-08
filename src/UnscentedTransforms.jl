module UnscentedTransforms
    include("gaussian.jl")
    include("sigma_points.jl")
    include("domainlimits.jl")
    include("math.jl")
    include("kalman_filtering.jl")
    
    export StateSpaceModel, LinearPredictor, NonlinearPredictor, MvGaussian, UvGaussian, SigmaPoints, SigmaParams, SigmaWeights, ArgLimits
    export kalman_filter!, predict, predict!, update, update!, gaussian, ±, scale_weights, correlated
end
