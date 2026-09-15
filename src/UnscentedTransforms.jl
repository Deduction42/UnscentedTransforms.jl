module UnscentedTransforms
    include("_SigmaPoints.jl")
    include("_AbstractPredictor.jl")
    include("_StateSpaceModel.jl")
    include("math.jl")
    
    export StateSpaceModel, LinearPredictor, NonlinearPredictor, MvGaussian, SigmaPoints, SigmaParams, SigmaWeights
    export kalman_filter!, predict, predict!, update, update!
end
