module UnscentedTransforms
    include("_SigmaPoints.jl")
    include("_AbstractPredictor.jl")
    include("_StateSpaceModel.jl")
    
    export StateSpaceModel, LinearPredictor, NonlinearPredictor, GaussianVar, SigmaPoints, SigmaParams, SigmaWeights
    export kalman_filter!, predict, predict!, update, update!
end
