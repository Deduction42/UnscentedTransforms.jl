module UnscentedTransforms
    include("_NewSigmaPoints.jl")
    include("_AbstractPredictor.jl")
    include("_StateSpaceModel.jl")
    
    export StateSpaceModel, LinearPredictor, NonlinearPredictor, MvGaussian, SigmaPoints, SigmaParams, SigmaWeights
    export kalman_filter!, predict, predict!, update, update!
end
