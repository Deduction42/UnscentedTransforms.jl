"""
StateSpaceModel(state<:GaussianVar, predictor<:StatePredictor, observer<:StatePredictor, outlier=Inf)

A generic state-space model with a Gaussian state, but with potential linear/nonlinear predictors
Applying a Kalman filter will use the square-root UKF for nonlinear predictors/observers while
the original Kalman filter will be applied to any linear predictors/observers. 

This version also has an 'outlier' parameter that increases robustness against a certain outlier size.
For example, 'outlier = 6' penalizes prediction errors if they are beyond 6 standard deviations
away from zero (note that this standard deviation factors both prediction and measurement error).
Conventional Kalman filter behavior occurs when 'outlier=Inf'
"""
Base.@kwdef mutable struct StateSpaceModel{TX<:GaussianVar, TP<:StatePredictor, TO<:StatePredictor}
    state :: TX
    predictor :: TP 
    observer  :: TO
    outlier :: Float64 = Inf64
end 

function kalman_filter!(ss::StateSpaceModel, y::AbstractVector, u)
    predict!(ss, u)
    return update!(ss, y, u)
end


"""
predict!(ss::StateSpaceModel, u)

Predicts the state using the state-space model's predictor, and writes the predicted 
value back to the state.
"""
function predict!(ss::StateSpaceModel, u)
    x = predict_similar(ss.predictor, ss.state, u)

    if isfinite(x)
        ss.state = x 
    else
        @warn "Non-finite state result, prediction not applied"
    end

    return x
end

"""
update!(ss::StateSpaceModel, y::AbstractVector, u)

Uses the observation 'y' to perform a Kalman update on the state. If any elements of 
y are not finite, an update is performed on a reduced-dimension observer (where non-finite 
elements of y are ignored)
"""
function update!(ss::StateSpaceModel, y::AbstractVector, u)
    results = if all(isfinite, y)
        update(ss.observer, ss.state, y, u, outlier=ss.outlier)
    else
        ind = findall(isfinite, y)
        obs_view = view(ss.observer, ind)
        y_view = view(y, ind)

        update(obs_view, ss.state, y_view, u)
    end

    if isfinite(results.X)
        ss.state = results.X
    else
        "Non-finite state result, observation not applied"
    end

    return results 
end






