abstract type AbstractPredictor end 

"""
LinearPredictor(A::AbstractMatrix, B::AbstractMatrix, Σ::Cholesky)

Linear predictor with added noise covariance Σ (internally a Cholesky factorization is applied)
Prediction is given by 
y = Ax + Bu + ε 
where x and u are vectors and ε is white noise
"""
@kwdef struct LinearPredictor{TA<:AbstractArray, TB<:AbstractArray, TΣ<:Cholesky} <: AbstractPredictor
    A :: TA 
    B :: TB 
    Σ :: TΣ
end
LinearPredictor(A::AbstractMatrix, B::AbstractMatrix, Σ::AbstractMatrix) = LinearPredictor(A, B, cholesky(Σ))


"""
NonlinearPredictor(F::Function, Σ::Cholesky, θ::SigmaParams, multithreaded)

Nonlinear predictor with added noise covariance Σ (internally a Cholesky factorization is applied)
Prediction is given by 
y = f(x, u) + ε 
where x is a vector (u can be any object) and ε is white noise
"""
@kwdef struct NonlinearPredictor{TF<:Function, TΣ<:Cholesky} <: AbstractPredictor
    f :: TF
    Σ :: TΣ
    θ :: SigmaParams
    multithreaded :: Bool = false
end
NonlinearPredictor(f::Function, Σ::AbstractMatrix, θ::SigmaParams) = NonlinearPredictor(f, cholesky(Σ), θ)

const StatePredictor = Union{LinearPredictor, NonlinearPredictor}

#=======================================================================================================================
Views on predictors (enables removal of missing observations with minimal allocation)
=======================================================================================================================#
function Base.view(pred::LinearPredictor, inds)
    return LinearPredictor(
        view(pred.A, inds, :),
        view(pred.B, inds, :),
        covview(pred.Σ, inds)
    )
end

function Base.view(pred::NonlinearPredictor, inds; θ=pred.θ)
    fv(x, u) = view(pred.f(x, u), inds)
    return NonlinearPredictor(fv, covview(pred.Σ, inds), θ)
end


"""
StateSpaceModel(state<:MvGaussian, predictor<:StatePredictor, observer<:StatePredictor, outlier=Inf)

A generic state-space model with a Gaussian state, but with potential linear/nonlinear predictors
Applying a Kalman filter will use the square-root UKF for nonlinear predictors/observers while
the original Kalman filter will be applied to any linear predictors/observers. 

This version also has an 'outlier' parameter that increases robustness against a certain outlier size.
For example, 'outlier = 6' penalizes prediction errors if they are beyond 6 standard deviations
away from zero (note that this standard deviation factors both prediction and measurement error).
Conventional Kalman filter behavior occurs when 'outlier=Inf'
"""
Base.@kwdef mutable struct StateSpaceModel{TX<:MvGaussian, TP<:StatePredictor, TO<:StatePredictor}
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

    elseif !any(isfinite, y)
        @warn "No observations, skipping update"
        nothing

    else
        ind = findall(isfinite, y)
        obs_view = view(ss.observer, ind)
        y_view = view(y, ind)

        update(obs_view, ss.state, y_view, u)
    end

    if isnothing(results)
        return nothing
    elseif isfinite(results.X)
        ss.state = results.X
        return results
    else
        @warn "Non-finite state result, update not applied"
        return results
    end
end



"""
covview(ch::Cholesky, inds) = Cholesky(view(ch.U, inds, inds), :U, 0)

View of a cholesky decomposition of a covariance matrix subspace defined by 'inds'
"""
covview(ch::Cholesky, inds) = Cholesky(UpperTriangular(view(ch.U, inds, inds)))

#=======================================================================================================================
Prediction functions (uncertainty propagation)
=======================================================================================================================#

#Linear predictors
function predict(pred::LinearPredictor, X::MvGaussian, u)
    xh = pred.A*X.μ + pred.B*u 
    Σh = add_lcov(pred.Σ, pred.A*std(X).L)
    return MvGaussian(xh, Σh)
end

function predict_similar(pred::LinearPredictor, X::MvGaussian{T}, u) where T
    xh = T(pred.A*X.μ + pred.B*u)
    Σh = add_lcov(pred.Σ, pred.A*std(X).L)
    return MvGaussian(xh, Σh)
end

#Nonlinar predictors (returns the same type as X)
function predict(pred::NonlinearPredictor, X::MvGaussian, u)
    return predict(pred, SigmaPoints(pred.θ, X), u) + MvGaussian(pred.Σ)
end

function predict(pred::NonlinearPredictor, X::SigmaPoints, u)
    f(x) = pred.f(x, u)
    f_task(x) = Threads.@spawn(pred.f(x,u))

    if pred.multithreaded
        return SigmaPoints(X.weights, fetch.(map(f_task, X)))
    else
        return SigmaPoints(X.weights, map(f, X))
    end
end

function predict_similar(pred::NonlinearPredictor, X::MvGaussian, u)
    return predict(pred, SigmaPoints(pred.θ, X), u) + MvGaussian(pred.Σ)
end



#=======================================================================================================================
Update functions (Kalman-Update)
=======================================================================================================================#
function update(obs::LinearPredictor, X::MvGaussian{Tμ,TΣ}, y::AbstractVector, u; outlier=Inf) where {Tμ, TΣ} 
    (C, D, R, P) = (obs.A, obs.B, obs.Σ, std(X))
    yh = C*X.μ .+ D*u

    S = add_lcov(R, C*P.L) #Innovation covariance
    Z = MvGaussian(y.-yh, S) #Innovation distribution
    Pxy = (P.L*P.U)*C'#Obtain cross-covariance of state and measurement innovations
    K = (Pxy/S.U)/S.L #Kalman gain

    #Scale the gain based off outliers
    outlier_scaling!(K, Z, outlier)

    #Update the posterior
    μ = X.μ .+ K*Z.μ

    #This sometimes fails because rounding error on subtraction makes the matrix "negative", skip if that happens
    Σ = try
        sub_lcov(std(X), K*S.L)
    catch err
        @warn "Covariance update failed, skipping this step:\n" * sprint(showerror, err)
        X.Σ
    end

    return (X=MvGaussian(Tμ(μ), TΣ(Σ)), Y=MvGaussian(yh, S), K=K)
end


function update(obs::NonlinearPredictor, X::MvGaussian{Tμ,TΣ}, y::AbstractVector, u; outlier=Inf) where {Tμ, TΣ}
    #Build the sigma points from the Gaussian variable
    Xp = SigmaPoints(obs.θ, X)

    #Propagate the sigma points through the predictor
    Yp = predict(obs, Xp, u)
    Y  = Yp + MvGaussian(obs.Σ) #Predicted Y distribution
    Z  = MvGaussian(y .- mean(Y), std(Y)) #Innovation distribution

    S   = std(Z) #Innovation covariance
    Pxy = cov(Xp, Yp) #Obtain cross-covariance of state and measurement innovations
    K   = (Pxy/S.U)/S.L #Kalman gain

    #Scale the gain based off outliers
    outlier_scaling!(K, Z, outlier)

    #Update the posterior
    μ = X.μ .+ K*Z.μ
    Σ = try
        sub_lcov(std(X), K*S.L)
    catch err
        @warn "Covariance update failed, skipping this step:\n" * sprint(showerror, err)
        X.Σ
    end

    return (X=MvGaussian(Tμ(μ), TΣ(Σ)), Y=Y, K=K)
end

#Scale the gain based on outlier score of the prediction error distribution ΔY
function outlier_scaling!(K::AbstractArray, ΔY::MvGaussian, cutoff::Real)
    for (ii, Δy) in enumerate(ΔY.μ)
        z = Δy/std(ΔY, ii)
        K[:,ii] .= K[:,ii].*outlier_scale(z, cutoff)
    end
    return K 
end

#A scaling factor that yields a linear curve for z<=cutoff, and logarithmic for z>=cutoff
#The curve is continyous up to the 2nd derivative
function outlier_scale(z::Real, cutoff::Real)
    rz = abs(z)/cutoff
    return ifelse(rz<=1, one(rz), (1+log(rz))/rz)
end



