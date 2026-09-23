abstract type AbstractPredictor end 

"""
LinearPredictor(A::AbstractMatrix, B::AbstractMatrix, ε::MvGaussian)

Linear predictor with added noise `ε`
Prediction is given by 
```
y = Ax + Bu + ε 
```
"""
@kwdef struct LinearPredictor{TA<:AbstractArray, TB<:AbstractArray, Tε<:MvGaussian} <: AbstractPredictor
    A :: TA 
    B :: TB 
    ε :: Tε
end
LinearPredictor(A::AbstractMatrix, B::AbstractMatrix, Σ::AbstractMatrix) = LinearPredictor(A, B, MvGaussian(Σ))

"""
NonlinearPredictor(f::Function, ε::MvGaussian, θ::SigmaParams, multithreaded)

Nonlinear predictor `f` with added noise `ε`
Prediction is given by 
```
y = f(x, u) + ε 
```
"""
@kwdef struct NonlinearPredictor{TF, Tε<:MvGaussian} <: AbstractPredictor
    f :: TF
    ε :: Tε
    θ :: SigmaParams
    multithreaded :: Bool = false
end
NonlinearPredictor(f, Σ::AbstractMatrix, θ::SigmaParams, multithreaded::Bool) = NonlinearPredictor(f, MvGaussian(Σ), θ, multithreaded)

const StatePredictor = Union{LinearPredictor, NonlinearPredictor}


"""
VecFuncView(f::Function, inds)

An object that behaves like a view of a function `f` that produces an abstract vector. Calling it invokes 
```
fv(args...) = view(fv.f(args...), fv.inds)
```
"""
@kwdef struct VecFuncView{TF<:Function, TI}
    f :: Function 
    inds :: TI 
end

(fv::VecFuncView)(args...) = view(fv.f(args...), fv.inds)
domainlimits(fv::VecFuncView, args...) = domainlimits(fv.f, args...)

#=======================================================================================================================
Views on predictors (enables removal of missing observations with minimal allocation)
=======================================================================================================================#
function Base.view(pred::LinearPredictor, inds)
    return LinearPredictor(
        view(pred.A, inds, :),
        view(pred.B, inds, :),
        view(pred.ε, inds)
    )
end

function Base.view(pred::NonlinearPredictor, inds; θ=pred.θ, multithreaded=pred.multithreaded)
    return NonlinearPredictor(
        VecFuncView(pred.f, inds),
        view(pred.ε, inds), 
        θ,
        multithreaded
    )
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

    function StateSpaceModel{TX,TP,TO}(state, predictor, observer, outlier) where {TX<:MvGaussian, TP<:StatePredictor, TO<:StatePredictor}
        return new{TX,TP,TO}(correlated(state), predictor, observer, outlier)
    end

    function StateSpaceModel(state::MvGaussian, predictor::TP, observer::TO, outlier) where {TP<:StatePredictor, TO<:StatePredictor}
        newstate = correlated(state)
        return new{typeof(newstate),TP,TO}(newstate, predictor, observer, outlier)
    end
end

#=
function StateSpaceModel(state::MvGaussian{<:Any, <:Diagonal}, predictor::StatePredictor, observer::StatePredictor, outlier)
    corrstate = correlated(state)
    return StateSpaceModel(corrstate, predictor, observer, outlier)
end
=#

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
    x = predict(ss.predictor, ss.state, u)

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


#=======================================================================================================================
Prediction functions (uncertainty propagation)
=======================================================================================================================#

#Linear predictors with additive noise
function predict(pred::LinearPredictor, X::MvGaussian, u)
    xh = pred.A*X.μ + pred.B*u 
    Σh = add_lcov(std(pred.ε), pred.A*std(X).L)
    return MvGaussian(xh, Σh)
end

#Nonlinar predictors with additive noise (produces an MvGaussian)
function predict(pred::NonlinearPredictor, x::MvGaussian, u)
    θ = scale_spread(pred.f, pred.θ, x)
    Xp = predict(pred.f, SigmaPoints(θ, x), u, multithreaded=pred.multithreaded)
    return Xp + pred.ε #Addition of noise converts sigma points to MvGaussian
end

#Inner predict tunction that is applied directly to sigma points without additive noise
function predict(fu, X::SigmaPoints, u; multithreaded=false)
    f(i) = fu(X[i], u)
    f_task(i) = Threads.@spawn(fu(X[i], u))
    inds = eachindex(X)

    if multithreaded
        return SigmaPoints(X.weights, fetch.(map(f_task, inds)))
    else
        return SigmaPoints(X.weights, map(f, inds))
    end
end


#=======================================================================================================================
Update functions (Kalman-Update)
=======================================================================================================================#
function update(obs::LinearPredictor, x::MvGaussian{Tμ,TΣ}, y::AbstractVector, u; outlier=Inf) where {Tμ, TΣ} 
    (C, D, R, P) = (obs.A, obs.B, std(obs.ε), std(x))
    yh = C*x.μ .+ D*u

    S = add_lcov(R, C*P.L) #Innovation covariance
    Z = MvGaussian(y.-yh, S) #Innovation distribution
    Pxy = (P.L*P.U)*C'#Obtain cross-covariance of state and measurement innovations
    K = (Pxy/S.U)/S.L #Kalman gain

    #Scale the gain based off outliers
    outlier_scaling!(K, Z, outlier)

    #Update the posterior
    μ = x.μ .+ K*Z.μ

    #This sometimes fails because rounding error on subtraction makes the matrix "negative", skip if that happens
    Σ = try
        sub_lcov(std(x), K*S.L)
    catch err
        @warn "Covariance update failed, skipping this step:\n" * sprint(showerror, err)
        x.σ
    end

    return (X=MvGaussian(Tμ(μ), TΣ(Σ)), Y=MvGaussian(yh, S), K=K)
end


function update(obs::NonlinearPredictor, x::MvGaussian{Tμ,TΣ}, y::AbstractVector, u; outlier=Inf) where {Tμ, TΣ}
    #Build the sigma points from the Gaussian variable
    Xp = SigmaPoints(obs.θ, x)

    #Propagate the sigma points through the predictor
    Yp = predict(obs.f, Xp, u, multithreaded=obs.multithreaded)
    Y  = Yp + obs.ε #Predicted Y distribution
    Z  = MvGaussian(y .- mean(Y), std(Y)) #Innovation distribution

    S   = std(Z) #Innovation covariance
    Pxy = cov(Xp, Yp) #Obtain cross-covariance of state and measurement innovations
    K   = (Pxy/S.U)/S.L #Kalman gain

    #Scale the gain based off outliers
    outlier_scaling!(K, Z, outlier)

    #Update the posterior
    μ = x.μ .+ K*Z.μ
    Σ = try
        sub_lcov(std(x), K*S.L)
    catch err
        @warn "Covariance update failed, skipping this step:\n" * sprint(showerror, err)
        x.σ
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



