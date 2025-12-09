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
    Σh = add_lcov(pred.Σ, pred.A*X.Σ.L)
    return MvGaussian(xh, Σh)
end

function predict_similar(pred::LinearPredictor, X::MvGaussian{T}, u) where T
    xh = T(pred.A*X.μ + pred.B*u)
    Σh = add_lcov(pred.Σ, pred.A*X.Σ.L)
    return MvGaussian(xh, Σh)
end

#Nonlinar predictors (returns the same type as X)
function predict(pred::NonlinearPredictor, X::MvGaussian, u)
    Xp = predict(pred, SigmaPoints(X, pred.θ), u)
    return MvGaussian(pred.Σ, Xp)
end

function predict(pred::NonlinearPredictor, X::SigmaPoints, u)
    f(x) = pred.f(x, u)
    f_task(x) = Threads.@spawn(pred.f(x,u))

    if pred.multithreaded
        return SigmaPoints(fetch.(map(f_task, X.points)), X.weights)
    else
        return SigmaPoints(map(f, X.points), X.weights)
    end
end

function predict_similar(pred::NonlinearPredictor, X::MvGaussian, u)
    Xp = predict!(pred, SigmaPoints(X, pred.θ), u)
    return MvGaussian(pred.Σ, Xp)
end

function predict!(pred::NonlinearPredictor, X::SigmaPoints, u)
    f(x) = pred.f(x, u)

    if pred.multithreaded
        Threads.@threads for ii in eachindex(X.points)
            X.points[ii] = f(X.points[ii])
        end
    else
        for ii in eachindex(X.points)
            X.points[ii] = f(X.points[ii])
        end
    end
    return X
end



#=======================================================================================================================
Update functions (Kalman-Update)
=======================================================================================================================#
function update(obs::LinearPredictor, X::MvGaussian{Tμ,TΣ}, y::AbstractVector, u; outlier=Inf) where {Tμ, TΣ} 
    (C, D, R, P) = (obs.A, obs.B, obs.Σ, X.Σ)
    yh = C*X.μ .+ D*u

    S = add_lcov(R, C*P.L) #Innovation covariance
    Z = MvGaussian(y.-yh, S) #Innovation distribution
    Pxy = (P.L*P.U)*C'#Obtain cross-covariance of state and measurement innovations
    K = (Pxy/S.U)/S.L #Kalman gain

    #Scale the gain based off outliers
    outlier_scaling!(K, Z, outlier)

    #Update the posterior
    μ = X.μ .+ K*Z.μ
    Σ = sub_lcov(X.Σ, K*S.L)

    return (X=MvGaussian(Tμ(μ), TΣ(Σ)), Y=MvGaussian(yh, S), K=K)
end


function update(obs::NonlinearPredictor, X::MvGaussian{Tμ,TΣ}, y::AbstractVector, u; outlier=Inf) where {Tμ, TΣ}
    #Build the sigma points from the Gaussian variable
    Xp = SigmaPoints(X, obs.θ)

    #Propagate the sigma points through the predictor
    Yp = predict(obs, Xp, u)
    Y  = MvGaussian(obs.Σ, Yp) #Predicted Y distribution
    Z  = MvGaussian(y.-Y.μ, Y.Σ) #Innovation distribution

    S   = Z.Σ #Innovation covariance
    Pxy = cov(Xp, Yp) #Obtain cross-covariance of state and measurement innovations
    K   = (Pxy/S.U)/S.L #Kalman gain

    #Scale the gain based off outliers
    outlier_scaling!(K, Z, outlier)

    #Update the posterior
    μ = X.μ .+ K*Z.μ
    Σ = sub_lcov(X.Σ, K*S.L)

    return (X=MvGaussian(Tμ(μ), TΣ(Σ)), Y=Y, K=K)
end

#Scale the gain based on outlier score of the prediction error distribution ΔY
function outlier_scaling!(K::AbstractArray, ΔY::MvGaussian, cutoff::Real)
    for (ii, Δy) in enumerate(ΔY.μ)
        z = Δy/chol_std(ΔY.Σ, ii)
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
