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
    return SigmaPoints(map(f, X.points), X.weights)
end

function predict_similar(pred::NonlinearPredictor, X::MvGaussian, u)
    Xp = predict!(pred, SigmaPoints(X, pred.θ), u)
    return MvGaussian(pred.Σ, Xp)
end

function predict!(pred::NonlinearPredictor, X::SigmaPoints, u)
    f(x) = pred.f(x, u)
    for ii in eachindex(X.points)
        X.points[ii] = f(X.points[ii])
    end
    return X
end



#=======================================================================================================================
Update functions (Kalman-Update)
=======================================================================================================================#
function update(obs::LinearPredictor, X::MvGaussian{Tμ,TΣ}, y::AbstractVector, u; outlier=Inf) where {Tμ, TΣ} 
    (C, D, R, P) = (obs.A, obs.B, obs.Σ, X.Σ)
    yh = C*X.μ .+ D*u

    S = add_lcov(R, C*P.L)
    Pxy = (P.L*P.U)*C'#Obtain cross-covariance of state and measurement innovations
    K = (Pxy/S.U)/S.L #Kalman gain

    σz = chol_std(S)
    μ = X.μ .+ K*scale_innovation.(y.-yh, σz, outlier=outlier)
    Σ = add_lcov((I-K*C)*P.L, K*R.L)

    return (X=MvGaussian(Tμ(μ), TΣ(Σ)), Y=MvGaussian(yh, S), K=K)
end


function update(obs::NonlinearPredictor, X::MvGaussian{Tμ,TΣ}, y::AbstractVector, u; outlier=Inf) where {Tμ, TΣ}
    #Build the sigma points from the Gaussian variable
    Xp = SigmaPoints(X, obs.θ)

    #Propagate the sigma points through the predictor
    Yp = predict(obs, Xp, u)
    Y  = MvGaussian(obs.Σ, Yp) #Predicted Y distribution

    S   = Y.Σ #Innovation covariance
    Pxy = cov(Xp, Yp) #Obtain cross-covariance of state and measurement innovations
    K   = (Pxy/S.U)/S.L #Kalman gain

    σz = chol_std(Y.Σ)
    μ = X.μ .+ K*scale_innovation.(y.-Y.μ, σz, outlier=outlier)
    Σ = sub_lcov(X.Σ, K*S.L)

    return (X=MvGaussian(Tμ(μ), TΣ(Σ)), Y=Y, K=K)
end

#Scale the innoviation to avoid chasing outliers
function scale_innovation(Δy::Real, σy::Real; outlier)
    if isfinite(outlier)
        σε = (outlier/3)*σy
        return asinh(Δy/σε)*σε
    else
        return Δy
    end
end