abstract type AbstractPredictor end 

"""
LinearPredictor(A::AbstractMatrix, B::AbstractMatrix, Σ::Cholesky)

Linear predictor with added noise covariance Σ (internally a Cholesky factorization is applied)
Prediction is given by 
y = Ax + Bu + ε 
where x and u are vectors and ε is noise
"""
@kwdef struct LinearPredictor{TA<:AbstractArray, TB<:AbstractArray, TΣ<:Cholesky} <: AbstractPredictor
    A :: TA 
    B :: TB 
    Σ :: TΣ
end
LinearPredictor(A::AbstractMatrix, B::AbstractMatrix, Σ::AbstractMatrix) = LinearPredictor(A, B, cholesky(Σ))


"""
NonlinearPredictor(F::Function, Σ::Cholesky)

Nonlinear predictor with added noise covariance Σ (internally a Cholesky factorization is applied)
Prediction is given by y = Ax + Bu where x and u are vectors

"""
@kwdef struct NonlinearPredictor{TF<:Function, TΣ<:Cholesky} <: AbstractPredictor
    f :: TF
    Σ :: TΣ
end
NonlinearPredictor(f::Function, Σ::AbstractMatrix) = LinearPredictor(f, cholesky(Σ))


const StatePredictor  = Union{LinearPredictor, NonlinearPredictor}

#=======================================================================================================================
Prediction functions
=======================================================================================================================#
function predict(pred::LinearPredictor, X::GaussianVar, u)
    xh = pred.A*X.μ + pred.B*u 
    Σh = add_cov_sqrt(pred.Σ, pred.A*X.Σ.L)
    return GaussianVar(xh, Σh)
end

function predict_similar(pred::LinearPredictor, X::GaussianVar{T}, u) where T
    xh = T(pred.A*X.μ + pred.B*u)
    Σh = add_cov_sqrt(pred.Σ, pred.A*X.Σ.L)
    return GaussianVar(xh, Σh)
end

function predict(pred::NonlinearPredictor, X::GaussianVar, u) 
    return predict(pred, SigmaPoints(X), u)
end

function predict_similar(pred::NonlinearPredictor, X::GaussianVar, u) 
    return predict!(pred, SigmaPoints(X), u)
end

function predict(pred::NonlinearPredictor, X::SigmaPoints, u)
    f(x) = pred.f(x, u)
    points = map(f, X.points)
    return add_cov(SigmaPoints(points=points, weights=X.weights), pred.Σ)
end

function predict!(pred::NonlinearPredictor, X::SigmaPoints, u)
    f(x) = pred.f(x, u)
    for ii in eachindex(X.points)
        X.points[ii] = f(X.points[ii])
    end
    return add_cov(X, pred.Σ)
end

#=======================================================================================================================
Update functions
=======================================================================================================================#
