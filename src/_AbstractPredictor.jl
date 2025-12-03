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