#======================================================================================================================================
ToDo:

(1) Don't index X.points[ii], use direct X[ii]
(2) Define 
    -   add_cov!(ch::Cholesky, X::SigmaPoints, μ)
    -   add_cov!(ch::Cholesky, X::SigmaPoints) = add_cov!(ch::Cholesky, X::SigmaPoints, mean(X))
    -   cholcov(X::SigmaPointsm, μ) = (N = length(μ); add_cov!(Cholesky(UpperTriangular(zeros(N,N))), X, μ)
    -   MvGaussian(X::SigmaPoints{<:AbstractVector}) = (μ = mean(X); MvGaussian(μ, cholcov(X, μ)))
(3) SigmaPoints "Source" will start off as a Multivariate Gaussian, but will end up as a vector 
    -   cholcol/meancol will need to be defined for those objects as well
    -   A good round-trip test is to see if 
        MvGaussian(predict(identity, SigmaPoints(d, θ))) ≈ d 
(4) We will need to consider UvGaussian
    -   We will also need to consider multiple arguments to a function such as f(x::UvGaussian, y::UvGaussian) -> z 
    -   This means the "source" is a Tuple{UvGaussian,UvGaussian} which are assumed to be independent
(5) May want to define "+" for SigmaPoints and MvGaussian/UvGaussian
    -   May want to add type Zero so that MvGaussian{Zero, Cholesky} doesn't need to worry about means

Post cleanup:
(1) Redefine "Σ" as "σ" for MvGaussian covariance field, as the square root form is being used 
(2) Redefine weight field "c" as "rc" and store the square-root result
======================================================================================================================================#

Base.length(MvGaussian) = length(MvGaussian.μ)

cholcol(x::MvGaussian{<:Any, <:Cholesky}, i::Int) = view(x.Σ.U, :, i)
cholcol(x::MvGaussian{<:Any, <:Diagonal}, i::Int) = view(x.Σ, :, i)
meancol(x::MvGaussian) = x.μ

"""
SigmaPoints{T}(source::T, weights::SigmaWeights)

Unscented transform using L+1 vectors as points
"""
Base.@kwdef struct SigmaPoints{T} <: AbstractVector
    source   :: T
    weights  :: SigmaWeights
end

SigmaPoints(X::MvGaussian, θ::SigmaParams) = SigmaPoints(X, SigmaWeights(length(X), θ))

Base.IndexStyle(::Type{<:SigmaPoints}) = IndexLinear()
Base.length(x::SigmaPoints) = 2*length(x.source) + 1
Base.firstindex(x::SigmaPoints) = 1
Base.lastindex(x::SigmaPoints) = length(x)

function Base.getindex(points::SigmaPoints{<:MvGaussian}, i::Int)
    μ = meancol(points.source)
    i == firstindex(points) && return μ

    N  = length(points.source)
    rc = sqrt(points.weights.c)

    if 2 <= i <= (N+1)
        return map((x, Δ)-> x + rc*Δ, μ, cholcol(points.source, i))
    elseif (N+2) <= i <= (2N+1)
        return map((x, Δ)-> x - rc*Δ, μ, cholcol(points.source, i))
    end 
    throw(BoundsError(points, i))
end