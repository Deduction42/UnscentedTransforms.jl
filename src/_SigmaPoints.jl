#======================================================================================================================================
ToDo:


Post cleanup:
(1) Redefine "Σ" as "σ" for MvGaussian covariance field, as the square root form is being used 
(2) Redefine add_cov to sqrtadd2, sqrtadd2left, sqrtadd2right


-   This is a good resource to verify other scaling rules
    https://www.mathworks.com/help/ident/ug/extended-and-unscented-kalman-filter-algorithms-for-online-state-estimation.html
======================================================================================================================================#

using LinearAlgebra
using StaticArrays
using Accessors
import Statistics.mean
import Statistics.cov
import Statistics.std
import Statistics.var


"""
ZeroVec 

A vector-like object of zeros that does not allocate memory, indexing it always returns `false`
"""
struct ZeroVec end 
Base.getindex(x::ZeroVec, i::Int) = false
Base.:+(z::ZeroVec, x::AbstractVector) = x 
Base.:+(x::AbstractVector, z::ZeroVec) = x
Base.:-(z::ZeroVec, x::AbstractVector) = -x
Base.:-(x::AbstractVector, z::ZeroVec) = x 


abstract type AbstractGaussian end

"""
gaussian(μ, σ)

Build a Gaussian object from mean μ, uncertainty σ. If both μ and σ are numbers, a univariate Gaussian is constructed.
If μ is a vector and σ is a triangular or diagonal matrix, it is assumed to be square root form, otherwise a cholesky decomposition is performed
"""
function gaussian end 

"""
±(μ, σ)

Produces a Gaussian uncertainty object. If μ is a vector and σ is a matrix/factorization, a multivariate distribution is built
"""
±(μ, σ) = gaussian(μ, σ)

"""
UvGaussian(x, σ)

An uncertaint value with uncertainty that, by default, is assumed to follow a Gaussian distribution
"""
@kwdef struct UvGaussian{T} <: AbstractGaussian
    μ :: T
    σ :: T
end
UvGaussian(μ::T1, σ::T2) where {T1,T2} = UvGaussian{promote_type(T1,T2)}(μ, σ)
gaussian(μ::Number, σ::Number) = UvGaussian(μ, σ)

Base.length(x::UvGaussian) = 1
meantype(x::UvGaussian) = typeof(x.μ)

mean(x::UvGaussian) = x.μ
std(x::UvGaussian) = x.σ
var(x::UvGaussian) = abs2(x.σ)
cholcol(x::UvGaussian, i::Integer) = x.σ[i]
meancol(x::UvGaussian) = x.μ

"""
MvGaussian(x, Σ)

An uncertaint vector that by default, is assumed to follow a Gaussian distribution. The uncertainty in this object
takes the square-root form. Diagonal and triangular matrices are already assumed to be in square root form. Otherwise 
the constructor performs a cholesky decomposition.
"""
@kwdef struct MvGaussian{TX<:Union{ZeroVec,AbstractVector}, TM<:Union{Diagonal,Factorization}} <: AbstractGaussian
    μ :: TX
    Σ :: TM
    MvGaussian{TX,TM}(x, m) where {TX<:Union{ZeroVec,AbstractVector}, TM<:Union{Diagonal,Factorization}} = new{TX,TM}(x, m)
    function MvGaussian(x::Union{ZeroVec,AbstractVector}, m::Factorization)
        cm = cholesky(m)
        return new{typeof(x), typeof(cm)}(x, m)
    end
    function MvGaussian(x::Union{ZeroVec,AbstractVector}, m::Diagonal)
        return new{typeof(x), typeof(m)}(x, m)
    end
end
MvGaussian(x::Union{ZeroVec,AbstractVector}, m::AbstractMatrix) = MvGaussian(x, cholesky(m))
MvGaussian(x::Union{ZeroVec,AbstractVector}, m::Union{LowerTriangular,UpperTriangular}) = MvGaussian(x, Cholesky(m))
MvGaussian(m::Union{Diagonal,Factorization}) = MvGaussian(ZeroVec(), m)
gaussian(μ::AbstractVector, σ::Union{Factorization, AbstractMatrix}) = MvGaussian(μ, σ)

MvGaussian(args::UvGaussian...) = MvGaussian(SVector(map(mean, args)), Diagonal(SVector(map(std, args))))
MvGaussian(args::AbstractVector{<:UvGaussian}) = MvGaussian(map(mean, args), Diagonal(map(std, args)))

Base.convert(::Type{MvGaussian{TX,TM}}, x::MvGaussian) where {TX,TM} = MvGaussian(TX(x.μ), TM(x.Σ))
Base.length(MvGaussian) = length(MvGaussian.μ)
meantype(x::MvGaussian) = typeof(x.μ)

mean(x::MvGaussian) = x.μ
std(x::MvGaussian) = x.Σ 
cov(x::MvGaussian{<:Any, <:Cholesky}) = x.L*x.U 
cov(x::MvGaussian{<:Any, <:Diagonal}) = x.Σ*x.Σ

mean(x::MvGaussian, i::Integer) = x.μ[i]
std(x::MvGaussian{<:Any, <:Cholesky}, i::Integer) = chol_std(x.Σ, i)
std(x::MvGaussian{<:Any, <:Diagonal}, i::Integer) = x.Σ[i,i]

cholcol(x::MvGaussian{<:Any, <:Cholesky}, i::Integer) = view(x.Σ.L, :, i)
cholcol(x::MvGaussian{<:Any, <:Diagonal}, i::Integer) = view(x.Σ, :, i)
cholrow(x::MvGaussian{<:Any, <:Cholesky}, i::Integer) = view(x.Σ.R, :, i)
cholrow(x::MvGaussian{<:Any, <:Diagonal}, i::Integer) = view(x.Σ, :, i)
meancol(x::MvGaussian) = x.μ

#Getting an index from a multivariate Gaussian produces a univariate Gaussian
Base.getindex(x::MvGaussian, i::Integer) = UvGaussian(mean(x, i), std(x, i))


"""
SigmaParams(α = 1.0, κ = 0.0, β = 2.0)

Sigma point parameters for the Unscented Transform. The default values are geared toward Gaussian distribution 
assumptions (frequent in this package) and covering enough of the distribution to capture nonlinearities
    α=1
    κ=0 
    β=2
Some sources suggest a value of α~0.001, but this essentially reverts to a linear approximation. The main motivation for this 
small value recommendation is to prevent sampling sigma points that go out of common-sense boundaries or asymptotes. This package 
intead defaults to α=1 (so that points envelope ~50% of the distribution) and encourages users to either
1. Clamp values inside the function as a first step 
2. Define asymptotes of functions if they exist, and this package will shrink α if an asymptote is close
"""
Base.@kwdef struct SigmaParams
    α :: Float64 = 1.0
    κ :: Float64 = 0.0
    β :: Float64 = 2.0
end

"""
SigmaWeights(c :: Float64, μ :: Tuple{Float64, Float64}, Σ :: Tuple{Float64, Float64})

Weights for sigma points, calculated from SigmaParams and the state dimension L
"""
Base.@kwdef struct SigmaWeights
    L  :: Float64
    α² :: Float64 
    κ  :: Float64 
    β  :: Float64
    rc :: Float64
    Wn :: Float64
    Wμ :: Float64
    Wσ :: Float64
end

function SigmaWeights(L::Integer, P::SigmaParams=SigmaParams())
    α² = abs2(P.α)
    κ  = P.κ
    β  = P.β

    c  = α²*(L+κ) #Sigma point step size 
    Wn = 0.5/c #Off-center weights (both mean and cov)
    Wμ = 1 - L/c #Mean center weight 
    Wσ = Wμ + 1 - α² + β #Cov center weight
    return SigmaWeights(L=L, α²=α², κ=κ, β=β, rc=sqrt(c), Wn=Wn, Wμ=Wμ, Wσ=Wσ)
end
SigmaWeights(θ::SigmaWeights) = θ

"""
SigmaPoints{T}(source::T, weights::SigmaWeights)

Unscented transform using 2N+1 vectors as points
"""
Base.@kwdef struct SigmaPoints{S,T} <: AbstractVector{T}
    source   :: S
    weights  :: SigmaWeights

    SigmaPoints(source::AbstractGaussian, θ::SigmaWeights) = new{typeof(source), meantype(source)}(source, θ)

    function SigmaPoints(source::AbstractVector, θ::SigmaWeights) 
        Base.require_one_based_indexing(source)
        return new{typeof(source), eltype(source)}(source, θ)
    end
end

SigmaPoints(X, θ::SigmaParams) = SigmaPoints(X, SigmaWeights(dimlength(X), θ))

Base.IndexStyle(::Type{<:SigmaPoints}) = IndexLinear()
Base.length(x::SigmaPoints{<:AbstractGaussian}) = 2*length(x.source) + 1
Base.length(x::SigmaPoints{<:AbstractVector}) = length(x.source)
Base.size(x::SigmaPoints) = (length(x),)
Base.firstindex(x::SigmaPoints) = 1
Base.lastindex(x::SigmaPoints) = length(x)

halfindex(x::SigmaPoints) = firstindex(x) + dimlength(x)
dimlength(x::AbstractGaussian) = length(mean(x))
dimlength(x::UvGaussian) = 1
dimlength(X::SigmaPoints{<:AbstractGaussian}) = length(X.source)
dimlength(X::SigmaPoints{<:UvGaussian}) = 1
dimlength(X::SigmaPoints{<:AbstractVector}) = length(X.source[begin])

#If source is a vector, simply index it
Base.getindex(points::SigmaPoints{<:AbstractVector}, i::Int) = points.source[i]

#If source is a Gaussian, generate the sigma point
function Base.getindex(points::SigmaPoints{<:AbstractGaussian}, i::Int)
    μ = meancol(points.source)
    i == firstindex(points) && return μ

    N  = length(points.source)
    rc = points.weights.rc

    if firstindex(points) < i <= halfindex(points)
        ic = i-1
        return map((x, Δ)-> x + rc*Δ, μ, cholcol(points.source, ic))

    elseif halfindex(points) < i <= lastindex(points)
        ic =  i-(N+1)
        return map((x, Δ)-> x - rc*Δ, μ, cholcol(points.source, ic))
    end 
    throw(BoundsError(points, i))
end

#If the source is a tuple of UvGaussians, generate the sigma point
function Base.getindex(points::SigmaPoints{<:Tuple{Vararg{<:UvGaussian}}}, i::Int)
    μ = map(mean, points.source)
    i == firstindex(points) && return μ

    N  = length(points.source)
    rc = points.weights.rc

    if firstindex(points) < i <= halfindex(points)
        iμ = i-1
        return @set μ[iμ] = μ[iμ] + rc*std(points.source[iμ])
    elseif halfindex(points) < i <= lastindex(points)
        iμ = i-(N+1)
        return @set μ[iμ] = μ[iμ] - rc*std(points.source[iμ])
    end 
    throw(BoundsError(points, i))
end

function MvGaussian(X::SigmaPoints) 
    μ = mean(X)
    return MvGaussian(μ, std(X, μ))
end

function UvGaussian(x::SigmaPoints)
    μ = mean(X)
    return UvGaussian(x::SigmaPoints, xtd(X, μ))
end

#Dispatch patterns for generic function
gaussian(x::SigmaPoints{<:AbstractVector{<:AbstractVector}}) = MvGaussian(x)
gaussian(x::SigmaPoints{<:AbstractVector{<:Number}}) = UvGaussian(x)
gaussian(x::SigmaPoints{<:AbstractGaussian}) = x.source


#======================================================================================================================================
Stats functions
======================================================================================================================================#
function mean(X::SigmaPoints{<:AbstractVector})
    (w0, wn) = (X.weights.Wμ, X.weights.Wn)
    μ = w0.*X[begin]
    outer_inds = (firstindex(X)+1):lastindex(X)

    if ismutable(μ)
        for ind in outer_inds
            μ .+= wn.*X[ind]
        end
        return μ
    else
        return sum(ind-> wn.*X[ind], outer_inds, init=μ)
    end
end

function mean(X::SigmaPoints{<:AbstractVector{<:Number}}) 
    (w0, wn) = (X.weights.Wμ, X.weights.Wn)
    μ = w0*X[begin]
    outer_inds = (firstindex(X)+1):lastindex(X)

    return sum(ind-> wn*X[ind], outer_inds, init=μ)
end


mean(X::SigmaPoints{<:AbstractGaussian}) = mean(X.source)

"""
Returns a weighted covariance matrix of two sets of sigma points, based on weights from the first set
"""
function cov(X::SigmaPoints, Y::SigmaPoints)
    weight(ii::Integer) = ifelse(ii==1, X.weights.Wσ, X.weights.Wn)

    (nx, ny) = (length(X), length(Y))
    if nx != ny
        error("Two sets of sigma points must have the same number of points ($(nx) ≠ $(ny))")
    end

    (μx, μy) = (mean(X), mean(Y))
    T = promote_type(Float64, eltype(μx), eltype(μy))
    S = zeros(T, length(μx), length(μy))
    ii = 0
    for (x, y) in zip(X, Y)
        ii += 1
        S .+= weight(ii) .* (x.-μx) .* (y.-μy)'
    end
    return S
end

function std(X::SigmaPoints{<:AbstractVector{<:AbstractVector}}, μ::AbstractVector) 
    x0 = X[begin]
    nd = dimlength(X) 
    ch = Cholesky(UpperTriangular(zeros(eltype(x0), nd, nd)))
    add_cov!(ch, X, μ)
    return ch 
end

function std(X::SigmaPoints{<:AbstractVector{<:Number}}, μ::Number) 
    (w0, wn) = (X.weights.Wσ, X.weights.Wn)

    σ² = sum(x-> wn*(x-μ)^2, X[(begin+1):end]) #Off-center weights
    return sqrt(σ² + w0*(X[begin]-μ)^2) #Center weights
end


std(X::SigmaPoints{<:AbstractVector}) = std(X, mean(X))
std(X::SigmaPoints{<:AbstractGaussian}) = std(X.source)

function cov(x::SigmaPoints)
    ch = std(x)
    return ch.L*ch.U
end

#======================================================================================================================================
Helper functions for adding covariances in square root form
======================================================================================================================================#
"""
add_cov(σ1, σ2)

Returns the square-root form of adding covariances
"""
function add_cov end

add_cov(ch::Cholesky, X::SigmaPoints) = add_cov!(ch, X)
add_cov(ch::Cholesky, X::SigmaPoints, μ::AbstractVector) = add_cov!(copy(ch), X, μ)
add_cov!(ch::Cholesky, X::SigmaPoints) = add_cov!(ch, X, mean(X))

function add_cov!(ch::Cholesky, X::SigmaPoints, μ::AbstractVector)
    (w0, wn) = (X.weights.Wσ, X.weights.Wn)
    x = zeros(eltype(X[begin]), length(X[begin]))

    #Add all of the surrounding points
    for ii in (firstindex(X)+1):lastindex(X)
        x .= X[ii] .- μ
        chol_update!(ch, x, wn)
    end

    #Add central point (where weight could be negative) 
    #Because of negative weight, doing this last reduces risk of negative covariacne
    x .= X[begin] .- μ
    chol_update!(ch, x, w0)

    return ch
end

add_cov(ch1::Cholesky, ch2::Cholesky) = add_cov!(copy(ch1), ch2)
add_cov(x1::Number, x2::Number) = sqrt(abs2(x1) + abs2(x2))


function add_cov!(ch1::Cholesky, ch2::Cholesky)
    x = zeros(eltype(ch2.U), size(ch2.U, 1))

    for xi in eachcol(ch2.L)
        x .= xi
        lowrankupdate!(ch1, x)
    end
    return ch1
end


"""
add_lcov(ch::Cholesky, L::AbstractMatrix)

Eqivalent of `cholesky(ch.U'ch.U + L*L')`
"""
add_lcov(ch::Cholesky, L::AbstractMatrix) = add_lcov!(copy(ch), L)

function add_lcov!(ch::Cholesky, L::AbstractMatrix)
    x = zeros(eltype(L), size(L, 1))

    for xi in eachcol(L)
        x .= xi
        lowrankupdate!(ch, x)
    end
    return ch
end

"""
add_rcov(A::AbstractMatrix, B::AbstractMatrix)

Equivalent of `cholesky(A'A + B'B)`
"""
function add_rcov(A::AbstractMatrix, B::AbstractMatrix)
    R = qr!([A;B]).R

    #Force positive diagonal by flipping row signs
    for ii in axes(R,1)
        if R[ii,ii] < 0
            R[ii,:] .= flipsign.(R[ii,:], -1)
        end
    end
    return Cholesky(UpperTriangular(R))
end


"""
add_lcov(A::AbstractMatrix, B::AbstractMatrix)

Equivalent of cholesky(A*A' + B*B')
"""
function add_lcov(A::AbstractMatrix, B::AbstractMatrix)
    L = lq!([A B]).L

    #Force positive diagonal by flipping row signs
    for ii in axes(L,1)
        if L[ii,ii] < 0
            L[:,ii] .= flipsign.(L[:,ii], -1)
        end
    end
    return Cholesky(LowerTriangular(L))
end

"""
sub_lcov(ch::Cholesky, L::AbstractMatrix)

Equivalent of cholesky(ch.U'ch.U - L*L')
"""
sub_lcov(ch::Cholesky, L::AbstractMatrix) = sub_lcov!(deepcopy(ch), L)

function sub_lcov!(ch::Cholesky, L::AbstractMatrix)
    x = zeros(eltype(L), size(L, 1))

    for xi in eachcol(L)
        x .= xi
        lowrankdowndate!(ch, x)
    end
    return ch
end

#=
function old_cov(X::SigmaPoints)
    weight(ii::Integer) = ifelse(ii==1, X.weights.Σ[1], X.weights.Σ[2])

    nx = length(first(X))
    μx = mean(X)
    T  = promote_type(Float64, eltype(μx))
    S  = zeros(T, nx, nx)
    ii = 0
    for x in X
        S .+= weight(ii) .* (x.-μx) .* (x.-μx)'
    end
    hermitianpart!(S) 

    return S
end

function sigma_points(X::MvGaussian, w::SigmaWeights)
    σc = sqrt(w.c)
    points = [X.μ]
    
    for l in eachcol(X.Σ.L)
        Δ = σc.*l
        push!(points, X.μ .+ Δ)
        push!(points, X.μ .- Δ)
    end

    return SigmaPoints(source=points, weights=w)
end
=#

"""
chol_update!(ch::Cholesky, x::AbstractVector, w::Real)

Updates cholesky decomposition it gives the equivalent of 
cholesky(ch.U'*ch.U + w*(x'*x))
This function is non-allocating and the vector "x" is destroyed in the process
"""
function chol_update!(ch::Cholesky, x::Vector, w::Real)
    x .= sqrt(abs(w)) .* x
    return w >= 0 ? lowrankupdate!(ch, x) : lowrankdowndate!(ch, x)
end


chol_var(ch::Cholesky) = map(ii->chol_var(ch, ii), axes(ch.U, 2))
chol_std(ch::Cholesky) = map(ii->chol_std(ch, ii), axes(ch.U, 2))

chol_std(ch::Cholesky, ii::Integer) = sqrt(chol_var(ch, ii))
function chol_var(ch::Cholesky, ii::Integer)
    v = view(ch.U, :, ii)
    return dot(v,v)
end


Base.isfinite(x::MvGaussian) = all(isfinite, x.μ) & all(isfinite, x.Σ.U)
