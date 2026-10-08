using LinearAlgebra
using StaticArrays
using Accessors
import Statistics.mean
import Statistics.cov
import Statistics.std


abstract type AbstractGaussian end

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
Base.view(x::ZeroVec, inds...) = x
Base.copy(z::ZeroVec) = ZeroVec()


"""
ConstVec

A vector-like object of zeros that does not allocate memory, indexing it always returns a constant
"""
struct ConstVec{T}
    all :: T
end 
Base.getindex(x::ConstVec, i::Int) = x.all
Base.view(x::ConstVec, inds...) = x


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
Base.copy(x::UvGaussian) = UvGaussian(copy(mean(x)), copy(std(x)))
uv_zero(::Type{T}) where T = UvGaussian(zero(T), zero(T))
meantype(x::UvGaussian) = typeof(x.μ)

mean(x::UvGaussian) = x.μ
std(x::UvGaussian) = x.σ
cov(x::UvGaussian) = abs2(x.σ)
cholcol(x::UvGaussian, i::Integer) = x.σ[i]
meancol(x::UvGaussian) = x.μ

"""
MvGaussian(x, Σ)

An uncertaint vector that by default, is assumed to follow a Gaussian distribution. The uncertainty in this object
takes the lower square-root form. Diagonal and triangular matrices are already assumed to be in square root form. Otherwise 
the constructor performs a cholesky decomposition.
"""
@kwdef struct MvGaussian{Tμ<:Union{ZeroVec,AbstractVector}, Tσ} <: AbstractGaussian
    μ :: Tμ
    σ :: Tσ
    MvGaussian{Tμ,Tσ}(x, m) where {Tμ<:Union{ZeroVec,AbstractVector}, Tσ} = new{Tμ,Tσ}(x, lowerform(m))
    function MvGaussian(x::Union{ZeroVec,AbstractVector}, m)
        c = lowerform(m) #Enforce lower triangular form
        return new{typeof(x), typeof(c)}(x, c)
    end
end
MvGaussian(σ::Union{Cholesky,AbstractMatrix}) = MvGaussian(ZeroVec(), σ)
gaussian(μ::AbstractVector, σ::Union{Cholesky, AbstractMatrix}) = MvGaussian(μ, σ)

MvGaussian(args::UvGaussian...) = MvGaussian(SVector(map(mean, args)), Diagonal(SVector(map(std, args))))
MvGaussian(args::AbstractVector{<:UvGaussian}) = MvGaussian(map(mean, args), Diagonal(map(std, args)))

Base.convert(::Type{MvGaussian{TX,TM}}, x::MvGaussian) where {TX,TM} = MvGaussian(convert(TX, x.μ), convert(TM, x.σ))
Base.length(x::MvGaussian) = length(x.μ)
Base.copy(x::MvGaussian) = MvGaussian(copy(mean(x)), copy(std(x)))
meantype(x::MvGaussian) = typeof(x.μ)
correlated(x::MvGaussian) = MvGaussian(x.μ, densechol(x.σ))

mean(x::MvGaussian) = x.μ
std(x::MvGaussian) = x.σ 
cov(x::MvGaussian{<:Any, <:Cholesky}) = AbstractMatrix(std(x))
cov(x::MvGaussian{<:Any, <:Diagonal}) = x.σ*x.σ

mean(x::MvGaussian, i::Integer) = x.μ[i]
std(x::MvGaussian{<:Any, <:Cholesky}, i::Integer) = cholstd(x.σ, i)
std(x::MvGaussian{<:Any, <:Diagonal}, i::Integer) = x.σ[i,i]

cholcol(x::MvGaussian{<:Any, <:Cholesky}, i::Integer) = view(x.σ.L, :, i)
cholcol(x::MvGaussian{<:Any, <:Diagonal}, i::Integer) = view(x.σ, :, i)
cholrow(x::MvGaussian{<:Any, <:Cholesky}, i::Integer) = view(x.σ.U, :, i)
cholrow(x::MvGaussian{<:Any, <:Diagonal}, i::Integer) = view(x.σ, :, i)
meancol(x::MvGaussian) = x.μ

#Getting an index from a multivariate Gaussian produces a univariate Gaussian
Base.getindex(x::MvGaussian, i::Integer) = UvGaussian(mean(x, i), std(x, i))
Base.view(x::MvGaussian, inds) = MvGaussian(view(x.μ, inds), _stdview(x.σ, inds))
mv_zero(::Type{T}, n) where T = MvGaussian(ZeroVec(), zerochol(T, n))
mv_zero(::Type{MvGaussian}, n::Integer) = mv_zero(Float64, n)

_stdview(σ::Cholesky, inds) = Cholesky(UpperTriangular(view(σ.U, inds, inds)))
_stdview(σ::Diagonal, inds) = Diagonal(view(σ.diag, inds))

#Enforce the lower form of cholesky decomposition
function lowerform(ch::Cholesky{T,M}) where {T,M}
    if ch.uplo == 'L'
        return ch 
    elseif ch.uplo == 'U'
        factors = convert(M, ch.factors')
        return Cholesky(LowerTriangular(factors))
    end
    error("Cholesky `uplo` should either be 'U' or 'L'")
end

lowerform(m::Hermitian) = lowerform(cholesky(m))
lowerform(m::UpperTriangular{T,M}) where {T,M} = Cholesky(LowerTriangular{T,M}(Matrix(m.data')))
lowerform(m::LowerTriangular{T,M}) where {T,M} = Cholesky(m)
lowerform(m::Diagonal) = m

zerochol(::Type{T}, N::Integer) where T = Cholesky(LowerTriangular(zeros(T, N, N)))
zerochol(v::AbstractVector) = zerochol(eltype(v), length(v))
densechol(d::Diagonal) = Cholesky(LowerTriangular(Matrix(d)))
densechol(ch::Cholesky{<:Any, <:Diagonal}) = Cholesky(LowerTriangular(Matrix(ch.L)))
densechol(ch::Cholesky) = ch