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
meantype(x::UvGaussian) = typeof(x.μ)

mean(x::UvGaussian) = x.μ
std(x::UvGaussian) = x.σ
cov(x::UvGaussian) = abs2(x.σ)
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
    σ :: TM
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
MvGaussian(m::Hermitian) = MvGaussian(ZeroVec(), cholesky(m))
gaussian(μ::AbstractVector, σ::Union{Factorization, AbstractMatrix}) = MvGaussian(μ, σ)

MvGaussian(args::UvGaussian...) = MvGaussian(SVector(map(mean, args)), Diagonal(SVector(map(std, args))))
MvGaussian(args::AbstractVector{<:UvGaussian}) = MvGaussian(map(mean, args), Diagonal(map(std, args)))

Base.convert(::Type{MvGaussian{TX,TM}}, x::MvGaussian) where {TX,TM} = MvGaussian(convert(TX, x.μ), convert(TM, x.σ))
Base.length(x::MvGaussian) = length(x.μ)
meantype(x::MvGaussian) = typeof(x.μ)
correlated(x::MvGaussian{<:Any,<:Diagonal}) = MvGaussian(x.μ, diag2chol(x.σ))
correlated(x::MvGaussian{<:Any,<:Cholesky}) = x

mean(x::MvGaussian) = x.μ
std(x::MvGaussian) = x.σ 
cov(x::MvGaussian{<:Any, <:Cholesky}) = AbstractMatrix(std(x))
cov(x::MvGaussian{<:Any, <:Diagonal}) = x.σ*x.σ

mean(x::MvGaussian, i::Integer) = x.μ[i]
std(x::MvGaussian{<:Any, <:Cholesky}, i::Integer) = chol_std(x.σ, i)
std(x::MvGaussian{<:Any, <:Diagonal}, i::Integer) = x.σ[i,i]

cholcol(x::MvGaussian{<:Any, <:Cholesky}, i::Integer) = view(x.σ.L, :, i)
cholcol(x::MvGaussian{<:Any, <:Diagonal}, i::Integer) = view(x.σ, :, i)
cholrow(x::MvGaussian{<:Any, <:Cholesky}, i::Integer) = view(x.σ.U, :, i)
cholrow(x::MvGaussian{<:Any, <:Diagonal}, i::Integer) = view(x.σ, :, i)
meancol(x::MvGaussian) = x.μ

#Getting an index from a multivariate Gaussian produces a univariate Gaussian
Base.getindex(x::MvGaussian, i::Integer) = UvGaussian(mean(x, i), std(x, i))
Base.view(x::MvGaussian, inds) = MvGaussian(view(x.μ, inds), _stdview(x.σ, inds))

_stdview(σ::Cholesky, inds) = Cholesky(UpperTriangular(view(σ.U, inds, inds)))
_stdview(σ::Diagonal, inds) = Diagonal(view(σ.diag, inds))
