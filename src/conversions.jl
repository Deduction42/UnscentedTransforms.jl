#=========================================================================================================================
Gaussian conversions
=========================================================================================================================#
function UvGaussian{T}(x::UvGaussian) where {T}
    return UvGaussian{T}(mean(x), std(x))
end

function MvGaussian{V,M}(x::MvGaussian{<:AbstractVector,<:Diagonal}) where {V,M<:Cholesky}
    return MvGaussian{V,M}(mean(x), Cholesky(LowerTriangular(std(x))))
end

function MvGaussian{V,M}(x::MvGaussian{<:AbstractVector,<:Cholesky}) where {V,M<:Diagonal}
    return MvGaussian{V,M}(mean(x), Diagonal(std(x).L))
end

function MvGaussian{V,M}(x::MvGaussian{<:AbstractVector,<:Any}) where {V,M}
    return MvGaussian{V,M}(mean(x), std(x))
end

function MvGaussian{V,M}(args::AbstractVector{<:UvGaussian}) where {V,M<:Diagonal} 
    return MvGaussian{V,M}(map(mean, args), Diagonal(map(std, args)))
end

function MvGaussian{V,M}(args::AbstractVector{<:UvGaussian}) where {V,M<:Cholesky} 
    return MvGaussian{V,M}(map(mean, args), LowerTriangular(Diagonal(map(std, args))))
end

Base.convert(::Type{UvGaussian{T}}, x::UvGaussian) where T = UvGaussian{T}(x)
Base.convert(::Type{MvGaussian{C,M}}, x::MvGaussian) where {C,M} = MvGaussian{C,M}(x)
Base.convert(::Type{MvGaussian{C,M}}, x::Cholesky) where {C,M} = MvGaussian{C,M}(x)
Base.convert(::Type{MvGaussian{C,M}}, x::Diagonal) where {C,M} = MvGaussian{C,M}(x)
Base.convert(::Type{MvGaussian{C,M}}, x::AbstractVector{<:UvGaussian}) where {C,M} = MvGaussian{C,M}(x)

#=========================================================================================================================
Predictor conversions
=========================================================================================================================#
function NonlinearPredictor{Tf,Tε}(p::NonlinearPredictor) where {Tf,Tε}
    return NonlinearPredictor{Tf,Tε}(p.f, p.ε, p.θ, p.multithreaded)
end

function LinearPredictor{TA,TB,Tε}(p::LinearPredictor) where {TA, TB, Tε}
    return LinearPredictor{TA,TB,Tε}(p.A, p.B, p.ε)
end

Base.convert(::Type{NonlinearPredictor{Tf,Tε}}, p::NonlinearPredictor) where {Tf,Tε} = NonlinearPredictor{Tf,Tε}(p)
Base.convert(::Type{LinearPredictor{TA,TB,Tε}}, p::NonlinearPredictor) where {TA,TB,Tε} = LinearPredictor{TA,TB,Tε}(p)

#=========================================================================================================================
Miscillaneous conversions
=========================================================================================================================#
Base.convert(::Type{ZeroVec}, v::AbstractVector{<:Number}) = all(iszero(v)) ? ZeroVec() : throw(ArgumentError("Cannot convert to ZeroVec, all elements must be zero"))