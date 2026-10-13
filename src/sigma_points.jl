#======================================================================================================================================
ToDo:

Post cleanup:
(1) Redefine add_cov to sumstd, sumstd_left, sumstd_right

-   This is a good resource to verify other scaling rules
    https://www.mathworks.com/help/ident/ug/extended-and-unscented-kalman-filter-algorithms-for-online-state-estimation.html
======================================================================================================================================#
import ArrayInterface.can_setindex
can_setindex(::Type{ZeroVec}) = false

"""
SigmaParams(α=1.0, κ=0.0, ϵ=1e-6)

Sigma point parameters for the Unscented Transform. The inner value is κ (\\kappa) which denotes a constant offest form the dimension 
(as defined in the classical single-value parameterization). There is an option to set α (\\alpha) instead which is a distance scalar 
(as is done in the three-value parameterization, but here it is only used to adjust κ). This package intead defaults to 
α=1 (or equivalently κ=0) so that points envelope ~50% of the distribution. Smaller values of α result in tighter-clumped 
values around the mean which can help with constraints. This package also uses autoscaling to help obey constraints which 
encourages users to do two things:

1. Clamp values inside the function as a first step (projection)
2. Define asymptotes/domain boundaries of functions if they exist (used for scaling)

Due to computational precision issues, an optional ϵ (\\epsilon) parameter is provided. A value of 1e-6 is a good balance for Float64 
it is large enough to enough precision on (1-ϵ^2) while being small enough to obey most constraints without clamping/projection. 
If estimates are unstable, ϵ may need to be increased at the cost of relying more on clamping/projection.
"""
struct SigmaParams
    κ :: Float64
    α :: Float64
    ϵ :: Float64
    proj :: Bool
end

function SigmaParams(; κ=0.0, α=NaN, ϵ=1e-6, proj=true)
    return isfinite(α) ? SigmaParams(NaN, α, ϵ, proj) : SigmaParams(κ, NaN, ϵ, proj) 
end

SigmaParams(κ::Number) = SigmaParams(κ, NaN, 1e-6, true)


"""
SigmaWeights(N::Integer, κ::SigmaParams)

Generates a vector-like object of SigmaWeights from a dimension number N and the classical unscanted transform parameter κ
"""
Base.@kwdef struct SigmaWeights{V<:Union{ConstVec{Float64},AbstractVector{Float64}}} <: AbstractVector{Float64}
    N  :: Int
    w0 :: Float64
    wi :: V
    ϵ :: Float64
    proj = true
end

function SigmaWeights(N::Integer, θ::SigmaParams)
    #Distance used to move sigma points
    δ = if !isnan(θ.κ) 
        sqrt(θ.κ + N) 
    elseif !isnan(θ.α) 
        θ.α*sqrt(N) 
    else 
        error("either α or κ must be set to a non-Nan value")
    end

    wi = 0.5/abs2(δ)
    w0 = 1 - 2*N*wi

    return SigmaWeights(N=N, w0=w0, wi=ConstVec(wi), ϵ=θ.ϵ, proj=θ.proj)
end

SigmaWeights(w::SigmaWeights) = w

Base.IndexStyle(::Type{<:SigmaWeights}) = IndexLinear()
Base.size(w::SigmaWeights) = (2*w.N + 1,)
Base.firstindex(w::SigmaWeights) = 0
Base.lastindex(w::SigmaWeights) = 2*w.N
Base.getindex(w::SigmaWeights, i::Int) = iszero(i) ? w.w0 : w.wi[i]
Base.axes(w::SigmaWeights) = map(n->0:(n-1), size(w))
Base.has_offset_axes(w::SigmaWeights) = true

"""
SigmaPoints{T}(source::T, weights::SigmaWeights)

Unscented transform using 2N+1 vectors as points
"""
Base.@kwdef struct SigmaPoints{S,T} <: AbstractVector{T}
    weights  :: SigmaWeights
    source   :: S
    
    SigmaPoints(w::SigmaWeights, source::UvGaussian) = new{typeof(source), meantype(source)}(w, source)
    SigmaPoints(w::SigmaWeights, source::MvGaussian) = new{typeof(source), meantype(source)}(w, source)
    SigmaPoints(w::SigmaWeights, source::AbstractGaussian) = new{typeof(source), meantype(source)}(w, source)
    SigmaPoints(w::SigmaWeights, source::Tuple{Vararg{<:UvGaussian}}) = new{typeof(source), meantype(source[begin])}(w, source)

    function SigmaPoints(w::SigmaWeights, source::AbstractVector) 
        Base.require_one_based_indexing(source)
        return new{typeof(source), eltype(source)}(w, source)
    end
end

SigmaPoints(θ::SigmaParams, x) = SigmaPoints(SigmaWeights(dimlength(x), θ), x)
SigmaPoints(θ::SigmaParams, x1, xn...) = SigmaPoints(SigmaWeights(length(xn)+1, θ), x1, xn...)
SigmaPoints(w::SigmaWeights, x1::UvGaussian, xn::UvGaussian...) = SigmaPoints(w, (x1, xn...))

Base.IndexStyle(::Type{<:SigmaPoints}) = IndexLinear()
Base.size(x::SigmaPoints) = size(x.weights)
Base.firstindex(x::SigmaPoints) = firstindex(x.weights)
Base.lastindex(x::SigmaPoints) = lastindex(x.weights)
Base.axes(x::SigmaPoints) = axes(x.weights)
Base.has_offset_axes(obj::SigmaPoints) = true

halfindex(x::SigmaPoints) = x.weights.N
dimlength(x::MvGaussian) = size(std(x), 1)
dimlength(x::UvGaussian) = 1
dimlength(tx::Tuple{Vararg{<:AbstractGaussian}}) = sum(dimlength, tx, init=0)
dimlength(X::SigmaPoints{<:AbstractGaussian}) = dimlength(X.source)
dimlength(X::SigmaPoints{<:UvGaussian}) = 1
dimlength(X::SigmaPoints{<:AbstractVector}) = length(X.source[begin])
dimlength(x::SigmaPoints{<:Tuple{Vararg{<:UvGaussian}}}) = length(x.source)

#If source is a vector, simply index it
Base.getindex(X::SigmaPoints{<:AbstractVector}, i::Int) = X.source[i+1]
Base.setindex!(X::SigmaPoints{<:AbstractVector}, v, i::Int) = setindex!(X.source, v, i)
Base.similar(X::SigmaPoints{<:AbstractGaussian}) = SigmaPoints(X.weights, Vector{eltype(X)}(undef, length(X)))
Base.similar(X::SigmaPoints, ::Type{S}, dims::Tuple{UnitRange}) where S = SigmaPoints(X.weights, Vector{S}(undef, length(dims[begin])))

mv_zero(X::SigmaPoints) = mv_zero(eltype(X[begin]), dimlength(X))
uv_zero(X::SigmaPoints) = uv_zero(typeof(X[begin]))

#If source is a Gaussian, generate the sigma point
function Base.getindex(X::SigmaPoints{<:AbstractGaussian}, i::Int)
    μ = meancol(X.source)
    i == firstindex(X) && return μ

    w  = X.weights
    δ  = sqrt(0.5/w[i])

    if firstindex(X) < i <= halfindex(X)
        return map((x, σij)-> x + δ*σij, μ, cholcol(X.source, i))
    elseif halfindex(X) < i <= lastindex(X)
        return map((x, σij)-> x - δ*σij, μ, cholcol(X.source, i-halfindex(X)))
    end 
    throw(BoundsError(X, i))
end

#If the source is a tuple of UvGaussians, generate the sigma point
function Base.getindex(X::SigmaPoints{<:Tuple{Vararg{<:UvGaussian}}}, i::Int)
    μ = map(mean, X.source)
    i == firstindex(X) && return μ

    δ = sqrt(0.5/X.weights[i])

    if firstindex(X) < i <= halfindex(X)
        iμ = i
        return @set μ[iμ] = μ[iμ] + δ*std(X.source[iμ])
    elseif halfindex(X) < i <= lastindex(X)
        iμ = i-halfindex(X)
        return @set μ[iμ] = μ[iμ] - δ*std(X.source[iμ])
    end 
    throw(BoundsError(X, i))
end

MvGaussian(X::SigmaPoints) = combine!(+, mv_zero(X), X)
UvGaussian(X::SigmaPoints) = combine(+, uv_zero(X), X)

#Dispatch patterns for generic function
gaussian(X::SigmaPoints{<:AbstractVector{<:AbstractVector}}) = MvGaussian(X)
gaussian(X::SigmaPoints{<:AbstractVector{<:Number}}) = UvGaussian(X)
gaussian(X::SigmaPoints{<:AbstractGaussian}) = X.source

#======================================================================================================================================
Combining Gaussians with SigmaPoints
======================================================================================================================================#
combine(op::Union{typeof(+), typeof(-)}, g::MvGaussian, X::SigmaPoints) = combine!(op, copy(g), X)

function combine!(op::Union{typeof(+), typeof(-)}, g::MvGaussian, X::SigmaPoints)
    w = X.weights
    x0 = X[begin]
    μ = vecupdate!(op, mean(g), x0, true)
    δ = Vector(μ)
    ch = densechol(std(g))

    for i in (firstindex(X)+1):lastindex(X)
        δ .= X[i] .- x0
        if !w.proj #Only update mean if projected is not used
            μ = vecupdate!(+, μ, δ, w[i])
        end
        cholupdate!(ch, δ, w[i])
    end
    return MvGaussian(μ, ch)
end

function combine(op::Union{typeof(+), typeof(-)}, g::UvGaussian, X::SigmaPoints)
    w  = X.weights
    x0 = X[begin]
    μ  = op(mean(g), x0)
    σ² = abs2(std(g))

    for i in (firstindex(X)+1):lastindex(X)
        δ = X[i] - x0
        if !w.proj #Only update mean if projected is not used
            μ  += w[i]*δ
        end
        σ² += w[i]*abs2(δ)
    end
    return UvGaussian(μ, sqrt(σ²))
end

#======================================================================================================================================
Combining Gaussians with each other
======================================================================================================================================#
combine(op::Union{typeof(+), typeof(-)}, g1::MvGaussian, g2::MvGaussian) = combine!(op, copy(g1), g2)
combine(op::Union{typeof(+), typeof(-)}, g1::UvGaussian, g2::UvGaussian) = UvGaussian(op(mean(g1), mean(g2)), sqrt(cov(g1)+cov(g2)))

function combine!(op::Union{typeof(+), typeof(-)}, g1::MvGaussian, g2::MvGaussian)
    μ = vecupdate!(op, mean(g1), mean(g2), true)
    σ = densechol(std(g1))
    δ = Vector(cholcol(g2, 1))

    for i in axes(σ.L, 2)
        δ .= cholcol(g2, i)
        lowrankupdate!(σ, δ)
    end

    return MvGaussian(μ, σ)
end

#======================================================================================================================================
Helper functions for adding covariances in square root form
======================================================================================================================================#
function cov(X::SigmaPoints, Y::SigmaPoints)
    w = X.weights
    weight(ii::Integer) = ifelse(ii==1, X.weights.Wσ, X.weights.Wn)

    (nx, ny) = (length(X), length(Y))
    if nx != ny
        error("Two sets of sigma points must have the same number of points ($(nx) ≠ $(ny))")
    end

    (μx, μy) = (X[begin], Y[begin])
    T = promote_type(Float64, eltype(μx), eltype(μy))
    S = zeros(T, length(μx), length(μy))
    inds = (firstindex(X)+1):lastindex(X)

    for i in inds
        wi = w[i]
        xi = X[i]
        yi = Y[i]
        S .+= wi .* (xi.-μx) .* (yi.-μy)'
    end
    return S
end


"""
choladdleft(ch::Cholesky, L::AbstractMatrix)

Eqivalent of `cholesky(ch.U'ch.U + L*L')`
"""
choladdleft(ch::Cholesky, L::AbstractMatrix) = choladdleft!(copy(ch), L)
choladdleft(d::Diagonal, L::AbstractMatrix) = choladdleft!(densechol(d), L)
choladdleft(ch::Cholesky{<:Any, <:Diagonal}) = choladdleft!(densechol(ch), L)

function choladdleft!(ch::Cholesky, L::AbstractMatrix)
    x = zeros(eltype(L), size(L, 1))

    for xi in eachcol(L)
        x .= xi
        lowrankupdate!(ch, x)
    end
    return ch
end

"""
choladdleft(A::AbstractMatrix, B::AbstractMatrix)

Equivalent of cholesky(A*A' + B*B')
"""
function choladdleft(A::AbstractMatrix, B::AbstractMatrix)
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
cholsubleft(ch::Cholesky, L::AbstractMatrix) = cholsubleft!(deepcopy(ch), L)

function cholsubleft!(ch::Cholesky, L::AbstractMatrix)
    x = zeros(eltype(L), size(L, 1))

    for xi in eachcol(L)
        x .= xi
        lowrankdowndate!(ch, x)
    end
    return ch
end

"""
chol_update!(ch::Cholesky, x::AbstractVector, w::Real)

Updates cholesky decomposition it gives the equivalent of 
cholesky(ch.U'*ch.U + w*(x'*x))
This function is non-allocating and the vector "x" is destroyed in the process
"""
function cholupdate!(ch::Cholesky, x::Vector, w::Real)
    x .= sqrt(abs(w)) .* x
    return w >= 0 ? lowrankupdate!(ch, x) : lowrankdowndate!(ch, x)
end

function vecupdate!(op::Union{typeof(+), typeof(-)}, μ::AbstractVector, x::AbstractVector, w::Real)
    if can_setindex(μ)
        μ .+= op(w).*x
        return μ
    end
    return μ .+ op(w).*x
end


vecupdate!(op::Union{typeof(+), typeof(-)}, μ::ZeroVec, x::AbstractVector, w::Real) = x*op(w)
vecupdate!(op::Union{typeof(+), typeof(-)}, μ::AbstractVector, x::ZeroVec, w::Real) = μ
vecupdate!(op::Union{typeof(+), typeof(-)}, μ::ZeroVec, x::ZeroVec, w::Real) = ZeroVec()

cholvar(ch::Cholesky) = map(ii->cholvar(ch, ii), axes(ch.U, 2))
cholstd(ch::Cholesky) = map(ii->cholstd(ch, ii), axes(ch.U, 2))

cholstd(ch::Cholesky, ii::Integer) = sqrt(cholvar(ch, ii))
function cholvar(ch::Cholesky, ii::Integer)
    v = view(ch.U, :, ii)
    return dot(v,v)
end


Base.isfinite(x::MvGaussian) = all(isfinite, x.μ) & all(isfinite, x.σ.L)
