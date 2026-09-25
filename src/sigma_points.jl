#======================================================================================================================================
ToDo:

Post cleanup:
(1) Redefine add_cov to sumstd, sumstd_left, sumstd_right

-   This is a good resource to verify other scaling rules
    https://www.mathworks.com/help/ident/ug/extended-and-unscented-kalman-filter-algorithms-for-online-state-estimation.html
======================================================================================================================================#


"""
SigmaParams(α=1.0, κ=0.0, αmin=1e-6)

Sigma point parameters for the Unscented Transform. The inner value is κ which denotes a constant offest form the dimension 
(as defined in the classical single-value parameterization). There is an option to set α instead which is a distance scalar 
(as is done in the three-value parameterization, but here it is only used to adjust κ). This package intead defaults to 
α=1 (or equivalently κ=0) so that points envelope ~50% of the distribution. Smaller values of α result in tighter-clumped 
values around the mean which can help with constraints. This package also uses autoscaling to help obey constraints which 
encourages users to do two things:

1. Clamp values inside the function as a first step (projection)
2. Define asymptotes/domain boundaries of functions if they exist (used for scaling)

Due to computational precision issues, an optional αmin parameter is provided. A value of 1e-6 is a good balance for Float64 
it is large enough to enough precision on (1-αmin^2) while being small enough to obey most constraints without clamping/projection. 
If estimates are unstable, αmin may need to be increased at the cost of relying more on clamping/projection.
"""
struct SigmaParams
    κ :: Float64
    α :: Float64
    αmin :: Float64
end

function SigmaParams(; κ=0.0, α=NaN, αmin=1e-6)
    return isfinite(α) ? SigmaParams(NaN, α, αmin) : SigmaParams(κ, NaN, αmin) 
end

SigmaParams(κ::Number) = SigmaParams(κ, NaN, 1e-6)


"""
SigmaWeights(N::Integer, κ::SigmaParams)

Generates a vector-like object of SigmaWeights from a dimension number N and the classical unscanted transform parameter κ
"""
Base.@kwdef struct SigmaWeights{V<:Union{ConstVec{Float64},AbstractVector{Float64}}} <: AbstractVector{Float64}
    N  :: Int
    w0 :: Float64
    wi :: V
    αmin :: Float64
end

function SigmaWeights(N::Integer, κ::SigmaParams)
    #Distance used to move sigma points
    δ = if !isnan(κ.κ) 
        sqrt(κ.κ + N) 
    elseif !isnan(κ.α) 
        κ.α*sqrt(N) 
    else 
        error("either α or κ must be set to a non-Nan value")
    end

    wi = 0.5/abs2(δ)
    w0 = 1 - 2*N*wi

    return SigmaWeights(N=N, w0=w0, wi=ConstVec(wi), αmin=κ.αmin)
end

SigmaWeights(θ::SigmaWeights) = θ

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
    
    SigmaPoints(θ::SigmaWeights, source::UvGaussian) = new{typeof(source), meantype(source)}(θ, source)
    SigmaPoints(θ::SigmaWeights, source::MvGaussian) = new{typeof(source), meantype(source)}(θ, source)
    SigmaPoints(θ::SigmaWeights, source::AbstractGaussian) = new{typeof(source), meantype(source)}(θ, source)
    SigmaPoints(θ::SigmaWeights, source::Tuple{Vararg{<:UvGaussian}}) = new{typeof(source), meantype(source[begin])}(θ, source)

    function SigmaPoints(θ::SigmaWeights, source::AbstractVector) 
        Base.require_one_based_indexing(source)
        return new{typeof(source), eltype(source)}(θ, source)
    end
end

SigmaPoints(θ::SigmaParams, x) = SigmaPoints(SigmaWeights(dimlength(x), θ), x)
SigmaPoints(θ::SigmaParams, x1, xn...) = SigmaPoints(SigmaWeights(length(xn)+1, θ), x1, xn...)
SigmaPoints(θ::SigmaWeights, x1::UvGaussian, xn::UvGaussian...) = SigmaPoints(θ, (x1, xn...))

Base.IndexStyle(::Type{<:SigmaPoints}) = IndexLinear()
Base.size(x::SigmaPoints) = size(x.weights)
Base.firstindex(x::SigmaPoints) = firstindex(x.weights)
Base.lastindex(x::SigmaPoints) = lastindex(x.weights)
Base.axes(x::SigmaPoints) = axes(x.weights)
Base.has_offset_axes(obj::SigmaPoints) = true

halfindex(x::SigmaPoints) = x.weights.N
dimlength(x::AbstractGaussian) = length(mean(x))
dimlength(x::UvGaussian) = 1
dimlength(tx::Tuple{Vararg{<:AbstractGaussian}}) = sum(dimlength, tx, init=0)
dimlength(X::SigmaPoints{<:AbstractGaussian}) = length(X.source)
dimlength(X::SigmaPoints{<:UvGaussian}) = 1
dimlength(X::SigmaPoints{<:AbstractVector}) = length(X.source[begin])
dimlength(x::SigmaPoints{<:Tuple{Vararg{<:UvGaussian}}}) = length(x.source)

#If source is a vector, simply index it
Base.getindex(X::SigmaPoints{<:AbstractVector}, i::Int) = X.source[i+1]
Base.setindex!(X::SigmaPoints{<:AbstractVector}, v, i::Int) = setindex!(X.source, v, i)
Base.similar(X::SigmaPoints{<:AbstractGaussian}) = SigmaPoints(X.weights, Vector{eltype(X)}(undef, length(X)))
Base.similar(X::SigmaPoints, ::Type{S}, dims::Tuple{UnitRange}) where S = SigmaPoints(X.weights, Vector{S}(undef, length(dims[begin])))

#If source is a Gaussian, generate the sigma point
function Base.getindex(X::SigmaPoints{<:AbstractGaussian}, i::Int)
    #display(i)
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

MvGaussian(X::SigmaPoints) = MvGaussian(mean(X), std(X))
UvGaussian(X::SigmaPoints) = UvGaussian(mean(X), std(X))


#Dispatch patterns for generic function
gaussian(X::SigmaPoints{<:AbstractVector{<:AbstractVector}}) = MvGaussian(X)
gaussian(X::SigmaPoints{<:AbstractVector{<:Number}}) = UvGaussian(X)
gaussian(X::SigmaPoints{<:AbstractGaussian}) = X.source


#======================================================================================================================================
Stats functions
======================================================================================================================================#
function mean(X::SigmaPoints{<:AbstractVector})
    w = X.weights
    μ = w[begin].*X[begin]
    outer_inds = (firstindex(X)+1):lastindex(X)

    if ismutable(μ)
        for ind in outer_inds
            μ .+= w[ind].*X[ind]
        end
        return μ
    else
        return sum(ind-> w[ind].*X[ind], outer_inds, init=μ)
    end
end

function mean(X::SigmaPoints{<:AbstractVector{<:Number}}) 
    w = X.weights
    return sum(ind-> w[ind]*X[ind], eachindex(X))
end


mean(X::SigmaPoints{<:AbstractGaussian}) = mean(X.source)

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

function std(X::SigmaPoints{<:AbstractVector{<:AbstractVector}}) 
    x0 = X[begin]
    nd = dimlength(X) 
    ch = Cholesky(LowerTriangular(zeros(eltype(x0), nd, nd)))
    add_cov!(ch, X)
    return ch 
end

function std(X::SigmaPoints{<:AbstractVector{<:Number}})
    w = X.weights 
    μ = X[begin] 
    inds = (firstindex(X)+1):lastindex(X)

    σ² = sum(i-> w[i]*(X[i]-μ)^2, inds)
    return sqrt(σ²)
end


std(X::SigmaPoints{<:AbstractGaussian}) = std(X.source)

function cov(x::SigmaPoints)
    ch = std(x)
    return AbstractMatrix(ch)
end

#======================================================================================================================================
Helper functions for adding covariances in square root form
======================================================================================================================================#
"""
add_cov(σ1, σ2)

Returns the square-root form of adding covariances
"""
function add_cov end

add_cov(ch::Cholesky, X::SigmaPoints) = add_cov!(copy(ch), X)
add_cov(d::Diagonal, X::SigmaPoints) = add_cov!(diag2chol(d), X)

function add_cov!(ch::Cholesky, X::SigmaPoints)
    w = X.weights
    x = zeros(eltype(X[begin]), length(X[begin])) #Temporary storage vector that gets destroyed
    μ = X[begin] #Using first element of mean is "modified UKF" form which guarantees proper covariance
    inds = (firstindex(X)+1):lastindex(X)

    #Add all of the surrounding points
    for i in inds
        x .= X[i] .- μ
        chol_update!(ch, x, w[i])
    end

    return ch
end

add_cov(ch1::Cholesky, ch2::Cholesky) = add_cov!(copy(ch1), ch2)
add_cov(x1::Number, x2::Number) = sqrt(abs2(x1) + abs2(x2))

function add_cov!(ch1::Cholesky, ch2::Cholesky)
    x = zeros(eltype(ch2.L), size(ch2.L, 1))

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
    return Cholesky(LowerTriangular(R'))
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


diag2chol(d::Diagonal) = Cholesky(LowerTriangular(Matrix(d)))

chol_var(ch::Cholesky) = map(ii->chol_var(ch, ii), axes(ch.U, 2))
chol_std(ch::Cholesky) = map(ii->chol_std(ch, ii), axes(ch.U, 2))

chol_std(ch::Cholesky, ii::Integer) = sqrt(chol_var(ch, ii))
function chol_var(ch::Cholesky, ii::Integer)
    v = view(ch.U, :, ii)
    return dot(v,v)
end


Base.isfinite(x::MvGaussian) = all(isfinite, x.μ) & all(isfinite, x.σ.L)
