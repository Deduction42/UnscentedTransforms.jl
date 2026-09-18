using LinearAlgebra
import Statistics.mean
import Statistics.cov
import Statistics.std
import Statistics.var

"""
MvGaussian(x, Σ)

Random vector that follows a Gaussian distribution. 
If passed a matrix, the constructor automatically takes Cholesky decomposition.
"""
@kwdef struct MvGaussian{TX<:AbstractVector, TM<:Cholesky}
    μ :: TX
    Σ :: TM
end
MvGaussian(x::AbstractVector, m::AbstractMatrix) = MvGaussian(x, cholesky(m))
Base.convert(::Type{MvGaussian{TX,TM}}, x::MvGaussian) where {TX,TM} = MvGaussian(TX(x.μ), TM(x.Σ))

"""
SigmaWeights(c :: Float64, μ :: Tuple{Float64, Float64}, Σ :: Tuple{Float64, Float64})

Weights for sigma points, calculated from SigmaParams and the state dimension L
"""
Base.@kwdef struct SigmaWeights
    c :: Float64
    μ :: Tuple{Float64, Float64}
    Σ :: Tuple{Float64, Float64}
end

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

function SigmaWeights(L::Int64, θ::SigmaParams=SigmaParams())
    α = θ.α
    κ = θ.κ
    β = θ.β

    λ  = α^2*(L+κ)-L                          #scaling factor
    c  = L + λ                                #scaling factor
    Wn = 0.5/c
    Wμ = ((λ/c), Wn)                         #weights for means
    WΣ = (Wμ[1] + (1-α^2+β), Wn)             #weights for covariance
    return SigmaWeights(c=c, μ=Wμ, Σ=WΣ)
end
SigmaWeights(v::AbstractVector, θ::SigmaParams=SigmaParams()) = SigmaWeights(length(v), θ)

"""
SigmaPoints{T<:AbstractVector}(points::Vector{T}, weights::SigmaWeights)

Unscented transform using L+1 vectors as points
"""
Base.@kwdef struct SigmaPoints{T<:AbstractVector}
    points   :: Vector{T}
    weights  :: SigmaWeights
end

SigmaPoints(X::MvGaussian, θ::SigmaParams) = SigmaPoints(X, SigmaWeights(length(X.μ), θ))

function SigmaPoints(X::MvGaussian, w::SigmaWeights)
    σc = sqrt(w.c)
    points = [X.μ]
    
    for l in eachcol(X.Σ.L)
        Δ = σc.*l
        push!(points, X.μ .+ Δ)
        push!(points, X.μ .- Δ)
    end

    return SigmaPoints(points=points, weights=w)
end


MvGaussian(𝒳::SigmaPoints) = MvGaussian(mean(𝒳), cholesky(cov(𝒳)))

"""
MvGaussian(𝒳::SigmaPoints, Σ::Cholesky)

Returns the GuassianVar equivalent of adding variance Σ to 𝒳
"""
function MvGaussian(𝒳::SigmaPoints, Σ::Cholesky)
    ch = copy(Σ)

    (w0, w1) = (𝒳.weights.Σ[1], 𝒳.weights.Σ[2])
    x = zeros(eltype(𝒳.points[begin]), length(𝒳.points[begin]))
    μ = mean(𝒳)

    #Add all of the surrounding points
    for ii in (firstindex(𝒳.points)+1):lastindex(𝒳.points)
        x .= 𝒳.points[ii] .- μ
        chol_update!(ch, x, w1)
    end

    #Add central point (where weight could be negative) 
    #Because of negative weight, doing this last reduces risk of negative covariacne
    x .= 𝒳.points[begin] .- μ
    chol_update!(ch, x, w0)

    return MvGaussian(μ, ch)
end
MvGaussian(Σ::Cholesky, 𝒳::SigmaPoints) = MvGaussian(𝒳, Σ)

"""
add_cov(ch::Cholesky, ch2::Cholesky)

Returns a cholesky decomposition equivalent to performing
cholesky(ch.U'*ch.U + ch2.U'*ch2.U)
"""
add_cov(ch::Cholesky, ch2::Cholesky) = add_cov!(copy(ch), ch2)

function add_cov!(ch::Cholesky, ch2::Cholesky)
    x = zeros(eltype(ch2.U), size(ch2.U, 1))

    for xi in eachcol(ch2.L)
        x .= xi
        lowrankupdate!(ch, x)
    end
    return ch
end


"""
add_lcov(ch::Cholesky, L::AbstractMatrix)

Updates cholesky decomposition ch to be the equivalent of
cholesky(ch.U'ch.U + L*L')
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

Returns the equivalent of
cholesky(A'A + B'B)
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

Returns the equivalent of
cholesky(A*A' + B*B')
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
sub_cov_sqrt(ch::Cholesky, L::AbstractMatrix)

Updates cholesky decomposition ch to be the equivalent of
cholesky(ch.U'ch.U - L*L')
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
Returns a weighted mean vector of a set of sigma points
"""
function mean(𝒳::SigmaPoints{T}) where T
    wμ = 𝒳.weights.μ
    centerpoint = first(𝒳.points)
    outerpoints = @view 𝒳.points[(begin+1):end]

    μ = wμ[1].*centerpoint

    if ismutable(μ)
        for xi in outerpoints
            μ .+= wμ[2].*xi
        end
        return μ
    else
        return sum(xi-> wμ[2].*xi, outerpoints, init=μ)
    end
end


"""
Returns a weighted covariance matrix of two sets of sigma points, based on weights from the first set
"""
function cov(𝒳::SigmaPoints{T1}, 𝒴::SigmaPoints{T2}) where {T1, T2}
    weight(ii::Integer) = ifelse(ii==1, 𝒳.weights.Σ[1], 𝒳.weights.Σ[2])

    (nx, ny) = (length(𝒳.points), length(𝒴.points))
    if nx != ny
        error("Two sets of sigma points must have the same number of points")
    end


    (μx, μy) = (mean(𝒳), mean(𝒴))
    T = promote_type(Float64, eltype(T1), eltype(T2))
    S = zeros(T, length(first(𝒳.points)), length(first(𝒴.points)))
    ii = 0
    for (x, y) in zip(𝒳.points, 𝒴.points)
        ii += 1
        S .+= weight(ii) .* (x.-μx) .* (y.-μy)'
    end
    return S
end

function cov(𝒳::SigmaPoints{T1}) where T1
    weight(ii::Integer) = ifelse(ii==1, 𝒳.weights.Σ[1], 𝒳.weights.Σ[2])

    nx = length(first(𝒳.points))
    μx = mean(𝒳)
    T  = promote_type(Float64, eltype(T1))
    S  = zeros(T, nx, nx)
    ii = 0
    for x in 𝒳.points
        S .+= weight(ii) .* (x.-μx) .* (x.-μx)'
    end
    hermitianpart!(S) 

    return S
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

std(x::MvGaussian)  = chol_std(x.Σ)
var(x::MvGaussian)  = chol_var(x.Σ)
mean(x::MvGaussian) = x.μ

chol_var(ch::Cholesky) = map(ii->chol_var(ch, ii), axes(ch.U, 2))
chol_std(ch::Cholesky) = map(ii->chol_std(ch, ii), axes(ch.U, 2))

chol_std(ch::Cholesky, ii::Integer) = sqrt(chol_var(ch, ii))
function chol_var(ch::Cholesky, ii::Integer)
    v = view(ch.U, :, ii)
    return dot(v,v)
end


Base.isfinite(x::MvGaussian) = all(isfinite, x.μ) & all(isfinite, x.Σ.U)