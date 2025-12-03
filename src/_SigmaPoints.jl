using LinearAlgebra
import Statistics.mean
import Statistics.cov

"""
GaussianVar(x, Σ)

Random vector that follows a Gaussian distribution. 
If passed a matrix, the constructor automatically takes Cholesky decomposition.
"""
@kwdef struct GaussianVar{TX<:AbstractVector, TM<:Cholesky}
    μ :: TX
    Σ :: TM
end
GaussianVar(x::AbstractVector, m::AbstractMatrix) = GaussianVar(x, cholesky(m))

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
SigmaParams(α = 0.001, κ = 0.0, β = 2.0)

Sigma point parameters for the Unscented Transform (α~0 => Linear Gaussian, α~1=> Nonlinear Gaussian, κ=0, β=2 for Gaussian)
"""
Base.@kwdef struct SigmaParams
    α :: Float64 = 0.001
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

function SigmaPoints(x::GaussianVar, w::SigmaWeights)
    σc = sqrt(w.c)
    points = [x.μ]
    
    for l in eachcol(x.Σ.L)
        Δ = σc.*l
        push!(points, x.μ .+ Δ)
        push!(points, x.μ .- Δ)
    end

    return SigmaPoints(points=points, weights=w)
end
SigmaPoints(x::GaussianVar, θ::SigmaParams) = SigmaPoints(x, SigmaWeights(x.μ, θ))

GaussianVar(𝒳::SigmaPoints) = GaussianVar(mean(𝒳), cholesky(cov(𝒳)))

"""
add_cov(𝒳::SigmaPoints, Σ::Cholesky)

Creates a GaussianVar from 𝒳 and adds Σ to the variance
"""
function add_cov(𝒳::SigmaPoints, Σ::Cholesky)
    ch = deepcopy(Σ)

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

    return GaussianVar(μ, ch)
end


"""
add_cov!(ch::Cholesky, ch2::Cholesky)

Updates cholesky decomposition ch to be the equivalent of
cholesky(ch.U'ch.U + ch2.U'+ch2.U)
"""
function add_cov!(ch::Cholesky, ch2::Cholesky)
    x = zeros(eltype(ch2.U), size(ch2.U, 1))

    #Add all of the subsequent points
    for xi in eachcol(ch2.L)
        x .= xi
        lowrankupdate!(ch, x)
    end

    return ch
end
add_cov(ch::Cholesky, ch2::Cholesky) = add_cov!(deepcopy(ch), ch2)

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



#Scale the innoviation to avoid chasing outliers
function scale_innovation(Δy::Real, σy::Real; outlier)
    if isfinite(outlier)
        σε = (outlier/3)*σy
        return asinh(Δy/σε)*σε
    else
        return Δy
    end
end

function chol_std(ch::Cholesky, ii::Integer)
    sqrtdot(x) = sqrt(dot(x,x))
    return sqrtdot(view(ch.U, :, ii))
end