
#======================================================================================================================================
Linear univariate functions (exact shortcuts)
======================================================================================================================================#
Base.:+(x::Number, g::UvGaussian) = UvGaussian(x + mean(g), std(g))
Base.:+(g::UvGaussian, x::Number) = UvGaussian(x + mean(g), std(g))
Base.:+(g1::UvGaussian, g2::UvGaussian) = combine(+, g1, g2)

Base.:-(g::UvGaussian) = UvGaussian(-mean(g), std(g))
Base.:-(x::Number, g::UvGaussian) = UvGaussian(x - mean(g), std(g))
Base.:-(g::UvGaussian, x::Number) = UvGaussian(mean(g) - x, std(g))
Base.:-(g::UvGaussian, x::AbstractVector) = MvGaussian(mean(g) - x, std(g))
Base.:-(g1::UvGaussian, g2::UvGaussian) = combine(-, g1, g2)

Base.:*(x::Number, g::UvGaussian) = UvGaussian(x*mean(g), x*std(g))
Base.:*(g::UvGaussian, x::Number) = UvGaussian(x*mean(g), x*std(g))

#======================================================================================================================================
Linear multivariate functions (exact shortcuts)
======================================================================================================================================#
Base.:+(x::AbstractVector, g::MvGaussian) = MvGaussian(x + mean(g), std(g))
Base.:+(g::MvGaussian, x::AbstractVector) = MvGaussian(x + mean(g), std(g))
Base.:+(g1::MvGaussian, g2::MvGaussian) = combine(+, g1, g2)
Base.:+(X::SigmaPoints, g::MvGaussian) = combine(+, g, X)
Base.:+(g::MvGaussian, X::SigmaPoints) = combine(+, g, X)

Base.:-(v::ZeroVec) = v
Base.:-(g::MvGaussian) = MvGaussian(-mean(g), std(g))
Base.:-(x::AbstractVector, g::MvGaussian) = MvGaussian(x - mean(g), std(g))
Base.:-(g1::MvGaussian, g2::MvGaussian) = combine(-, g1, g2)
Base.:-(X::SigmaPoints, g::MvGaussian) = combine(-, g, X)
Base.:-(g::MvGaussian, X::SigmaPoints) = combine(-, g, X)

Base.:*(x::Number, g::MvGaussian{<:Cholesky}) = MvGaussian(x*mean(g), Cholesky(x*std(g).L))
Base.:*(g::MvGaussian{<:Cholesky}, x::Number) = MvGaussian(x*mean(g), Cholesky(x*std(g).L))
Base.:*(x::Number, g::MvGaussian{<:Diagonal}) = MvGaussian(x*mean(g), x*std(g))
Base.:*(g::MvGaussian{<:Diagonal}, x::Number) = MvGaussian(x*mean(g), x*std(g))

function Base.:*(A::AbstractMatrix, x::MvGaussian)
    μ = A*mean(x)
    σ = choladdleft!(zerochol(μ), A*std(x).L)
    return MvGaussian(μ, σ)
end

function Base.muladd(A::AbstractMatrix, x::MvGaussian, ε::MvGaussian)
    μ = A*mean(x) + mean(ε)
    σ = choladdleft(std(ε), A*std(x).L)
    return MvGaussian(μ, σ)
end

#======================================================================================================================================
Unscented factory methods
======================================================================================================================================#
function gaussian(f, θ::SigmaParams, args::UvGaussian...)
    Nd = length(args)
    Np = 2*Nd + 1
    w  = scale_weights(f, SigmaWeights(Nd, θ), args...)

    X0 = SigmaPoints(w, args...)
    indvec = SVector{Np}(firstindex(X0):lastindex(X0))
    X1 = SigmaPoints(w, map(ind->f(X0[ind]...), indvec)) #Non-allocating result

    return gaussian(X1)
end

function gaussian(f, θ::SigmaParams, x::MvGaussian)
    Nd = dimlength(x)
    w  = scale_weights(f, SigmaWeights(Nd, θ), x)
    X  = map(f, SigmaPoints(w, x))
    return gaussian(X)
end

#======================================================================================================================================
Joint Probabilities
======================================================================================================================================#
joint(x::AbstractGaussian) = x

function joint(x1::UvGaussian, x2::UvGaussian)
    σ² = inv(inv(cov(x1)) + inv(cov(x2)))
    μ  = σ²*(mean(x1)/cov(x1) + mean(x2)/cov(x2))
    return UvGaussian(μ, sqrt(σ²))
end

function joint(x1::MvGaussian, x2::MvGaussian)
    z = x2 - x1 #Innovation distribution
    K = cov(x1)/std(z) #Kalman gain

    #New mean
    μ = x1.μ .+ K*z.μ

    #New lower covariance, this sometimes fails because rounding error on subtraction makes the matrix "negative", skip if that happens
    σ = try
        cholsubleft(std(x1), K*std(z).L)
    catch err
        @warn "Covariance update failed, skipping this step:\n" * sprint(showerror, err)
        x1.σ
    end
    return MvGaussian(μ, σ)
end

joint(x1::UvGaussian, x2::UvGaussian, xN::UvGaussian...) = joint(joint(x1, x2), xN...)
joint(x1::MvGaussian, x2::MvGaussian, xN::MvGaussian...) = joint(joint(x1, x2), xN...)

#======================================================================================================================================
Nonlinear univariate functions
======================================================================================================================================#

#Special definitions 
Base.:≈(x1::AbstractGaussian, x2::AbstractGaussian) = (mean(x1)≈mean(x2)) & (cov(x1)≈cov(x2))
Base.:*(x1::UvGaussian, x2::UvGaussian) = gaussian(*, SigmaParams(), x1, x2)
Base.:/(x1::UvGaussian, x2::UvGaussian) = gaussian(*, SigmaParams(), x1, inv(x2))
domainlimits(f::typeof(inv), ::Type{<:Number}) = ArgLimits(0)

#Periodic asymptotes
function Base.tan(x::UvGaussian)
    μ = mean(x) - round(Int64, mean(x)/(2π))*2π
    return gaussian(_rngtan, SigmaParams(), UvGaussian(μ, std(x)))
end

#Generic definitions
for f in (:inv, :abs, :abs2, :sin, :cos, :sinh, :cosh, :tanh,
          :exp, :exp2, :exp10, :expm1, :frexp, :exponent)
    @eval Base.$f(x::UvGaussian) = gaussian($f, SigmaParams(), x)
end

#Functions requiring positive arguments
for f in (:sqrt, :cbrt, :log, :log2, :log10)
    @eval Base.$f(x::UvGaussian) = gaussian(x->$f(max(x, 0)), SigmaParams(), x)
    @eval domainlimits(g::typeof($f), ::Type{<:Real}) = ArgLimits(0)
end

#Functions requiring intervals between 0 and 1
for f in (:asin, :acos,)
    @eval Base.$f(x::UvGaussian) = gaussian(x->$f(clamp(x, 0, 1)), SigmaParams(), x)
    @eval domainlimits(g::typeof($f), ::Type{<:Real}) = ArgLimits(0, 1)
end


#Tangent function is tricky due to periodic asymptotes
function _rngtan(x::Number)
    (-π < x < π) || throw(DomainError(x, "Argument must be between -π and π"))
    return tan(x)
end
domainlimits(f::typeof(_rngtan), ::Type{<:Number}) = ArgLimits(-π, π)