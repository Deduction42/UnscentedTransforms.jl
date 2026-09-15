#======================================================================================================================================
Math functions
======================================================================================================================================#
Base.:+(x::Number, g::UvGaussian) = UvGaussian(x + g.μ, g.σ)
Base.:+(g::UvGaussian, x::Number) = UvGaussian(x + g.μ, g.σ)
Base.:+(g1::UvGaussian, g2::UvGaussian) = UvGaussian(g1.μ + g2.μ, add_cov(g1.σ, g2.σ))
Base.:+(g1::MvGaussian, g2::MvGaussian) = MvGaussian(g1.μ + g2.μ, add_cov(g1.Σ, g2.Σ))

function Base.:+(X::SigmaPoints, g::MvGaussian) 
    μx = mean(X)
    return MvGaussian(mean(g) + μx, add_cov(g.Σ, X, μx))
end
Base.:+(g::MvGaussian, X::SigmaPoints) = g + X

Base.:*(x::Number, g::UvGaussian) = UvGaussian(x*g.μ, x*g.σ)
Base.:*(g::UvGaussian, x::Number) = UvGaussian(x*g.μ, x*g.σ)
Base.:*(x::Number, g::MvGaussian{<:Cholesky}) = MvGaussian(x*g.μ, Cholesky(x*g.Σ.U))
Base.:*(g::MvGaussian{<:Cholesky}, x::Number) = MvGaussian(x*g.μ, Cholesky(x*g.Σ.U))
Base.:*(x::Number, g::MvGaussian{<:Diagonal}) = MvGaussian(x*g.μ, x*g.Σ)
Base.:*(g::MvGaussian{<:Diagonal}, x::Number) = MvGaussian(x*g.μ, x*g.Σ)

predict(f, θ::SigmaParams, args::UvGaussian...) = predict(f, SigmaWeights(length(args), θ), args...)

function predict(f, θ::SigmaWeights, args::UvGaussian...)
    Np = 2*length(args) + 1
    
    old_points = SigmaPoints(args, θ)
    indvec = SVector{Np}(firstindex(old_points):lastindex(old_points))
    new_points = map(ind->f(old_points[ind]...), indvec) #Non-allocating result

    return gaussian(new_points)
end

"""
domainlimits(f::Function)

Species values in the domain "f" that a distribution should not cross. 
    -   This could pertain to an out-of-domain limit like 0 for sqrt(x)
        Well behaved cases must have all positive values, negative values are ill-defined
    -   This could also pertain to an asymptote like 0 for inv(x) 
        Well-behaved cases must have the same sign as the mean, zero-crossing behaviour is ill-defined
"""
domainlimits(f::Function) = ()
domainlimits(f::typeof(log)) = (0,)
domainlimits(f::typeof(sqrt)) = (0,)
domainlimits(f::typeof(inv)) = (0,)
domainlimits(f::typeof(asin)) = (0,1)
domainlimits(f::typeof(acos)) = (0,1)

function select_alpha(f, g::AbstractGaussian)
    current = 1.0
    
    limits = domainlimits(f)
    (limits isa Tuple) || error("domainlimits($(f)) must return a tuple, instead it returned $(limits)")

    for lim in limits 
        current = _min_scale_deviation(current, g, lim)
    end

    return current 
end

function _min_scale_deviation(current::Number, g::MvGaussian, limit::AbstractVector)
    Base.require_one_based_indexing(limit)
    μ = mean(g)

    for i in eachindex(μ)
        gi = UvGaussian(μ[i] + maximum(abs, cholrow(g, i)))
        current = _min_scale_deviation(current, gi, limit[i])
    end

    return current 
end

function _min_scale_deviation(current::Number, g::UvGaussian{T}, limit::Number) where T
    z  = abs(convert(T, limit) - g.μ)/g.σ
    zs = z*(1 + inv(z+1))/2
    return ifelse(isfinite(limit), convert(typeof(current), zs), current)
end