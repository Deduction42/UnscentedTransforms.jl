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
discontinuities(f::Function)

Specifies any discontinutities in "f" that a distribution should not cross. 
    -   This could pertain to an out-of-domain limit like 0 for sqrt(x)
        Well behaved cases must have all positive values, negative values are ill-defined
    -   This could also pertain to an asymptote like 0 for inv(x) 
        Well-behaved cases must have the same sign as the mean, zero-crossing behaviour is ill-defined

This function must produce a tuple of discontinuities. The default is no discontinuities.
    `discontinuities(f::Function) = ()`

Examples:

(1) For a single-argument function with a single discontinuitiy at zero (like inv) you would want 
    `discontinuities(f::typeof(inv)) = (0,)`

(2) For a single-argument function with two discontinuties (like asin at 0 and 1), you would want 
    `discontinuities(f::typeof(asin)) = (0,1)`

(3) For a vector-argument function with a single discontinuity, return a tuple of a vector. If any element 
    of the vector doesn't matter, simply place a non-fininte number in that element
    `vecdiv(x) = x[1]/x[2]`
    `discontinuities(typeof(vecdiv)) = ([NaN,0], )`

(4) For a two-argument function with a single disonctinuity, use a non-finite number for any argument that doesn't matter.
    `mydiv(x,y) = x/y`
    `discontinuities(f::typeof(mydiv)) = ((NaN, 0),)`

(5) Currently, only "box discontinuities" are supported, which means that you can lump multiple discontinuities in a single object 
    `invmul(x,y) = inv(x*y)`
    `discontinuities(f::typeof(invmul)) = ((0,0),)`

(6) Box discontinuitieis also means can also repeat discontinuities or place NaN if that argument's discontinuity is already taken care of
    `myfunc(x,y) = log(x*asin(y))`
    `discontinuities(f::typeof(myfunc)) = ((0,0), (NaN,1)) #This is valid`
    `discontinuities(f::typeof(myfunc)) = ((0,0), (0,1)) #This expresses the same thing`

(7) Multidimensional discontinuities that do not line up with dimensional axes are currently unsupported. We cannot currently guarantee 
    alpha constraining to avoid crossing discontinuities on functions like 
    `myfunc(x, y) = inv(x - y)`
    In such cases, it's best to use a sequential approach:
    (1) Produce a UvGaussian from an unscented transform on d = (x-y) first (which has an exact solution) and then 
    (2) Create a new UvGaussian on an unscented transform of inv(d)
"""
discontinuities(f::Function) = ()

#Default discontinuities for some functions
discontinuities(f::typeof(log)) = (0,)
discontinuities(f::typeof(sqrt)) = (0,)
discontinuities(f::typeof(inv)) = (0,)
discontinuities(f::typeof(asin)) = (0,1)
discontinuities(f::typeof(acos)) = (0,1)

function select_alpha(f, gs::AbstractGaussian...)
    current = 1.0
    
    limits = discontinuities(f)
    (limits isa Tuple) || error("discontinuities($(f)) must return a tuple, instead it returned $(limits)")

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
        current =  _min_scale_deviation(current, gi, limit[i])
    end

    return current 
end

function _min_scale_deviation(current::Number, gs::Tuple{<:Vararg{<:UvGaussian}}, limits::Tuple{<:Vararg{<:Number}})
    length(gs) == length(limits) || error("Inputs must be same-length Tuples of UvGaussian and Number, recieved $(gs) and $(limits)")

    for i in eachindex(gs)
        current = _min_scale_deviation(current, gs[i], limits[i])
    end

    return current
end

function _min_scale_deviation(current::Number, g::UvGaussian{T}, limit::Number) where T
    z  = abs(convert(T, limit) - g.μ)/g.σ
    zs = z*(1 + inv(z+1))/2
    return ifelse(isfinite(limit), convert(typeof(current), zs), current)
end