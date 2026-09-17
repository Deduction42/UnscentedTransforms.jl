#======================================================================================================================================
Math functions
======================================================================================================================================#
Base.:+(x::Number, g::UvGaussian) = UvGaussian(x + mean(g), std(g))
Base.:+(g::UvGaussian, x::Number) = UvGaussian(x + mean(g), std(g))
Base.:+(g1::UvGaussian, g2::UvGaussian) = UvGaussian(mean(g1) + mean(g2), add_cov(std(g1), std(g2)))
Base.:+(g1::MvGaussian, g2::MvGaussian) = MvGaussian(mean(g1) + mean(g2), add_cov(std(g1), std(g2)))

function Base.:+(X::SigmaPoints, g::MvGaussian) 
    μx = mean(X)
    return MvGaussian(mean(g) + μx, add_cov(std(g), X, μx))
end
Base.:+(g::MvGaussian, X::SigmaPoints) = g + X

Base.:-(v::ZeroVec) = v
Base.:-(g::UvGaussian) = UvGaussian(-mean(g), std(g))
Base.:-(g::MvGaussian) = MvGaussian(-mean(g), std(g))
Base.:-(x::Number, g::UvGaussian) = UvGaussian(x - mean(g), std(g))
Base.:-(g::UvGaussian, x::Number) = UvGaussian(x - mean(g), std(g))
Base.:-(g1::UvGaussian, g2::UvGaussian) = UvGaussian(mean(g1) - mean(g2), add_cov(std(g1), std(g2)))
Base.:-(g1::MvGaussian, g2::MvGaussian) = MvGaussian(mean(g1) - mean(g2), add_cov(std(g1), std(g2)))

function Base.:-(X::SigmaPoints, g::MvGaussian) 
    μx = mean(X)
    return MvGaussian(mean(g) - μx, add_cov(g.Σ, X, μx))
end
Base.:-(g::MvGaussian, X::SigmaPoints) = g - X

Base.:*(x::Number, g::UvGaussian) = UvGaussian(x*mean(g), x*std(g))
Base.:*(g::UvGaussian, x::Number) = UvGaussian(x*mean(g), x*std(g))
Base.:*(x::Number, g::MvGaussian{<:Cholesky}) = MvGaussian(x*mean(g), Cholesky(x*std(g).U))
Base.:*(g::MvGaussian{<:Cholesky}, x::Number) = MvGaussian(x*mean(g), Cholesky(x*std(g).U))
Base.:*(x::Number, g::MvGaussian{<:Diagonal}) = MvGaussian(x*mean(g), x*std(g))
Base.:*(g::MvGaussian{<:Diagonal}, x::Number) = MvGaussian(x*mean(g), x*std(g))

predict(f, θ::SigmaParams, args::UvGaussian...) = predict(f, SigmaWeights(length(args), θ), args...)

function predict(f, θ::SigmaWeights, args::UvGaussian...)
    Np = 2*length(args) + 1
    θ  = scale_step(f, θ, args...)

    old_points = SigmaPoints(args, θ)
    indvec = SVector{Np}(firstindex(old_points):lastindex(old_points))
    new_points = map(ind->f(old_points[ind]...), indvec) #Non-allocating result

    return gaussian(new_points)
end

"""
domainlimits(f::Function)

Specifies any domain values in "f" that a distribution should not cross. 
-   This could pertain to an out-of-domain limit like 0 for sqrt(x)
    Well behaved cases must have all positive values, negative values are ill-defined
-   This could also pertain to an asymptote like 0 for inv(x) 
    Well-behaved cases must have the same sign as the mean, zero-crossing behaviour is ill-defined

This function must produce a tuple of domainlimits. The default value is an empty tuple (no limits).
If this is incorrect, users will need to define a lower-level dispatch to produce limits for their function.
    `domainlimits(f::Function) = ()`

Examples:

(1) For a single-argument function with a single discontinuitiy at zero (like inv) you would want 

    ```
    domainlimits(f::typeof(inv)) = (0,)
    ```

(2) For a single-argument function with two discontinuties (like asin at 0 and 1), you would want 

    ```
    domainlimits(f::typeof(asin)) = (0,1)
    ```

(3) For a vector-argument function with a single discontinuity, return a tuple of a vector. If any element 
    of the vector doesn't matter, simply place a non-fininte number in that element

    ```
    vecdiv(x) = x[1]/x[2]
    domainlimits(typeof(vecdiv)) = ([NaN,0], )
    ```

(4) For a two-argument function with a single disonctinuity, use a non-finite number for any argument that doesn't matter.

    ```
    mydiv(x,y) = x/y
    domainlimits(f::typeof(mydiv)) = ((NaN, 0),)
    ```

(5) Currently, only "box domainlimits" are supported, which means that you can lump multiple domainlimits in a single object 

    ```
    invmul(x,y) = inv(x*y)`
    domainlimits(f::typeof(invmul)) = ((0,0),)
    ```

(6) Box discontinuitieis also means can also repeat domainlimits or place NaN if that argument's discontinuity is already taken care of

    ```
    myfunc(x,y) = log(x*asin(y))
    domainlimits(f::typeof(myfunc)) = ((0,0), (NaN,1)) #This is valid
    domainlimits(f::typeof(myfunc)) = ((0,0), (0,1)) #This expresses the same thing
    ```

(7) Multidimensional domainlimits that do not line up with dimensional axes are currently unsupported. We cannot currently guarantee 
    alpha constraining to avoid crossing domainlimits on functions like 

    ```
    myfunc(x, y) = inv(x - y)
    ```

    In such cases, it's best to use a sequential approach:
    (1) Produce a UvGaussian from an unscented transform on `d = (x-y)` first (which has an exact solution) and then 
    (2) Create a new UvGaussian on an unscented transform of `inv(d)`
"""
domainlimits(f::Function) = ()

#Default domainlimits for some functions
domainlimits(f::typeof(log)) = (0,)
domainlimits(f::typeof(sqrt)) = (0,)
domainlimits(f::typeof(inv)) = (0,)
domainlimits(f::typeof(asin)) = (0,1)
domainlimits(f::typeof(acos)) = (0,1)


function scale_step(f, current::SigmaWeights, gs::AbstractGaussian...)
    ϵ = 4*sqrt(eps(typeof(current.rc))) #Step size large enough to guarantee overcoming machine precision

    #Check the step scale calculate a shrinking factor ϕ that is large enough to overcome machine precission error
    rc2 = _scale_step(f, current.rc, gs...)
    ϕ = max(rc2/current.rc, ϵ)

    #Return the current SigmaWeights if the step size doesn't need shrinking
    (ϕ >= 1) && return current 
    
    #Create a new SigmaWeights object with the altered step size
    ϕ² = abs2(ϕ)
    θ  = SigmaWeights(@set current.rc = ϕ*current.rc)
    c  = abs2(θ.rc)

    #Adjust the affected weights for the new step size
    @reset θ.α² = θ.α²*ϕ²
    @reset θ.Wn = 0.5/c
    @reset θ.Wμ = 1 - θ.L/c
    @reset θ.Wσ = θ.Wμ + 1 - θ.α² + θ.β

    return θ
end

function _scale_step(f, stepscale::Number, gs::AbstractGaussian...)    
    limits = domainlimits(f)

    #Ensure limits are a Tuple and return stepscale if there are no limits
    (limits isa Tuple) || error("domainlimits($(f)) must return a tuple, instead it returned $(limits)")
    isempty(limits) && return stepscale 

    for lim in limits
        stepscale = _min_step_scale(stepscale, g, lim)
    end

    return stepscale
end

function _min_step_scale(stepscale::Number, g::MvGaussian, limit::AbstractVector)
    Base.require_one_based_indexing(limit)
    μ = mean(g)

    for i in eachindex(μ)
        gi = UvGaussian(μ[i] + maximum(abs, cholrow(g, i)))
        stepscale =  _min_step_scale(stepscale, gi, limit[i])
    end

    return stepscale 
end

function _min_step_scale(stepscale::Number, gs::Tuple{Vararg{<:UvGaussian}}, limits::Tuple{Vararg{<:Number}})
    length(gs) == length(limits) || error("Inputs must be same-length Tuples of UvGaussian and Number, recieved $(gs) and $(limits)")

    for i in eachindex(gs)
        stepscale = _min_step_scale(stepscale, gs[i], limits[i])
    end

    return stepscale
end

function _min_step_scale(oldscale::Number, g::UvGaussian{T}, limit::Number) where T
    z = abs(convert(T, limit) - g.μ)/g.σ
    newscale = convert(typeof(oldscale), z*(1 + inv(z+1))/2)

    #This ifelse statement returns old value if newscale is NaN
    return ifelse(newscale < oldscale, newscale, oldscale)
end