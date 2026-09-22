"""
ArgLimits(args...)

A set of limits on an argument to a function. These limits warn the SigmaPoint generator of excessive nonlinearity 
so that the sigma points do not cross this boundary. For exmaple, inv(x) and sqrt(x) both have ill-defined behaviour 
at zero, so both of them have default behaviour of 
```
domainlimits(f::typeof(log), ::Type{<:Number}) = ArgLimits(0.0)
domainlimits(f::typeof(sqrt), ::Type{<:Number}) = ArgLimits(0.0)
```
"""
struct ArgLimits{N,T} 
    list :: NTuple{N,T}
    ArgLimits{N,T}(args...) where {N,T} = new{N,T}(args)
    global tuple2arglims(list::Tuple) = new{length(list), Base.promote_typeof(list...)}(list)
end

function ArgLimits(args...)
    allequal(length, args) || throw(ArgumentError("All limit arguments must have the same length, received $(args)"))
    return tuple2arglims(args)
end

ArgLimits() = ArgLimits{0, Union{}}()

const DomainLimits = Union{ArgLimits, Tuple{Vararg{<:ArgLimits}}}

#=
function ArgLimits(arg1::Tuple, argN::Tuple...)
    args = (arg1, argN...)
    allequal(length, args) || throw(ArgumentError("All limit arguments must have the same length, received $(args)"))
    argtypes = map(typeof, args)

    N = length(args)
    T = reduce(_promote_fieldtypes, argtypes)

    return ArgLimits{N,T}(arg1, argN...)
end

function _promote_fieldtypes(::Type{T1}, ::Type{T2}) where {N, T1<:Tuple{Vararg{<:Any,N}}, T2<:Tuple{Vararg{<:Any,N}}} 
    T = map(promote_type, fieldtypes(T1), fieldtypes(T2))
    return Tuple{T...}
end
=#

import Base.tail
Base.first(limits::ArgLimits) = first(limits.list)
Base.tail(limits::ArgLimits{1}) = first(limits)
Base.tail(limits::ArgLimits) = tuple2arglims(tail(limits.list))


"""
domainlimits(f::Function)

Specifies any domain values in "f" that a distribution should not cross. 
-   This could pertain to an out-of-domain limit like 0 for sqrt(x)
    Well behaved cases must have all positive values, negative values are ill-defined
-   This could also pertain to an asymptote like 0 for inv(x) 
    Well-behaved cases must have the same sign as the mean, zero-crossing behaviour is ill-defined

This function must produce an `ArgLimits` object, where each argument `ArgLimits(args...)` represents a limit. 
For functions with multiple arguments, each value in `args` must be a Tuple. 
The default for functions is `domainlimits(f::Function) = ArgLimits()`. 
If this is incorrect, users will need to define a lower-level dispatch to produce limits for their function.

Examples:

(1) For a single-argument function with a single discontinuitiy at zero (like inv) you would want 

    ```
    domainlimits(f::typeof(inv), ::Type{<:Number}) = ArgLimits(0)
    ```

(2) For a single-argument function with two discontinuties (like asin at 0 and 1), you would want 

    ```
    domainlimits(f::typeof(asin), ::Type{<:Number}) = ArgLimits(0,1)
    ```

(3) For a vector-argument function with a single discontinuity, return a tuple of a vector. If any element 
    of the vector doesn't matter, simply place a non-fininte number in that element

    ```
    vecdiv(x) = x[1]/x[2]
    domainlimits(typeof(vecdiv), ::Type{<:AbstractVector}) = ArgLimits([NaN,0])
    ```

(4) For a two-argument function return a length-2 tuple containing the limits for each argument

    ```
    mydiv(x,y) = x/y
    domainlimits(f::typeof(mydiv), ::Type{<:Numbeer}, ::Type{<:Numbeer}) = (ArgLimits(), ArgLimits(0))
    ```

(5) Vector discontinuities are boxed-form. If more than one discontinuity is needed, NaN elements are ignored

    ```
    myfunc(x,y) = log(x*asin(y))
    domainlimits(f::typeof(myfunc), ::Type{<:AbstractVector}) = ArgLimits([0,0], [NaN,1]) #This is valid
    domainlimits(f::typeof(myfunc), ::Type{<:AbstractVector}) = ArgLimits([0,0], [0,1]) #This expresses the same result
    ```

(6) Multidimensional domain limits that do not line up with dimensional axes are currently unsupported. We cannot currently guarantee 
    alpha constraining to avoid crossing domainlimits on functions like 

    ```
    myfunc(x, y) = inv(x - y)
    ```

    In some cases, a sequential workaround can be achieved:
    (1) Produce a UvGaussian from an unscented transform on `d = (x-y)` first (which has an exact solution) and then 
    (2) Create a new UvGaussian on an unscented transform of `inv(d)` which already has a domain limit of 0
"""
domainlimits(f::Function) = ArgLimits()
domainlimits(f::Function, t1::Type{<:Any}) = domainlimits(f)
domainlimits(f::Function, t1::Type{<:Any}, ts::Type{<:Any}...) = domainlimits(f, t1)

#Default domainlimits for some functions
domainlimits(f::typeof(log), ::Type{<:Number}) = ArgLimits(0.0)
domainlimits(f::typeof(sqrt), ::Type{<:Number}) = ArgLimits(0.0)
domainlimits(f::typeof(inv), ::Type{<:Number}) = ArgLimits(0.0)
domainlimits(f::typeof(asin), ::Type{<:Number}) = ArgLimits(0.0, 1.0)
domainlimits(f::typeof(acos), ::Type{<:Number}) = ArgLimits(0.0, 1.0)


function scale_spread(f, current::Union{SigmaWeights,SigmaParams}, gs::AbstractGaussian...) 
    limits = domainlimits(f, map(meantype, gs)...)
    (limits isa DomainLimits) || error("domainlimits($(f)) must return an `ArgLimits` or a tuple of htem, instead it returned $(limits)")
    return scale_spread(limits, current, gs...)
end

function scale_spread(limits::DomainLimits, current::SigmaWeights, gs::AbstractGaussian...)
    rc = current.rc
    rc2 = _scale_step(rc, limits, gs)
    return scale_spread(rc2/rc, current)
end

function scale_spread(limits::DomainLimits, current::SigmaParams, gs::AbstractGaussian...)
    L  = sum(dimlength, gs)
    rc = current.α*sqrt(L + current.κ)
    rc2 = _scale_step(rc, limits, gs)
    return scale_spread(rc2/rc, current)
end

function scale_spread(ϕ::Number, current::SigmaWeights)
    (ϕ < 1) || return current  #Only proceeed if we know ϕ < 1

    #Create a new SigmaWeights object with the altered step size
    #display("Rescaling α=$(sqrt(current.α²)*ϕ)")
    θ  = SigmaWeights(@set current.rc = ϕ*current.rc)
    ϕ² = abs2(ϕ)
    c  = abs2(θ.rc)

    #Adjust the affected weights for the new step size
    @reset θ.α² = θ.α²*ϕ²
    @reset θ.Wn = 0.5/c
    @reset θ.Wμ = 1 - θ.L/c
    @reset θ.Wσ = θ.Wμ + 1 - θ.α² + θ.β

    return θ
end

function scale_spread(ϕ::Number, current::SigmaParams)
    (ϕ < 1) || return current #Only proceeed if we know ϕ < 1
    #display("Rescaling α=$(current.α*ϕ)")

    θ = @set current.α = current.α*ϕ
    return θ
end

#Dispatch patterns over tuples of AbstractGaussian
function _scale_step(stepscale::Number, lims::Tuple{Vararg{ArgLimits,N1}}, gs::Tuple{Vararg{AbstractGaussian,N2}}) where {N1,N2} 
    N1 == N2 || throw(ArgumentError("Number of limits $(lims) must match the number of arguments $(gs)"))
    newscale = _scale_step(stepscale, first(lims), first(gs))
    return _scale_step(newscale, tail(lims), tail(gs))
end
_scale_step(stepscale::Number, lims::Tuple{<:ArgLimits}, gs::Tuple{<:AbstractGaussian}) = _scale_step(stepscale, first(lims), first(gs))
_scale_step(stepscale::Number, lims::ArgLimits, gs::Tuple{<:AbstractGaussian}) = _scale_step(stepscale, lims, first(gs))


function _scale_step(stepscale::Number, lims::ArgLimits{N}, g::AbstractGaussian) where N
    newscale = _scale_step_arg(stepscale, first(lims), g)
    return _scale_step(newscale, tail(lims), g)
end

_scale_step(stepscale::Number, lims::ArgLimits{1}, g::AbstractGaussian) = _scale_step_arg(stepscale, first(lims), g)
_scale_step(stepscale::Number, lims::ArgLimits{0}, g::AbstractGaussian) = stepscale 

function _scale_step_arg(stepscale::Number, limit::AbstractVector, g::MvGaussian)
    Base.require_one_based_indexing(limit)
    μ = mean(g)

    for i in eachindex(μ)
        gi = UvGaussian(μ[i], maximum(abs, cholrow(g, i)))
        stepscale =  _scale_step_arg(stepscale, limit[i], gi)
    end

    return stepscale 
end

function _scale_step_arg(stepscale::T, limit::Number, g::UvGaussian) where T <: Number
    ϵ = 1e-12 #Ensure step is large enough for sufficient machine precision
    z = abs(limit - g.μ)/g.σ #Limit distance with respect to standard deviations 
    zc = max(ϵ, z*(1 + inv(z+1))/2) #Corrected distance that approaches the limit, but stops short
    newscale = convert(T, zc)

    #This ifelse statement returns old value if newscale is NaN
    return ifelse(newscale < stepscale, newscale, stepscale)
end

#=
function _min_step_scale(stepscale::Number, limits::Tuple{Vararg{ArgLimits{1,<:Number}, N}}, gs::UvGaussian...) where N
    length(gs) == N || error("Inputs must be same-length Tuples of UvGaussian and Number, recieved $(gs) and $(limits)")

    for i in eachindex(gs)
        stepscale = _min_step_scale(stepscale, gs[i], limits[i])
    end

    return stepscale
end
=#

