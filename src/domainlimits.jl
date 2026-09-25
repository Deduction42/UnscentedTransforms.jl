#================================================================================================================================
Implements a sigma point scaling algorithm that keeps all the points within a boxed set of constraints
This is commonly used to force positive values on states/parameters that aren't naturally constrained by the predictor functions 
This algorithm ATTEMPTS to keep points inside the constraints, but numerical precision/stability issues may prevent this
if the mean is very close to a constraint. The underlying function should enforce contstraints. This may introduce some distribution
error, but it will be far less than the error introduced by not scaling the points.

The algorithm works as follows:
For each off-center sigma weight w[1:end]
    1.  Translate the weight into a point-distance scalar δ (where δ = sqrt(0.5/w[i]))
    2.  Find the minimum distance Δu between μ[i] and lim[i] for all boundaries in lim
    3.  Calcualte vδ[i] = gamma_scale(Δu[i]/σ[i])
        -   gamma_scale(z) converts the quantile of a standard normal distribution to an equivalent gamma quantile that fits a boundary
        -   gamma_scale(z) ≈ z*(1 + inv(z+1))/2 as a current approximation (better ones probably exist, use Symbolic Regression in the future)
    4.  Return δnew = max(minimum(vδ, init=δ), ϵ)
        -   minimum should use a function "f(i)" to generate vδ for each index i
    5.  Convert δnew to wnew[i], wnew[i+N] = 0.5/abs2(δ)
    6.  Calcualte a new w0 = 1 - sum(wnew)

================================================================================================================================#

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


function scale_weights(f, current::Union{SigmaParams, SigmaWeights}, args::AbstractGaussian...) 
    limits = domainlimits(f, map(meantype, args)...)
    (limits isa DomainLimits) || error("domainlimits($(f)) must return an `ArgLimits` or a tuple of them, instead it returned $(limits)")
    return scale_weights(current, limits, args...)
end

scale_weights(current::SigmaParams, arglims::DomainLimits, args::AbstractGaussian...) = scale_weights(SigmaWeights(dimlength(args), current), arglims, args...)
scale_weights(current::SigmaWeights, arglims::ArgLimits{0}, args::AbstractGaussian...) = current #No-op shortcut
scale_weights(current::SigmaWeights, arglims::DomainLimits, args::AbstractGaussian...) = _scaled_weights(current, arglims, args...)

#Used to find the sigma weights for a correlated multivariate distribution 
function _scaled_weights(w::SigmaWeights{<:ConstVec}, arglims::ArgLimits{<:Any,<:AbstractVector}, arg::MvGaussian)
    #Closure to wrap a single index
    _scale_weight_closure(i::Integer) = _scale_weight(w.wi.all, arglims, mean(arg), cholcol(arg, i))

    vw = map(_scale_weight_closure, similar_indices(mean(arg)))
    w0 = 1 - 2*sum(vw)
    return SigmaWeights(w.N, w0, [vw; vw])
end

#Used to find the sigma weights for a set of uncorrelated variables
function _scaled_weights(w::SigmaWeights{<:ConstVec}, arglims::Tuple{Vararg{ArgLimits{<:Any,<:Number}}}, args::UvGaussian...)
    #Closure to calculate a new weight from the scaled spread
    _scale_weight_closure(xlims, x) =_scale_weight(w.wi.all, xlims, x)

    length(arglims) == length(args) || throw(DimensionMismatch("Arguments must have the same length (recieved $(arglims), $(args))"))
    vw = SVector(map(_scale_weight_closure, arglims, args))
    w0 = 1 - 2*sum(vw)
    return SigmaWeights(w.N, w0, [vw; vw])
end

#Used to find the sigma weights for a single variable 
function _scaled_weights(w::SigmaWeights{<:ConstVec}, arglims::ArgLimits, arg::UvGaussian)
    wi = _scale_weight(w.wi.all, arglims, arg)
    w0 = 1 - 2*wi 
    return SigmaWeights(w.N, w0, SVector(wi, wi))
end

#Used to convert spread to weights 
function _scale_weight(w::T, args...) where T<:Number 
    δ = sqrt(0.5/w)
    return convert(T, 0.5/abs2(_scale_spread(δ, args...)))
end

#Used to find the spread factor δ for a single sigma point
function _scale_spread(δ::T, arglims::ArgLimits{N,<:AbstractVector}, μ::AbstractVector, σ::AbstractVector) where {N, T<:Number}
    for i in eachindex(μ)
        ithlims = ith_arglims(arglims, i)
        δ = convert(T, _scale_spread(δ, ithlims, UvGaussian(μ[i], σ[i])))
    end 
    return δ
end

#Used to find the spread factor δ for a single element of a sigma point, returns a minimum
function _scale_spread(δ::T, arglims::ArgLimits{N,<:Number}, g::UvGaussian) where {N, T<:Number}
    ϵ = 1e-6

    iszero(g.σ) && return max(ϵ, δ) #Return old value if standard deviation is zero

    Δμ = _bound_distmin_gaussian(arglims, g)
    return max(ϵ, _bound_scale(δ, Δμ))
end

#Finds a gausian distribution over the minimum distance from a set of limits
function _bound_distmin_gaussian(arglims::ArgLimits{N,<:Number}, xi::UvGaussian) where N
    Δmin = minimum(x->abs(x-xi.μ), arglims.list)
    return UvGaussian(Δmin, xi.σ)
end

#Scales a spread factor δ based on the previous value and the distribution over the distances
function _bound_scale(δ::T, Δμ::UvGaussian) where T <: Number
    δnew = convert(T, gamma_scale(abs(Δμ.μ)/Δμ.σ))

    #Strict minimum return (returns old value if newscale is NaN)
    return ifelse(δnew < δ, δnew, δ)
end

gamma_scale(z::Number) = z*(1 + inv(z+1))/2

function ith_arglims(arglims::ArgLimits{N,<:Union{AbstractVector, Tuple}}, ind::Int) where N
    return tuple2arglims(map(x->x[ind], arglims.list))
end

similar_indices(v::AbstractVector) = eachindex(v)
similar_indices(v::StaticVector{N}) where N = SVector{N}(eachindex(v))

#=
function scale_spread(limits::DomainLimits, current::SigmaWeights, gs::AbstractGaussian...)
    δ  = sqrt(0.5/w[1])
    δ2 = _scale_step(δ, limits, gs)
    return scale_spread(δ2/δ, current)
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
=#

#=
function _min_step_scale(stepscale::Number, limits::Tuple{Vararg{ArgLimits{1,<:Number}, N}}, gs::UvGaussian...) where N
    length(gs) == N || error("Inputs must be same-length Tuples of UvGaussian and Number, recieved $(gs) and $(limits)")

    for i in eachindex(gs)
        stepscale = _min_step_scale(stepscale, gs[i], limits[i])
    end

    return stepscale
end
=#

