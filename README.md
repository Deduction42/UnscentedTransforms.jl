[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Coverage Status](https://coveralls.io/repos/github/Deduction42/UnscentedTransforms.jl/badge.svg?branch=main)](https://coveralls.io/github/Deduction42/UnscentedTransforms.jl?branch=main)

# UnscentedTransforms
This is a package designed to function much like [Measurements.jl](https://github.com/juliaphysics/measurements.jl) but instead of using linear error propagation formulas (which are a first order approximation), it uses unscented transforms (which are a second-order approximation). While typically more accurate than linear error propagation rules, unscented transforms come with the added challenge of dealing with cases where the sigma points cover an invalid domain limit (such as logs needing to be positive) or an asymptote (such as taking an inverse). This package provides (currently experimental) tooling to be able to handle these scenarios.

## Basic Usage
This package deals with algebra around univariate and multivariate gaussian distributions (`UvGaussian` and `MvGaussian`). These can be constructed using the `gaussian` function or the `±` (\pm) operator to build a gaussian variable with a mean and a standard deviation. When building multivariate objects, the matrix needs to be in lower square-root form (such as a cholesky decomposition, lower triangular, or diagonal matrix of standard deviations); the lower form is the most natural multivariate analog to expressing correlated error terms and is used for constructing sigma points.

```julia
using UnscentedTransforms
using LinearAlgebra

julia> xuv = 5 ± 0.1
5.0 ± 0.1

julia> xuv = gaussian(5, 0.1)
5.0 ± 0.1

julia> xmv = [0, 0.1] ± cholesky([1 0.1; 0.1 0.2])
2-element Vector{Float64}:
 0.0
 0.1
± 2×2 LowerTriangular{Float64, Matrix{Float64}}:
 1.0   ⋅ 
 0.1  0.43589
```
Linear operations do not construct sigma points but instead use linear shortcut rules.

```julia 
julia> xuv*2 + 3
13.0 ± 0.2

julia> [1 2; 0.1 0.2]*xmv + [0,1]
2-element Vector{Float64}:
 0.2
 1.02
± 2×2 LowerTriangular{Float64, Matrix{Float64}}:
 1.48324    ⋅ 
 0.148324  0.0
```

Unfortunately, UnscentedTransforms does not parse out expressions and identify identities naturally like Measurements.jl. For complicated expressions where a variable is used multiple times, you will need to build a higher-level function and apply unscented transforms on it. Reserving unscented transforms for high-level functions is also computationally more efficient as it reduces the number of conversions between sigma points and Gaussian distributions. This is done by calling `gaussian(f::Function, p::SigmaParams, args::UvGaussian...)` as shown below
```julia
# This assumes all instances of xuv are independent, and reconstructs Gaussians three times
julia> sin(xuv)/cos(xuv) - tan(xuv) 
-3.1086244689504383e-15 ± 2.0177204810295253

# This explicitly treats values as identical and reconstructs the Gaussian only once
julia> gaussian(x->sin(x)/cos(x) - tan(x), SigmaParams(), xuv) 
0.0 ± 0.0
```
This can also be applied to multivariate functions with partially correlated arguments
```julia
# Generic nonlinear function
julia> gaussian(x->([0.1 0.2; 0.2 -0.1]*x).^2, SigmaParams(), xmv)
2-element Vector{Float64}:
 0.0004
 0.0001
± 2×2 LowerTriangular{Float64, Matrix{Float64}}:
 0.023779    ⋅ 
 0.0442979  0.0258173

# A function that should always produce zero
julia> gaussian(x-> x-x, SigmaParams(), xmv)
2-element Vector{Float64}:
 0.0
 0.0
± 2×2 LowerTriangular{Float64, Matrix{Float64}}:
 0.0   ⋅ 
 0.0  0.0
 
# A multivariate function that produces a scalar
julia> gaussian(x-> x'*x, SigmaParams(), xmv) 
0.01 ± 1.456159331941392
```

## Kalman Filtering
Since the most common use of unscented transforms is the Unscented Kalman Filter (UKF), this package also provides UKF capabilities while being able to use classical linear fallbacks if the transition or observation functions happen to be linear. Robustness is a key objective of this package, so the ability to define boxed state constraints and penalize outlying observations are included features that differentiate this package from many other Kalman filtering packages.

```julia
using UnscentedTransforms
using LinearAlgebra
using StaticArrays
const Δt = 1.0

function predictor_func(X, u) # Function needs an input "u" argument even if it's not used
    k = exp(X[3])
    A = @SMatrix [
        0  -k   0;
        1   0   0;
        0   0   0 
    ]
    return exp(A*Δt)*X # Discretize A and then predict result
end

# Use the ± (unary operator) to build zero-mean noise terms for the Kalman filter
εx = ± Diagonal(SA[0.003, 0.00003, 0.1])
εy = ± Diagonal(SA[0.2, 0.1])

# Build a nonlinear predictor and a linear observer, using ±m (unary operator) will produce a zero-mean noise term

#Nonlinear predictors can use keyword arguments, UKF should not use the projected mean
predictor = NonlinearPredictor(f = predictor_func, ε = εx, θ = SigmaParams(proj=false)) 
#Linear predictors are purely positional based on 'A'x + 'B'u
observer = LinearPredictor(SA[1 0 0 ; 0 1 0], SMatrix{2,0,Float64}(), εy) 

# Initialize the state
state = SA[0, 0, log(0.1)] + (2*εx)

# Build the state space model
model = StateSpaceModel(
    state = state,
    predictor = predictor,
    observer  = observer,
    outlier  = 2.0 #Penalize observations more than 2 standard deviations from expected value
)

# Nonlinear predict step of a Kalman filter with u = nothing (unused), returns predicted state
predict!(model, nothing)

# Linear update step of the Kalman filter with u = []
info = update!(model,[0,0], Float64[])

julia> info.X # Final state after prediction
3-element SVector{3, Float64} with indices SOneTo(3):
 -8.977526205148166e-20
 -1.0121600199140066e-19
 -2.3025850929940455
± 3×3 LowerTriangular{Float64, Matrix{Float64}}:
 0.00643139   ⋅           ⋅ 
 0.00520817  0.00274702   ⋅ 
 0.0         0.0         0.223607

 julia> info.Y # Predicted value of Y before the update step (useful for tracking prediction error)
2-element SVector{2, Float64} with indices SOneTo(2):
 -9.020900888258299e-20
 -1.0164395367051604e-19
± 2×2 LowerTriangular{Float64, Matrix{Float64}}:
 0.200104      ⋅ 
 0.000168149  0.100174
```

## Sigma Point Parameterization

### Sigma Point Spread
This package uses the single-value parameterization approach from the original paper focused on the `κ` (\kappa) value. This can be set as follows:
```julia
SigmaParams(κ=0)
```
However, because the three-parameter approach is fairly common (such as the approached used by MATLAB) we can set an equivalent value of `α` (\alpha, being the relative spread of the points) which will calculate the corresponding value of κ for you. 
```julia
SigmaParams(α=1)
```
The default value of `α=1` which maps to a value of `κ=0`. While there are different recommendations on setting these values for varying reasons, we recommend these default values `α=1.0` or `κ=0.0` due to the following reasons: 
1. *Representation:* Setting `α=1` or `κ=0` defines an ellipse that covers a reasonable proportion of the data (68.27% of the data for a 1-dimensional Gaussian, approaching 50% for higher-dimensional Gaussians). Most of the data in this distribution is going to be this far from the mean.
2. *Avoiding Error Amplification:* Setting `α` so a very small value (such as `α<0.01`) can prevent the sigma points from hitting a point of excessive nonlinearity (which can happen if arguments are clamped). However, if a sigma point does hit a strong nonlinearity, these effects can be *amplified* due to large weights associated with small `α`.

### Weighted versus Projected Mean 
The traditional unscented transform calculates a weighted average for the mean. This can be set by using
```julia
SigmaParams(proj=false)
```
This package uses `SigmaParams(proj=true)` by default mostly because it produces intuitive mean results similar to Measurements.jl and it is less prone to *bouncing* 
```julia
julia> x = 0.0 ± 3.0
0.0 ± 3.0

# Most people expect the mean to pass through exp directly
julia> gaussian(exp, SigmaParams(), x) 
1.0 ± 13.51222824447916

# Traditional unscented transforms use weighted means that can shift it significantly
julia> gaussian(exp, SigmaParams(proj=false), x)
10.067661995777765 ± 13.51222824447916
```
Note that a monte carlo simulation would produce results close to `56.7 ± 479.0` (highest accuracy) while Measurements.jl would produce `1.0 ± 3.0`. While both UnscentedTransforms.jl and Measurements.jl underestimate the mean and the variance, the unscented transform with SigmaParams(proj=false) produced the closest result to the Monte Carlo simulation. **In geneneral, setting `SigmaParams(proj=false)` is generally more accurate, and is strongly recommended if you're using a Kalman filter, as it is more robust against divergence**. 

### The Bounce Effect (Clamped Parameter Estimation)
The UKF can struggle when sigma points enter a region of high nonlinearity (i.e. too nonlinear to be expressed by a 3rd order Taylor series). Unfortunately, this can easily happen during parameter estimation if one tries to constrain arguments by clamping. If a sigma point gets clamped, it produces a *bounce effect* on the parameter estimates, where a parameter drifts toward a constraint, but dramatically *bounces away* once a related sigma point gets clamped. In the example below, we are simply trying to constrain `x` to be positive.
```julia
positive(x::Real) = max(0,x)

# Produces a reasonable result
julia> x⁺ = gaussian(positive, SigmaParams(α=1.0, proj=false), 0.001 ± 0.001)
0.001 ± 0.001

# Growing the variance causes sigma points to hit a strong nonlinearity, causing the mean to bounce mildly away from the constraint
x⁺ = gaussian(positive, SigmaParams(α=1.0, proj=false), 0.001 ± 1.0)
0.5005 ± 0.7071071347398497

# Using a smaller value of α can prevent bouncing by keeping sigma points inside a well-defined region
x⁺ = gaussian(positive, SigmaParams(α=0.001, proj=false), 0.001 ± 1.0)
0.0009999999999763531 ± 1.0

# If variance grows such that points stray beyond that region, tighly clumped sigma points can AMPLIFY the bounce effect
# This means that small values of α can be risky if any states need clamping
x⁺ = gaussian(positive, SigmaParams(α=0.001, proj=false), 0.001 ± 2.0)
500.001 ± 1.5811388300841898

# Using projected means can mitigate this effect on the mean which is why it's default 
x⁺ = gaussian(positive, SigmaParams(α=0.001, proj=true), 0.001 ± 2.0)
0.001 ± 1.5811388300841898

x⁺ = gaussian(positive, SigmaParams(α=1, proj=true), 0.001 ± 2.0)
0.001 ± 1.4142137391497793
```

## Sigma Point Scaling
This package also supports automated scaling of sigma points which can be useful in two scenarios: 
1. Avoiding the *bounace effect* caused by sigma points being clamped. 
2. Preventing a sigma point from *jumping over an asymptote* if the function is not constrained

Scaling sigma points requires two actions to be taken:
1. Extending the experimental `UnscentedTransforms.domainlimits(f::typeof(<myfunc>), ::Type{<argument types>}...)` function to the appropriate argument limits `(ArgLimits(<first arg limits>...), ArgLimits(<second arg limits>...))`
2. Setting the `ϵ` (\epsilon) parameter in the `SigmaParams` that can act as a minimum effective value of `α`. 

Calling `gaussian(<myfunc>, SigmaParams(ϵ=1e-6), args::AbstractGaussian...)` will look up the domain limits of that function and apply them if any appear (by default functions return no limits `ArgLimits()`). The value of `ϵ` will set the minimum amount of spread; small values of `ϵ` will better prevent constraint violations but can risk roundoff error because points are too close together; this is especially true in multivariate scenarios. In particular, care must be taken agaisnt small values of `ϵ` when clamping parameters as it can amplify bouncing.
```julia
SigmaParams(α=1, ϵ=1e-6, proj=true) #Small values work for univariates
SigmaParams(α=1, ϵ=0.3, proj=false) #Don't use very small values of ϵ if proj=false and multivariate
```

### Constraining Sigma Points
Sigma points can be constrained by taking these two steps:

1. Clamping sigma points to be inside the feasible region at the beginning of the function. This is a simple solution, but can be prone to *bouncing*. 
2. Constraining sigma points by extending `UnscentedTransforms.domainlimits` to your function (experimental feature)
3. Setting an apprpriate `ϵ=0.3` parameter (either in your `gaussian` call or `NonlinearPredictor` arguments in the UKF)

#### Single-Argument Domain Limits
Constraining sigma points can be done by extending the `domainlimits` function. Here is an example of how to do this for a single-argument function:
```julia
# Define a clamped version of the function
positive(x::Real) = max(0,x)

# Observe the bounce effect that drives the mean above the projected result 0.03162
julia> gaussian(positive, SigmaParams(proj=false), 0.1 ± 0.2)
0.5002499375312305 ± 0.6854643797899985

# Define a "domainlimits" function for it (experimental)
UnscentedTransforms.domainlimits(g::typeof(positive), ::Type{<:Real}) = ArgLimits(0)

# Mean is no longer bounced away
julia> gaussian(positive, SigmaParams(ϵ=0.1, proj=false), 0.1 ± 0.2)
0.10000000000000003 ± 0.20000000000000004
```

#### Multiple-Argument Domain Limits
The following is an example of defining constraints for multiple arguments
```julia
# Note that this function isn't clamped
mydiv(x::Real, y::Real) = x/y

# Sigma point jumps asymptote, giving artificially low variance
julia> gaussian(mydiv, SigmaParams(proj=false), 1.0 ± 1.0, 0.5 ± 1.0)
0.8571428571428574 ± 2.6341555592265387

# Define a "domainlimits" function for it, for multiple arguments, use a Tuple with an ArgLimits for each argument
UnscentedTransforms.domainlimits(g::typeof(mydiv), ::Type{<:Real}, ::Type{<:Real}) = (ArgLimits(NaN), ArgLimits(0))

# Better capture of nonlinear behaviour
julia> gaussian(mydiv, SigmaParams(ϵ=0, proj=false), 1.0 ± 1.0, 0.5 ± 1.0)
28.181818181818176 ± 17.157510463037926
```

#### Multivariate Domain Limits
The following is an example on defining constraints on a vector (multivariate) argument
```julia
using StaticArrays 

# Define a clamped version of the function
my1div23(x::SVector{3}) = x[1]/(x[2]*x[3])

# Original result with asymptote jumping and reduced mean/variance
julia> gaussian(my1div23, SigmaParams(proj=false), SA[1.0,0.5,0.5] ± Diagonal(SA[1,1,1]))
1.0909090909090908 ± 5.454545454545454

# Define the domain limits as a single vector
UnscentedTransforms.domainlimits(f::typeof(my1div23), ::Type{<:SVector{3}}) = ArgLimits(SVector(NaN, 0, 0))

# More variance is captured
julia> gaussian(my1div23, SigmaParams(proj=false, ϵ=0), SA[1.0,0.5,0.5] ± Diagonal(SA[1,1,1]))
108.72727272727269 ± 48.363636363636346
```







