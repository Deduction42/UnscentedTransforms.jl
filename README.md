[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Coverage Status](https://coveralls.io/repos/github/Deduction42/UnscentedTransforms.jl/badge.svg?branch=main)](https://coveralls.io/github/Deduction42/UnscentedTransforms.jl?branch=main)


# UnscentedTransforms
This is a package designed to function much like [Measurements.jl](https://github.com/juliaphysics/measurements.jl) but instead of using linear error propagation formulas (which are a first order approximation), it uses unscented transforms (which are a second-order approximation). While typically more accurate than error propagation rules, unscented transforms come with the added challenge of dealing with cases where the sigma points cover an invalid domain limit (such as logs needing to be positive) or an asymptote (such as taking an inverse). This package provides tooling to be able to handle these scenarios.

## Basic Usage
This package deals with algebra around univariate and multivariate gaussian distributions (`UvGaussian` and `MvGaussian`). These can be constructed using the `gaussian` function or the `±` (\pm) operator to build a gaussian variable with a mean and a standard deviation. When building multivariate objects, the matrix needs to be in lower square-root form (such as a cholesky decomposition, lower triangular, or diagonal matrix of standard deviations).

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
Using lower forms means that the columns represent coefficients of standard normal random numbers that can be added to the mean. It is the most natural analog to expressing error on a vector. It is also computationally coefficient, because sigma points require the lower square root form to construct sigma points. Linear operations do not construct sigma points but instead use linear shortcut rules

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

Unfortunately, UnscentedTransforms does not parse out expressions and identify identities naturally like Measurements.jl. For complicated expressions where a variable is used multiple times, you will need to build a higher-level function and apply unscented transforms on it. This is done by calling `gaussian(f::Function, p::SigmaParams, args::UvGaussian...)` as shown below
```julia
julia> sin(xuv)/cos(xuv) - tan(xuv) # This assumes all instances of xuv are independent
0.03834714643546766 ± 2.024728880001174

julia> gaussian(x->sin(x)/cos(x) - tan(x), SigmaParams(), xuv) # This explicitly treats values as identical
0.0 ± 0.0
```
This can also be applied to multivariate functions with partially correlated arguments
```julia
julia> gaussian(x->([0.1 0.2; 0.2 -0.1]*x).^2, SigmaParams(), xmv)
2-element Vector{Float64}:
 0.0224
 0.0381
± 2×2 LowerTriangular{Float64, Matrix{Float64}}:
 0.023779    ⋅ 
 0.0442979  0.0258173

julia> gaussian(x-> x-x, SigmaParams(), xmv) # Values cancel each other out
2-element Vector{Float64}:
 0.0
 0.0
± 2×2 LowerTriangular{Float64, Matrix{Float64}}:
 0.0   ⋅ 
 0.0  0.0
 
julia> gaussian(x-> x'*x, SigmaParams(), xmv) # Function produces a scalar
1.21 ± 1.456159331941392 
```

## Kalman Filtering
Since the most common use of unscented transforms is the Unscented Kalman Filter (UKF), this package also provides UKF capabilities while being able to use classical linear fallbacks if the transition or observation functions happen to be linear. Robustness is a key objective of this package, so the ability to define boxed state constraints and penalize outlying observations are included features that differentiate this package from many other Kalman filtering packages.

```julia
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

# Diagonal matrices for independent model and measurement noidses
qd = Diagonal([0.003, 0.00003, 0.1])
rd = Diagonal([0.2, 0.1])

# Build a nonlinear predictor and a linear observer, using ±m (unary operator) will produce a zero-mean noise term
predictor = NonlinearPredictor(f = predictor_func, ε = ±qd, θ = SigmaParams(ϵ=0.01)) #Nonlinear predictors can use keyword arguments
observer = LinearPredictor(SA[1 0 0 ; 0 1 0], SMatrix{2,0,Float64}(), ±rd) #Linear predictors are purely positional based on 'A'x + 'B'u

# Initialize the state
state = SA[0, 0, log(0.1)] ± (2*qd)

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

## Parameterization

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
2. *Moment Capture:* The original paper recommends setting `κ=3-n` where `n` is the dimension in order to capture higher-order moments. However, this package (and the Unscented Kalman Filter) converts teh resulting sigma points to a Gaussian distribution, which discards higher-order moments.
3. *Domain Enforcement:* Papers using the three-parameter method often recommend setting `α` to a small number `0.0001 < α < 1` in order to avoid capturing "excess nonlinearity" which can happen when sigma points cross a limit or an asymptote. However, setting `α` to a value much smaller than 1 can cause roundoff errors and will make the UKF behave more like an EKF. This package allows users to set domain limits on sigma points to reduce scaling if the mean is near a box-constrained limit (for example, forcing certain states to be positive or to prevent crossing asymptotes) eliminating the need to set `α < 1`.

### Minimum Sigma Point Spread
As mentioned elsewhere, this package contains tooling to reduce the spread of sigma points if the mean is close to a box-constrained limit. I such cases, it's useful to define a minimum spread ϵ (\epsilon) which functions as a minimum α value for a given state index. This value is needed to ensure we don't run into precision errors. It is typically set to 1e-6 which works well for univariate cases but may need to be set to a larger number (such as ϵ=0.01) when dealing with higher dimensional multivariate states.
```julia
SigmaParams(α=1, ϵ=1e-6)
```
