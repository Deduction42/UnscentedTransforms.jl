[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Coverage Status](https://coveralls.io/repos/github/Deduction42/UnscentedTransforms.jl/badge.svg?branch=main)](https://coveralls.io/github/Deduction42/UnscentedTransforms.jl?branch=main)


# UnscentedTransforms
This is a package designed to function much like [Measurements.jl](https://github.com/juliaphysics/measurements.jl) but instead of using linear error propagation formulas (which are a first order approximation), it uses unscented transforms (which are a second-order approximation). While typically more accurate than error propagation rules, unscented transforms come with the added challenge of dealing with unusual behaviour if the mean is close (with respect to the standard deviation) to an invalid domain limit (such as logs needing to be positive) or an asymptote (such as taking an inverse). This package provides tooling to be able to handle these scenarios.

Propagation error rules assume that inputs are either uncorrelated or identical. UnscentedTransforms.jl however, allows using correlated multivariate inputs, where correlations are defined using a covariance matrix. Due to computational accuracy and the need to avoid recomputing matrix square-roots, the square-root (cholesky) form is used to express the covariance. If a diagonal matrix is used, it is assumed to already be in square-root form (i.e. a diagonal matrix of standard deviations, not variances).

Since the most common use of unscented transforms is the Unscented Kalman Filter (UKF), this package also provides UKF capabilities while being able to use classical linear fallbacks if the transition or observation functions happen to be linear. Robustness is a key objective of this package, so the ability to define boxed state constraints and penalize outlying observations are included features that differentiate this package from many other Kalman filtering packages.


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
