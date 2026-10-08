#======================================================================================================================================
Math functions
======================================================================================================================================#
Base.:+(x::Number, g::UvGaussian) = UvGaussian(x + mean(g), std(g))
Base.:+(g::UvGaussian, x::Number) = UvGaussian(x + mean(g), std(g))

Base.:+(g1::UvGaussian, g2::UvGaussian) = combine(+, g1, g2)
Base.:+(g1::MvGaussian, g2::MvGaussian) = combine(+, g1, g2)
Base.:+(X::SigmaPoints, g::MvGaussian) = combine(+, g, X)
Base.:+(g::MvGaussian, X::SigmaPoints) = combine(+, g, X)

Base.:-(v::ZeroVec) = v
Base.:-(g::UvGaussian) = UvGaussian(-mean(g), std(g))
Base.:-(g::MvGaussian) = MvGaussian(-mean(g), std(g))
Base.:-(x::Number, g::UvGaussian) = UvGaussian(x - mean(g), std(g))
Base.:-(g::UvGaussian, x::Number) = UvGaussian(x - mean(g), std(g))

Base.:-(g1::UvGaussian, g2::UvGaussian) = combine(-, g1, g2)
Base.:-(g1::MvGaussian, g2::MvGaussian) = combine(-, g1, g2)
Base.:-(X::SigmaPoints, g::MvGaussian) = combine(-, g, X)
Base.:-(g::MvGaussian, X::SigmaPoints) = combine(-, g, X)


Base.:*(x::Number, g::UvGaussian) = UvGaussian(x*mean(g), x*std(g))
Base.:*(g::UvGaussian, x::Number) = UvGaussian(x*mean(g), x*std(g))
Base.:*(x::Number, g::MvGaussian{<:Cholesky}) = MvGaussian(x*mean(g), Cholesky(x*std(g).L))
Base.:*(g::MvGaussian{<:Cholesky}, x::Number) = MvGaussian(x*mean(g), Cholesky(x*std(g).L))
Base.:*(x::Number, g::MvGaussian{<:Diagonal}) = MvGaussian(x*mean(g), x*std(g))
Base.:*(g::MvGaussian{<:Diagonal}, x::Number) = MvGaussian(x*mean(g), x*std(g))

gaussian(f, θ::SigmaParams, args::UvGaussian...) = gaussian(f, SigmaWeights(length(args), θ), args...)

function gaussian(f, θ::SigmaWeights, args::UvGaussian...)
    Np = 2*length(args) + 1
    θ  = scale_weights(f, θ, args...)

    old_points = SigmaPoints(θ, args...)
    indvec = SVector{Np}(firstindex(old_points):lastindex(old_points))
    new_points = SigmaPoints(θ, map(ind->f(old_points[ind]...), indvec)) #Non-allocating result

    return gaussian(new_points)
end