using Revise
using UnscentedTransforms
using Test
using Aqua
using LinearAlgebra
using StaticArrays
using Statistics
import Random

import UnscentedTransforms.choladdleft
import UnscentedTransforms.ArgLimits

@testset "Univariate Math" begin
    (μ1, σ1) = (0.1, 1.0)
    (μ2, σ2) = (1.0, 0.1)
    x1 = μ1 ± σ1 
    x2 = μ2 ± σ2

    #Linear operations
    @test (x1 + x2) == UvGaussian(μ1 + μ2, sqrt(σ1^2 + σ2^2))
    @test (x1 - x2) == UvGaussian(μ1 - μ2, sqrt(σ1^2 + σ2^2))
    @test 2*x1 == UvGaussian(μ1*2, σ1*2)
    @test x1*2 == UvGaussian(μ1*2, σ1*2)
    @test (1 ± 0.5) - 1 ≈ 0 ± 0.5
    @test 1 - (1 ± 0.5) ≈ 0 ± 0.5
    @test (1 ± 0.5) + 1 ≈ 2 ± 0.5
    @test 1 + (1 ± 0.5) ≈ 2 ± 0.5

    #Joint Probabilities
    @test joint(x1) == x1
    @test mean(joint(x1, x2)) ≈ inv(inv(cov(x1)) + inv(cov(x2)))*(inv(cov(x1))*mean(x1) + inv(cov(x2))*mean(x2))
    @test cov(joint(x1, x2)) ≈ inv(inv(cov(x1)) + inv(cov(x2)))
    @test joint(x1, x1) ≈ UvGaussian(mean(x1), std(x1)*sqrt(1/2))
    @test joint(x1, x1, x1) ≈ UvGaussian(mean(x1), std(x1)*sqrt(1/3))
    @test joint(x1, x1, x1, x1) ≈ UvGaussian(mean(x1), std(x1)/2) 

    #Nonlinear operations
    @test mean(x1*x2) ≈ mean(x1)*mean(x2)
    @test x1/(1 ± 0) ≈ x1
    @test std(x1*x2) > 1
    @test sin(0 ± 0.0001) ≈ 0 ± 0.0001
    @test mean(cos(0 ± 0.001)) < 1
    @test std(cos(0 ± 0.001)) < 0.001
    @test mean(tan(0 ± 0.001)) ≈ 0.0
    @test std(tan(0 ± 0.001)) > 0.001
    @test mean(exp(1 ± 0.001)) > exp(1)
    @test std(exp(1 ± 0.001)) > exp(1)*0.001
end


@testset "Multivariate Math" begin 
    x1 = [0.1, 0.2] ± LowerTriangular([0.3 0; 0.1 0.2])
    x2 = MvGaussian( [0.4 ± 0.2, 0.2 ± 0.1] )
    z1 = ± LowerTriangular([1 0; 0.1 1])
    A  = [0.3 -0.1; 0.2 0.4]

    #Linear operations
    @test mean(x1 + x2 + z1) ≈ mean(x1) + mean(x2)
    @test cov(x2 + x1 + z1) ≈ cov(x1) + cov(x2) + cov(z1)
    @test mean(x1 - x2 - z1) ≈ mean(x1) - mean(x2)
    @test cov(x2 - x1 - z1) ≈ cov(x1) + cov(x2) + cov(z1)
    @test mean(x1 + mean(x2)) ≈ mean(x1) + mean(x2)
    @test mean(mean(x1) + x2) ≈ mean(x1) + mean(x2)
    @test cov(x1 + mean(x2)) ≈ cov(x1)
    @test cov(mean(x1) + x2) ≈ cov(x2)
    @test mean(A*x1) ≈ A*mean(x1)
    @test cov(A*x1) ≈ A*cov(x1)*A'
    @test mean(muladd(A, x1, x2)) ≈ A*mean(x1) + mean(x2)
    @test cov(muladd(A, x1, x2)) ≈ A*cov(x1)*A' + cov(x2)
    @test mean(muladd(A, z1, x2)) ≈ mean(x2)
    @test cov(muladd(A, z1, x2)) ≈ A*cov(z1)*A' + cov(x2)

    #Joint probabilities
    @test joint(x1) == x1
    @test mean(joint(x1, x2)) ≈ inv(inv(cov(x1)) + inv(cov(x2)))*(inv(cov(x1))*mean(x1) + inv(cov(x2))*mean(x2))
    @test cov(joint(x1, x2)) ≈ inv(inv(cov(x1)) + inv(cov(x2)))
    @test joint(x1, x1) ≈ MvGaussian(mean(x1), std(x1).L*sqrt(1/2))
    @test joint(x1, x1, x1) ≈ MvGaussian(mean(x1), std(x1).L*sqrt(1/3))
    @test joint(x1, x1, x1, x1) ≈ MvGaussian(mean(x1), std(x1).L/2) 
    

end

@testset "Sigma Points" begin
    Random.seed!(1234)

    σ  = 0.01
    C  = rand(3,5)
    A  = rand(5,5)
    R  = Diagonal(fill(σ^2, 3))
    X  = randn(500, 5)*rand(5,5)
    Yh = X*C' 
    Y  = Yh .+ σ.*randn(500, 3)

    θ  = SigmaParams(α=0.5)
    mx = mean(X, dims=1)[:]
    my = mean(Y, dims=1)[:]
    Sx = cov(X)
    Sy = cov(Y)
    Sxy = cov(X,Y)

    Cx = cholesky(Sx)
    Cy = cholesky(Sy)

    Gx = MvGaussian(mx, Cx)
    Gy = MvGaussian(my, Cy)

    Px  = SigmaPoints(θ, Gx)
    Py  = SigmaPoints(θ, Gy)
    Pyh = map(x->C*x, Px)

    #Test round-trip conversionlas
    Pxh = SigmaPoints(source=collect(Px), weights=Px.weights)
    @test std(MvGaussian(Pxh)).U ≈ std(Gx).U
    @test MvGaussian(Pxh).μ ≈ Gx.μ

    #Test adding distributions and variances
    @test std(Gx + Gx).L ≈ cholesky(Sx + Sx).L
    @test std(Gx - Gx).L ≈ cholesky(Sx + Sx).L
    @test std(Gx + MvGaussian(Cx)).L ≈ cholesky(Sx + Sx).L
    @test std(MvGaussian(Cx) + Gx).L ≈ cholesky(Sx + Sx).L
    @test cov(Px, Px) ≈ cov(Gx)
    @test cov(Px, Pyh) ≈ cov(Pyh, Px)'
    
    #@test add_rcov(Cx.U*C', Cx.U*C').U ≈ cholesky(hermitianpart!(2*C*Sx*C')).U
    @test choladdleft(C*Cx.L, C*Cx.L).L ≈ cholesky(hermitianpart!(2*C*Sx*C')).L
    @test choladdleft(Cx, A*Cx.L).L ≈ cholesky(hermitianpart(Sx + A*Sx*A')).L
    @test std(MvGaussian(Px, Cx)).U ≈ cholesky(Sx).U
end

@testset "Sigma Point Scaling" begin 
    (μ1, σ1) = (0.1, 1.0)
    (μ2, σ2) = (1.0, 0.1)
    x1 = μ1 ± σ1 
    x2 = μ2 ± σ2

    #Scaling close to a domain limit 
    w = SigmaWeights(1, SigmaParams())
    w2 = scale_weights(inv, w, x1)
    @test sqrt(0.5/w2[1]) <= abs(0-mean(x1))/std(x1) #Step must be less than the standard deviation distance to zero
    @test all(d->d>0, SigmaPoints(weights=w2, source=x1)) #No sigma points should cross the 0 threshold
    

    #Multivariate scaling close to a domain limit
    x1 = 0.1 ± 1.0 
    x2 = 0.2 ± 2.0 
    x3 = 0.3 ± 3.0 

    prod3inv(x1::Number, x2::Number, x3::Number) = inv(x1*x2*x3)
    UnscentedTransforms.domainlimits(f::typeof(prod3inv)) = (ArgLimits(0), ArgLimits(0), ArgLimits(0))
    
    X1 = SigmaPoints(SigmaParams(), x1, x2, x3)
    w1 = X1.weights
    w2 = scale_weights(prod3inv, w1, x1, x2, x3)
    X2 = SigmaPoints(w2, x1, x2, x3)

    @test !all(v-> all(x-> x>0, v), X1) #Some of the new points cross the threshold
    @test all(v-> all(x-> x>0, v), X2) #None of the new points cross the threshold

    #Use a multivariate distribution for the same thing 
    prodinv3(x::SVector{3}) = inv(prod(x))
    UnscentedTransforms.domainlimits(f::typeof(prodinv3)) = ArgLimits(SVector(0,0,0))

    xm = MvGaussian(x1, x2, x3)
    X1 = SigmaPoints(SigmaParams(), xm)
    w1 = X1.weights
    w2 = scale_weights(prodinv3, w1, xm)
    X2 = SigmaPoints(w2, xm)
    X3 = SigmaPoints(w2, collect(X2))
    xh = MvGaussian(X3)

    @test !all(v-> all(x-> x>0, v), X1) #Some of the new points cross the threshold
    @test all(v-> all(x-> x>0, v), X2) #None of the new points cross the threshold
    @test std(xh).L ≈ std(xm) #Test round-trip conversion with altered weights


    #Use a more complicated scaling rule
    prodinv(x::AbstractVector) = inv(prod(x))
    UnscentedTransforms.domainlimits(f::typeof(prodinv), ::Type{<:StaticVector{N}}) where N = ArgLimits(zero(SVector{N,Float64}))

    Random.seed!(54321)
    data = randn(100,5)*rand(5,5)
    C = rand(3,5)
    σ = cholesky(cov(data))

    #Test far away from the limits
    x0 = MvGaussian(5 .+ zero(SVector{5}), σ.L)
    Xp = map(x->C*x, SigmaPoints(scale_weights(prodinv, SigmaParams(), x0), x0))
    xh = MvGaussian(Xp)

    @test all(v-> all(x-> x>0, v), Xp) #None of the new points cross the threshold
    @test mean(xh) ≈ C*mean(x0)
    @test std(xh).L*std(xh).U ≈ C*std(x0).L*std(x0).U*C'

    #Test moderately close to the limits
    x0 = MvGaussian(0.1 .+ zero(SVector{5}), σ.L)
    Xp = map(x->C*x, SigmaPoints(scale_weights(prodinv, SigmaParams(), x0), x0))
    xh = MvGaussian(Xp)

    @test all(v-> all(x-> x>0, v), Xp) #None of the new points cross the threshold
    @test mean(xh) ≈ C*mean(x0)
    @test std(xh).L*std(xh).U ≈ C*std(x0).L*std(x0).U*C'

end


