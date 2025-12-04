using UnscentedTransforms
using Revise
using Test
using LinearAlgebra
import Statistics.mean
import Statistics.cov
import Random

import UnscentedTransforms.add_cov
import UnscentedTransforms.add_lcov
import UnscentedTransforms.add_rcov

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

    Gx = GaussianVar(mx, Cx)
    Gy = GaussianVar(my, Cy)

    Px  = SigmaPoints(Gx, θ)
    Py  = SigmaPoints(Gy, θ)
    Pyh = SigmaPoints(points = map(x->C*x, Px.points), weights=Px.weights)

    #Test round-trip conversion
    @test GaussianVar(Px).Σ.U ≈ Gx.Σ.U
    @test GaussianVar(Px).μ ≈ Gx.μ

    #Test adding varainces
    @test cov(Px, Px) ≈ cov(Px)
    @test cov(Px, Pyh) ≈ cov(Pyh, Px)'

    @test add_cov(Cx, Cx).U ≈ cholesky(Sx + Sx).U
    @test add_lcov(Cx, A*Cx.L).L ≈ cholesky(hermitianpart(Sx + A*Sx*A')).L
    @test add_rcov(Cx.U*C', Cx.U*C').U ≈ cholesky(hermitianpart!(2*C*Sx*C')).U
    @test add_lcov(C*Cx.L, C*Cx.L).L ≈ cholesky(hermitianpart!(2*C*Sx*C')).L
    @test GaussianVar(Px, Cx).Σ.U ≈ cholesky(Sx + Sx).U
    # Write your tests here.
end

@testset "State Space" begin
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

    Gx = GaussianVar(mx, Cx)
    Gy = GaussianVar(my, Cy)

    Px  = SigmaPoints(Gx, θ)
    Py  = SigmaPoints(Gy, θ)
    Pyh = SigmaPoints(points = map(x->C*x, Px.points), weights=Px.weights)

    #Test round-trip conversion
    @test GaussianVar(Px).Σ.U ≈ Gx.Σ.U
    @test GaussianVar(Px).μ ≈ Gx.μ

    #Test adding varainces
    @test cov(Px, Px) ≈ cov(Px)
    @test cov(Px, Pyh) ≈ cov(Pyh, Px)'

    @test add_cov(Cx, Cx).U ≈ cholesky(Sx + Sx).U
    @test add_lcov(Cx, A*Cx.L).L ≈ cholesky(hermitianpart(Sx + A*Sx*A')).L
    @test add_rcov(Cx.U*C', Cx.U*C').U ≈ cholesky(hermitianpart!(2*C*Sx*C')).U
    @test add_lcov(C*Cx.L, C*Cx.L).L ≈ cholesky(hermitianpart!(2*C*Sx*C')).L
    @test GaussianVar(Px, Cx).Σ.U ≈ cholesky(Sx + Sx).U
    # Write your tests here.
end

