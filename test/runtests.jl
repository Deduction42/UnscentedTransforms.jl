using Revise
using UnscentedTransforms
using Test
using Aqua
using LinearAlgebra
using StaticArrays
using Statistics
import Random

import UnscentedTransforms.add_cov
import UnscentedTransforms.add_lcov
import UnscentedTransforms.add_rcov
import UnscentedTransforms.ArgLimits

@testset "Basic Math" begin
    (μ1, σ1) = (0.1, 1.0)
    (μ2, σ2) = (1.0, 0.1)
    x1 = μ1 ± σ1 
    x2 = μ2 ± σ2

    #Adding/subtracting
    @test (x1 + x2) == UvGaussian(μ1 + μ2, sqrt(σ1^2 + σ2^2))
    @test (x1 - x2) == UvGaussian(μ1 - μ2, sqrt(σ1^2 + σ2^2))
    @test 2*x1 == UvGaussian(μ1*2, σ1*2)
    @test x1*2 == UvGaussian(μ1*2, σ1*2)


    #Scaling close to a domain limit 
    w = SigmaWeights(1, SigmaParams())
    w2 = scale_spread(inv, w, x1)
    @test w2.rc < abs(0-mean(x1))/std(x1) #Step must be less than the standard deviation distance to zero
    @test all(d->d>0, SigmaPoints(weights=w2, source=x1)) #No sigma points should cross the 0 threshold

    #Multivariate scaling close to a domain limit
    x1 = 0.1 ± 1.0 
    x2 = 0.2 ± 2.0 
    x3 = 0.3 ± 3.0 

    prod3inv(x1, x2, x3) = inv(x1*x2*x3)
    UnscentedTransforms.domainlimits(f::typeof(prod3inv)) = (ArgLimits(0), ArgLimits(0), ArgLimits(0))
    
    X1 = SigmaPoints(SigmaParams(), x1, x2, x3)
    w1 = X1.weights
    w2 = scale_spread(prod3inv, w1, x1, x2, x3)
    X2 = SigmaPoints(w2, x1, x2, x3)

    @test !all(v-> all(x-> x>0, v), X1) #Some of the new points cross the threshold
    @test all(v-> all(x-> x>0, v), X2) #None of the new points cross the threshold

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
    Pyh = SigmaPoints(source=map(x->C*x, Px), weights=Px.weights)

    #Test round-trip conversion
    Pxh = SigmaPoints(source=collect(Px), weights=Px.weights)
    @test std(MvGaussian(Pxh)).U ≈ std(Gx).U
    @test MvGaussian(Pxh).μ ≈ Gx.μ

    #Test adding varainces
    @test cov(Px, Px) ≈ cov(Px)
    @test cov(Px, Pyh) ≈ cov(Pyh, Px)'

    @test add_cov(Cx, Cx).U ≈ cholesky(Sx + Sx).U
    @test add_lcov(Cx, A*Cx.L).L ≈ cholesky(hermitianpart(Sx + A*Sx*A')).L
    @test add_rcov(Cx.U*C', Cx.U*C').U ≈ cholesky(hermitianpart!(2*C*Sx*C')).U
    @test add_lcov(C*Cx.L, C*Cx.L).L ≈ cholesky(hermitianpart!(2*C*Sx*C')).L
    @test std(MvGaussian(Px, Cx)).U ≈ cholesky(Sx).U
end

@testset "State Space" begin
    #=========================================================================================================================
    Set up classic Kalman filter equations
    =========================================================================================================================#
    function classic_predict(obs::LinearPredictor, x::MvGaussian{Tμ,TΣ}, u) where {Tμ,TΣ}
        (A, B, Q, P) = (obs.A, obs.B, cov(obs.ε), cov(x))

        μ = A*x.μ + B*u
        Σ = hermitianpart!(A*P*A' + Q)
        return MvGaussian{Tμ,TΣ}(μ, cholesky(Σ))
    end

    function classic_update(obs::LinearPredictor, x::MvGaussian{Tμ,TΣ}, y::AbstractVector, u) where {Tμ,TΣ}
        (C, D, R, P) = (obs.A, obs.B, cov(obs.ε), cov(x))

        z  = y .- C*x.μ .+ D*u #Innovation 
        S  = C*P*C' + R #Innovation covariance 
        K  = (P*C')/S
        μ  = x.μ + K*z
        Σ  = hermitianpart!((I-K*C)*P)
        return MvGaussian{Tμ,TΣ}(μ, cholesky(Σ))
    end

    function classic_kalman!(ss::StateSpaceModel, y::AbstractVector, u)
        x = classic_predict(ss.predictor, ss.state, u)
        ss.state = classic_update(ss.observer, x, y, u)
        return ss 
    end


    #=========================================================================================================================
    Set up test system and simulation
    =========================================================================================================================#
    Random.seed!(1234)

    #Build the LTI system
    σω = 0.02
    σε = 0.01
    
    A  = exp(SA[
            -0.1    0.1   -0.1;
             0.1   -0.1    0.0;
             0.0    0.1    -0.1
        ])
    
    B  = @SMatrix [1.0; 1.0; 0]
    C  = SA[
        1.0 0.0 0.0;
        0.0 0.0 1.0
    ]
    D  = @SMatrix [0.0; 0.0]
    sQ = Diagonal(fill(σω, 3))
    sR = Diagonal(fill(σε, 2))
    sP = sQ*sqrt(10)

    #Simulate the inputs and outputs
    N  = 500
    U  = cumsum(randn(1,N), dims=2) .> 0
    ε  = σε.*randn(2,N)
    ω  = σω.*randn(3,N)
    X  = zeros(3,N)
    Y  = zeros(2,N)

    #Fill out the data from the simulated inputs
    Y[:,1] = C*X[:,1] + D*U[:,1]
    for ii in 2:N 
        u0 = U[:,ii-1]
        x0 = SVector{3}(X[:,ii-1])
        x  = A*x0 + B*u0 + ω[:,ii]
        y  = C*x0 + D*u0
        X[:,ii] = x
        Y[:,ii] = y
    end

    #Build a linear test system
    lin_state = correlated(MvGaussian(SVector{3}(X[:,1]), copy(sP)))
    lin_pred = LinearPredictor(A, B, sQ)
    lin_obs  = LinearPredictor(C, D, sR)
    lin_sys  = StateSpaceModel(state=lin_state, predictor=lin_pred, observer=lin_obs)

    #Build a nonlinear test system
    f_predict(x, u) = A*x + B*u 
    f_observe(x, u) = C*x + D*u 
    nl_state = correlated(MvGaussian(SVector{3}(X[:,1]), copy(sP)))
    nl_pred = NonlinearPredictor(f_predict, sQ, SigmaParams(), false)
    nl_obs  = NonlinearPredictor(f_observe, sR, SigmaParams(), false)
    nl_sys  = StateSpaceModel(state=nl_state, predictor=nl_pred, observer=nl_obs)


    #=========================================================================================================================
    Test single step predictions 
    =========================================================================================================================#
    state  = correlated(MvGaussian(SVector{3}(X[:,1]), copy(sP)))
    X_classicpred = classic_predict(lin_sys.predictor, state, U[:,1])
    X_linearpred  = predict(lin_sys.predictor, state, U[:,1])
    X_nonlinpred  = predict(nl_sys.predictor, state, U[:,1])

    #Linear/Classic consistency, predictions
    @test mean(X_linearpred) ≈ mean(X_classicpred)
    @test cov(X_linearpred) ≈ cov(X_classicpred)

    #Linear/Nonlinear consistency, predictions
    @test mean(X_linearpred) ≈ mean(X_nonlinpred)
    @test cov(X_linearpred) ≈ cov(X_nonlinpred)


    #=========================================================================================================================
    Test single step updates
    =========================================================================================================================#
    X_classicpost = classic_update(lin_sys.observer, state, Y[:,1], U[:,1])
    X_linearpost  = update(lin_sys.observer, state, Y[:,1], U[:,1]).X
    X_nonlinpost  = update(nl_sys.observer, state, Y[:,1], U[:,1]).X

    #Linear/Classic consistency, updates
    @test mean(X_linearpost) ≈ mean(X_classicpost)
    @test cov(X_linearpost) ≈ cov(X_classicpost)

    #Linear/Nonlinear consistency, updates
    @test mean(X_linearpost) ≈ mean(X_nonlinpost)
    @test cov(X_linearpost) ≈ cov(X_nonlinpost)

    #=========================================================================================================================
    Test long history consistency
    =========================================================================================================================#
    Xclassic = copy(X)
    for ii in 2:N 
        classic_kalman!(lin_sys, SVector{2}(Y[:,ii]), U[:,ii-1])
        Xclassic[:,ii] = lin_sys.state.μ
    end

    #Run the linear Kalman filter through long history
    Xlinear  = copy(X)
    lin_sys.state = correlated(MvGaussian(SVector{3}(X[:,1]), copy(sP)))
    for ii in 2:N 
        kalman_filter!(lin_sys, SVector{2}(Y[:,ii]), U[:,ii-1])
        Xlinear[:,ii] = lin_sys.state.μ
    end

    #Run the uncented kalman filter through long history
    Xnonlin  = copy(X)
    for ii in 2:N 
        kalman_filter!(nl_sys, SVector{2}(Y[:,ii]), U[:,ii-1])
        Xnonlin[:,ii] = nl_sys.state.μ
    end

    #Test linear/classic/nonlinear consistency for the final result
    @test Xlinear[:,N] ≈ Xclassic[:,N]
    @test Xlinear[:,N] ≈ Xnonlin[:,N]


    #=========================================================================================================================
    Test multithreaded nonlinear system 
    =========================================================================================================================#
    nl_state = MvGaussian(SVector{3}(X[:,1]), copy(sP))
    nl_pred = NonlinearPredictor(f_predict, sQ, SigmaParams(), true)
    nl_obs  = NonlinearPredictor(f_observe, sR, SigmaParams(), true)
    nl_sys  = StateSpaceModel(state=nl_state, predictor=nl_pred, observer=nl_obs)

    X_nonlinpred  = predict(nl_sys.predictor, state, U[:,1])
    X_nonlinpost  = update(nl_sys.observer, state, Y[:,1], U[:,1]).X

    #Linear/Nonlinear consistency, updates
    @test mean(X_linearpost) ≈ mean(X_nonlinpost)
    @test cov(X_linearpost) ≈ cov(X_nonlinpost)

    @test mean(X_linearpred) ≈ mean(X_nonlinpred)
    @test cov(X_linearpred) ≈ cov(X_nonlinpred)
end

@testset "Aqua.jl" begin
    Aqua.test_all(UnscentedTransforms)
end

nothing