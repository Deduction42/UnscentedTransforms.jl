using UnscentedTransforms
using LinearAlgebra
using StaticArrays
using Statistics
import Random

import UnscentedTransforms.choladdleft


@testset "State Space" begin
    #=========================================================================================================================
    Set up classic Kalman filter equations
    =========================================================================================================================#
    function classic_predict(obs::LinearPredictor, x::MvGaussian{Tμ,TΣ}, u) where {Tμ,TΣ}
        (A, B, Q, P) = (obs.A, obs.B, cov(obs.ε), cov(x))

        μ = A*x.μ + B*u
        Σ = hermitianpart!(A*P*A' + Q, :L)
        return MvGaussian{Tμ,TΣ}(μ, cholesky(Σ))
    end

    function classic_update(obs::LinearPredictor, x::MvGaussian{Tμ,TΣ}, y::AbstractVector, u) where {Tμ,TΣ}
        (C, D, R, P) = (obs.A, obs.B, cov(obs.ε), cov(x))

        z  = y .- C*x.μ .+ D*u #Innovation 
        S  = C*P*C' + R #Innovation covariance 
        K  = (P*C')/S
        μ  = x.μ + K*z
        Σ  = hermitianpart!((I-K*C)*P, :L)
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