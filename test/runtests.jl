using UnscentedTransforms
using Revise
using Test
using LinearAlgebra
using StaticArrays
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

    Gx = MvGaussian(mx, Cx)
    Gy = MvGaussian(my, Cy)

    Px  = SigmaPoints(Gx, θ)
    Py  = SigmaPoints(Gy, θ)
    Pyh = SigmaPoints(points = map(x->C*x, Px.points), weights=Px.weights)

    #Test round-trip conversion
    @test MvGaussian(Px).Σ.U ≈ Gx.Σ.U
    @test MvGaussian(Px).μ ≈ Gx.μ

    #Test adding varainces
    @test cov(Px, Px) ≈ cov(Px)
    @test cov(Px, Pyh) ≈ cov(Pyh, Px)'

    @test add_cov(Cx, Cx).U ≈ cholesky(Sx + Sx).U
    @test add_lcov(Cx, A*Cx.L).L ≈ cholesky(hermitianpart(Sx + A*Sx*A')).L
    @test add_rcov(Cx.U*C', Cx.U*C').U ≈ cholesky(hermitianpart!(2*C*Sx*C')).U
    @test add_lcov(C*Cx.L, C*Cx.L).L ≈ cholesky(hermitianpart!(2*C*Sx*C')).L
    @test MvGaussian(Px, Cx).Σ.U ≈ cholesky(Sx + Sx).U
    # Write your tests here.
end

@testset "State Space" begin

    #Classic Kalman filter equations for reference
    function classic_predict(obs::LinearPredictor, X::MvGaussian{Tμ,TΣ}, u) where {Tμ,TΣ}
        (A, B, Q, P) = (obs.A, obs.B, Matrix(obs.Σ), Matrix(X.Σ))

        μ = A*X.μ + B*u
        Σ = hermitianpart!(A*P*A' + Q)
        return MvGaussian{Tμ,TΣ}(μ, cholesky(Σ))
    end

    function classic_update(obs::LinearPredictor, X::MvGaussian{Tμ,TΣ}, y::AbstractVector, u) where {Tμ,TΣ}
        (C, D, R, P) = (obs.A, obs.B, Matrix(obs.Σ), Matrix(X.Σ))

        z  = y .- C*X.μ .+ D*u #Innovation 
        S  = C*P*C' + R #Innovation covariance 
        K  = (P*C')/S
        μ  = X.μ + K*z
        Σ  = hermitianpart!((I-K*C)*P)
        return MvGaussian{Tμ,TΣ}(μ, cholesky(Σ))
    end

    function classic_kalman!(ss::StateSpaceModel, y::AbstractVector, u)
        x = classic_predict(ss.predictor, ss.state, u)
        ss.state = classic_update(ss.observer, x, y, u)
        return ss 
    end


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
    Q  = Matrix(Diagonal(fill(σω^2, 3)))
    R  = Matrix(Diagonal(fill(σε^2, 2)))
    P  = Q*10

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
    lin_state = MvGaussian(SVector{3}(X[:,1]), copy(P))
    lin_pred = LinearPredictor(A, B, Q)
    lin_obs  = LinearPredictor(C, D, R)
    lin_sys  = StateSpaceModel(state=lin_state, predictor=lin_pred, observer=lin_obs)


    #Build a nonlinear test system
    f_predict(x, u) = A*x + B*u 
    f_observe(x, u) = C*x + D*u 
    nl_state = MvGaussian(SVector{3}(X[:,1]), copy(P))
    nl_pred = NonlinearPredictor(f_predict, cholesky(Q), SigmaParams(), false)
    nl_obs  = NonlinearPredictor(f_observe, cholesky(R), SigmaParams(), false)
    nl_sys  = StateSpaceModel(state=nl_state, predictor=nl_pred, observer=nl_obs)

    #Test single step predictions
    state  = MvGaussian(SVector{3}(X[:,1]), copy(P))
    X_classicpred = classic_predict(lin_sys.predictor, state, U[:,1])
    X_linearpred  = predict(lin_sys.predictor, state, U[:,1])
    X_nonlinpred  = predict(nl_sys.predictor, state, U[:,1])

    #Linear/Classic consistency, predictions
    @test X_linearpred.μ ≈ X_classicpred.μ
    @test Matrix(X_linearpred.Σ) ≈ Matrix(X_classicpred.Σ)

    #Linear/Nonlinear consistency, predictions
    @test X_linearpred.μ ≈ X_nonlinpred.μ
    @test Matrix(X_linearpred.Σ) ≈ Matrix(X_nonlinpred.Σ)


    #Test single step updates
    X_classicpost = classic_update(lin_sys.observer, state, Y[:,1], U[:,1])
    X_linearpost  = update(lin_sys.observer, state, Y[:,1], U[:,1]).X
    X_nonlinpost  = update(lin_sys.observer, state, Y[:,1], U[:,1]).X

    #Linear/Classic consistency, updates
    @test X_linearpost.μ ≈ X_classicpost.μ
    @test Matrix(X_linearpost.Σ) ≈ Matrix(X_classicpost.Σ)

    #Linear/Nonlinear consistency, updates
    @test X_linearpost.μ ≈ X_nonlinpost.μ
    @test Matrix(X_linearpost.Σ) ≈ Matrix(X_nonlinpost.Σ)


    #Run the classic Kalman filter through long history
    Xclassic = copy(X)
    for ii in 2:N 
        classic_kalman!(lin_sys, SVector{2}(Y[:,ii]), U[:,ii-1])
        Xclassic[:,ii] = lin_sys.state.μ
    end

    #Run the linear Kalman filter through long history
    Xlinear  = copy(X)
    lin_sys.state = MvGaussian(SVector{3}(X[:,1]), copy(P))
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


end
