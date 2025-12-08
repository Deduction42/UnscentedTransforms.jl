

#=
#Attempt to use inverse notch filter approach
using ControlSystemsBase
using Symbolics
@variables ρ² Ω₀

A = [2cos(Ω₀) -1; 1 0]
B = [0;0]
C = [-(1+ρ²) ρ²]
D = [0]
Gd = ss(A,B,C,D,1.0)

tfd = tf([1, -(1+ρ²)cos(Ω₀), ρ²], [1, -2cos(Ω₀), 1], 1.0)
=#
using Revise
using UnscentedTransforms
using StaticArrays
using LinearAlgebra
import Random
using Plots; plotlyjs()
const Δt = 0.1

outlier = 3.0 #Inf
Random.seed!(1234)

ω  = 2π/50
N  = 1000
σ  = 0.3
y  = sin.((1:N).*ω) .+ σ*randn(N)
k0 = 100*(ω)^2

#Test spike
y[150] = 15

#Test missing data after stabilization
dy = [0; diff(y)]./Δt
Y  = [dy'; y']
Y[:,200] .= NaN
Y[1,500] = NaN
Y[2,501] = NaN

#Variances
σ₊  = (σ+0.1)
vsQ = [0.01*σ₊/Δt, 0.0001*σ₊, 0.1]
vsR = [2*σ₊/Δt, σ₊]
vsP = [10*σ₊/Δt, 10*σ₊, 10]

QU = UpperTriangular(Matrix(Diagonal(vsQ)))
RU = UpperTriangular(Matrix(Diagonal(vsR)))
PU = UpperTriangular(Matrix(Diagonal(vsP)))

function predictor_func(X, u)
    k = exp(X[3])
    A = @SMatrix [
        0  -k   0;
        1   0   0;
        0   0   0 
    ]
    return exp(A*Δt)*X #Discretize A and then predict result
end

predictor = NonlinearPredictor(f=predictor_func, Σ=Cholesky(QU), θ=SigmaParams(α=1.0))

C = @SMatrix [1 0 0 ; 0 1 0]
D = @SMatrix zeros(2,0)
observer_func(X, u) = C*X

observer = LinearPredictor(C, D, Cholesky(RU))
#observer = NonlinearPredictor(f=observer_func, Σ=Cholesky(RU), θ=SigmaParams(α=1.0))

x0 = @SVector [0, 0, log(k0)]
state = MvGaussian(x0, Cholesky(PU))

model = StateSpaceModel(
    state = state,
    predictor = predictor,
    observer  = observer,
    outlier  = outlier
)

vs = [state]

for ii in 1:N
    kalman_filter!(model, Y[:,ii], Float64[])
    push!(vs, model.state)
end


fig = plot(y, lc=:blue, ls=:dot, label="measured")
plot!(fig, [s.μ[1] for s in vs[1:(end-1)]], label="velocity")
plot!(fig, [s.μ[2] for s in vs[1:(end-1)]], label="position")
#png(fig, joinpath(@__DIR__, "outlier cutoff $(outlier)"))

#plot([sqrt( min(5*σ, s.x[1]^2/exp(s.x[3])) + s.x[2]^2) for s in vs[1:(end-1)]]) #amplitude-equivalent energy

#=
figure()
labels = ["velocity", "position", "log spring"]

labels = ["velocity", "position", "log spring"]
for ii in 1:3
    subplot(3,1,ii)
    plot([s.x[ii] for s in vs[1:(end-1)]])
    ylabel(labels[ii])
end
title("Frequency Tracking Raw State: UKF")


figure()
title("Frequency Tracking Uncertainty: UKF")
labels = ["velocity", "position", "spring"]
for ii in 1:3
    subplot(3,1,ii)
    plot([sqrt(s.P[ii,ii]) for s in vs[1:(end-1)]])
    ylabel(labels[ii])
end

=#