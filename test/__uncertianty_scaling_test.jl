using Revise
using UnscentedTransforms
using Statistics
using Plots#; plotlyjs()

#Create an inversion function with no limits
myinv(x) = inv(x)
x = UvGaussian(0.1, 1.0)
w = SigmaWeights(1, SigmaParams(α=1))

new_std(σ::Number) = std(gaussian(inv, w, UvGaussian(mean(x), σ)))
new_std_raw(σ::Number) = std(gaussian(myinv, w, UvGaussian(mean(x), σ)))
new_std_lin(σ::Number) = σ*inv(mean(x))^2

vσ = 0.001:0.001:0.2
fig = plot()
plot!(vσ, new_std_lin.(vσ), label="linear")
plot!(vσ, new_std_raw.(vσ), label="raw unscented")
plot!(vσ, new_std.(vσ), label="scaled unscented")

png(fig, joinpath(@__DIR__, "uncertainty scaling"))
display(fig)

#=
plot!(vσ, new_std_lin.(vσ), label="linear", yaxis=:log10)
plot!(vσ, new_std_raw.(vσ), label="raw unscented", yaxis=:log10)
plot!(vσ, new_std.(vσ), label="scaled unscented", yaxis=:log10)
=#