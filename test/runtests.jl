using TestItems: @testitem
using TestItemRunner
using Revise

@testitem "Basic Functionality" begin
    include("tests_basic.jl")
end

@testitem "Unscented Kalman Filter" begin
    include("tests_ukf.jl")
end

@testitem "Aqua.jl" begin
    using UnscentedTransforms
    using Aqua
    Aqua.test_all(UnscentedTransforms)
end

nothing