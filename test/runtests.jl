#============================================================================================================================
Run these commands at startup to see coverage
julia --startup-file=no --depwarn=yes --threads=auto -e 'using Coverage; clean_folder("src"); clean_folder("test")'
julia --startup-file=no --depwarn=yes --threads=auto --code-coverage=user --project=. -e 'using Pkg; Pkg.test(coverage=true)'
julia --startup-file=no --depwarn=yes --threads=auto coverage.jl
============================================================================================================================#

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