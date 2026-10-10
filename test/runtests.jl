#============================================================================================================================
Run these commands at startup to see coverage
julia --startup-file=no --depwarn=yes --threads=auto -e 'using Coverage; clean_folder("src"); clean_folder("test")'
julia --startup-file=no --depwarn=yes --threads=auto --project=. -e 'using Pkg; Pkg.test(coverage="user")'
julia --startup-file=no --depwarn=yes --threads=auto coverage.jl
============================================================================================================================#

using Revise
using UnscentedTransforms
using Test
using Aqua

const TESTS = ["tests_basic", "tests_ukf", "tests_aqua"]

for t in TESTS
    include("$t.jl")
end
