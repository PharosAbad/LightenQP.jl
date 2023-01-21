using LightenQP
using Test

@testset "LightenQP.jl" begin
    
V = [1/100 1/80 1/100
     1/80 1/16 1/40
     1/100 1/40 1/25]
E = [109 / 100; 23 / 20; 119 / 100]

Q = OOQP(V, -E)
x, status = mpcQP(Q)

@test status > 0
@test isapprox(x.x[3], 1)

end
