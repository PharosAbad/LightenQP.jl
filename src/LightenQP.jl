"The primal-dual interior point algorithms supplied by OOQP"
module LightenQP

#=
#solving convex quadratic programming (QP) problems in the following form (called by OOQP)
	min   (1/2)x′Vx+q′x
	s.t.    Ax=b   	 Cx≤g

https://github.com/emgertz/OOQP OOQP in C/C++
https://github.com/oxfordcontrol/qpip/blob/master/qpip.m    solver OOQP, but in a pure matlab implementation
=#

using LinearAlgebra
#using SparseArrays
export OOQP	#, Settings, Solution
export solveOOQP, mpcQP

include("./types.jl")

# Algorithm MPC (Mehrotra Predictor-Corrector), Primal-Dual Interior-Point Algorithms
include("./MPC.jl")

#The general quadratic programming formulation recognized by LightenQP, solved by solveOOQP (Algorithm MPC)
include("./mpcQP.jl")

end
