"The primal-dual interior point algorithms supplied by OOQP"
module EasyQP
#=
#solving convex quadratic programming (QP) problems in the following form
	min   (1/2)x′Vx+q′x
	s.t.    Ax=b   	 Cx≤g

https://github.com/emgertz/OOQP OOQP in C/C++
https://github.com/oxfordcontrol/qpip/blob/master/qpip.m    solver OOQP, but in a pure matlab implementation
=#

using LinearAlgebra
export Options, mQP
export mpcQP


include("./types.jl")

include("./MPC.jl")


end
