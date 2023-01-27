#Example: no short-sale with upper bounded for the problem of portfolio selection
#=
	min   (1/2)z′Vz -z′μ
	s.t.   z′1=1 ,  0≤z≤u
=#
# User Guides: https://github.com/PharosAbad/LightenQP.jl/wiki/User-Guides#portfolio-selection-1

using LightenQP
using LinearAlgebra

V = [1/100 1/80 1/100
     1/80 1/16 1/40
     1/100 1/40 1/25]
E = [109 / 100; 23 / 20; 119 / 100]

#Q = OOQP(V, -E)
#x, status = solveOOQP(Q)



u = [0.7; +Inf; 0.7]     #Inf means no bounded
Q = OOQP(V, -E, u)       #OOQP + bounds 0 <= x <= u
#settings = Settings()
x, status = solveOOQP(Q) #solve by Algorithm MPC (Mehrotra Predictor-Corrector)

#v1.0.1
O = OOQP(V, E, u)
y, statusy = fPortfolio(O; L=1.0)
norm(x.x-y.x, Inf)  #0



#=
using EfficientFrontier
P = Problem(E, V, u)
nS = Settings(P)
aCL = EfficientFrontier.ECL(P; numSettings=nS)
aEF = eFrontier(aCL, P)

mu = x.x'*E
z = ePortfolio(aEF, mu)
x.x - z
=#



