#Example: no short-sale with upper bounded for the problem of portfolio selection
#=
	min   (1/2)z′Vz -z′μ
	s.t.   z′1=1 ,  0≤z≤u
=#

using EasyQP

V = [1/100 1/80 1/100
     1/80 1/16 1/40
     1/100 1/40 1/25]
E = [109 / 100; 23 / 20; 119 / 100]

#Q = mQP(V, -E)
#x, status = mpcQP(Q)



u = [0.7; +Inf; 0.7]
Q = mQP(V, -E, u)
#options = Options()
x, status = mpcQP(Q)

#=
using EfficientFrontier
P = Problem(E, V, u)
nS = Settings(P; rule = :maxImprovement)
aCL = EfficientFrontier.ECL(P; numSettings=nS)
aEF = eFrontier(aCL, P)

mu = x.x'*E
z = ePortfolio(mu, aEF)
x.x - z
=#
