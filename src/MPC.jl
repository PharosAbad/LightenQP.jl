#the solver, Algorithm MPC (Mehrotra Predictor-Corrector) from OOQP

"""

        solveOOQP(V::Matrix{T}, q::Vector{T}, A::Matrix{T}, b::Vector{T}, C::Matrix{T}, g::Vector{T}; settings=Settings{T}()) where T
        solveOOQP(Q::OOQP; settings=Settings())

solving convex quadratic programming problems (QP) in the following form define by Q::OOQP

```math
        min   (1/2)x′Vx+q′x
        s.t.   Ax=b ∈ R^{M}
               Cx≤g ∈ R^{L}
```

Outputs

    x::Solution   : structure containing the primal solution 'x', dual variables 'y' and 'z' corresponding to the equality 
                     and inequality multipliers respectively, and slack variables 's'
    status::Int     : > 0 if successful (=iter_count), 0 if infeasibility detected, < 0 if not converged (=-iter_count)

# Example
```
    using LightenQP
    V = [1/100 1/80 1/100
         1/80 1/16 1/40
         1/100 1/40 1/25]
    E = [109 / 100; 23 / 20; 119 / 100]
    u = [0.7; +Inf; 0.7]    #Inf means no bounded
    Q = OOQP(V, -E, u)  #OOQP + bounds 0 <= x <= u
    x, status = solveOOQP(Q)    #solve by Algorithm MPC (Mehrotra Predictor-Corrector)
```

See [`Documentation for LightenQP.jl`](https://github.com/PharosAbad/LightenQP.jl/wiki)

See also [`OOQP`](@ref),  [`Solution`](@ref), [`Settings`](@ref), [`mpcQP`](@ref)
"""
function solveOOQP(V::Matrix{T}, q::Vector{T}, A::Matrix{T}, b::Vector{T}, C::Matrix{T}, g::Vector{T}; settings=Settings{T}()) where {T}
    Q = OOQP(V, q; A=A, b=b, C=C, g=g)
    return solveOOQP(Q; settings=settings)
end


function solveOOQP(Q::OOQP{T}; settings=Settings{T}()) where {T}
    #cook up an initial solution
    Soln = Solution(Q)
    res = Residuals(Q)
    J, idxS = initJacobian(Q)
    normD = dataNorm(Q) #data norm initialize
    defaultStart!(Soln, res, Q, J, normD)
    (; z, s) = Soln

    iter = 1
    status = 0
    while iter <= settings.maxIter
        muval = calcMu(Soln)    #get the complementarity measure mu
        calcResiduals!(res, Q, Soln)    ##Update the right hand side residuals
        #termination test
        status = checkStatus(Q, Soln, res, settings, muval, normD)
        if status != 0
            break
        end
        # PREDICTOR STEP    find the RHS for this step        
        J[idxS] = -(s ./ z) #updateJacobian
        F = lu(J)
        #display((typeof(J),typeof(F)))
        stepAff = searchDirection(F, Q, Soln, res)
        alphaAff = stepBound(Soln, stepAff)  #determine the largest step that preserves consistency of the multiplier constraint  
        muAff = muStep(Soln, stepAff, alphaAff)
        sigma = (muAff / muval)^3
        #CENTERING-CORRECTOR STEP   
        rSfix!(res, stepAff, -sigma * muval)
        stepCC = searchDirection(F, Q, Soln, res)
        alphaMax = stepBound(Soln, stepCC)   #determine the largest step that preserves consistency of the multiplier constraint  
        stepSize = alphaMax * settings.scaleStep #use a crude step scaling factor
        addToSolution!(Soln, stepCC, stepSize)   #take the step and update mu
        iter += 1
    end
    iter = (status == 1) ? iter : ((status == 0) ? -iter : 0)
    return Soln, iter
end


function addToSolution!(Soln, step, alpha)
    #shift the problem variables by a scaled correction term
    (; x, y, z, s) = Soln
    x .+= alpha * step.x
    y .+= alpha * step.y
    z .+= alpha * step.z
    s .+= alpha * step.s
    return nothing
end


function calcMu(Soln)
    #calculate complementarity measure
    (; z, s) = Soln
    L = length(z)   # L==0 no inequalities case
    mu = L == 0 ? 0.0 : (z' * s) / L
    return mu
end


function calcResiduals!(res, Q, Soln)
    #Calculates the problem residuals based on the current variables
    (; V, A, C, q, b, g) = Q
    (; x, y, z, s) = Soln
    (; rV, rA, rC, rS) = res
    rV .= V * x - A' * y + C' * z + q   # V*x + q - A'*y + C'*z
    rA .= A * x - b     # A*x - b
    rC .= C * x + s - g # C*x - g + s
    rS .= z .* s    # w = SZ1
    return nothing
end


function checkStatus(Q, Soln, res, settings, muval, normD)
    #test for convergence or infeasibility
    gap = abs(dualityGap(Q, Soln))
    normR = residualNorm(res)
    phi = (normR + gap) ./ normD
    minPhi = min(settings.minPhi, phi)

    if muval <= settings.tolMu && normR <= settings.tolR * normD
        status = 1  #convergence
    elseif (phi > 1e-8 && phi > 1e4 * minPhi)
        status = -1 #infeasible
    else
        status = 0  #not converged
    end
    return status
end


function dataNorm(Q)
    (; V, A, C, q, b, g) = Q
    vecData = [V[:]; A[:]; C[:]; q[:]; b[:]; g[:]]  #vectorize the data, infinite values have been removed    
    return norm(vecData, Inf)
end


function defaultStart!(Soln, res, Q, J, normD)
    (; z, s) = Soln
    s0 = sqrt(normD)    #find some interior point (large z and s)
    z .= s0
    s .= s0
    calcResiduals!(res, Q, Soln)
    F = lu(J)
    step = searchDirection(F, Q, Soln, res)
    addToSolution!(Soln, step, 1.0) #take the full affine scaling step
    z .+= 1e3   #shiftBoundVariables
    s .+= 1e3
    return nothing
end


function dualityGap(Q, Soln)
    #Calculate duality gap = x'Vx + q'x - b'y + g'z
    (; V, q, b, g) = Q
    (; x, y, z) = Soln
    return x' * (V * x) + q' * x - b' * y + g' * z
end


function initJacobian(Q::OOQP{T}) where {T}
    (; V, A, C, N, M, L) = Q
    #Create the Jacoban matrix with a dummy Sigma

    #if T == BigFloat
    S = -Matrix{T}(I, L, L)
    Z1 = zeros(T, M, M)
    Z2 = zeros(T, L, M)
    #=     J = [V A' C'
             A Z1 Z2'
             C Z2 S]
    else
        S = -sparse(I, L, L)
        Z1 = spzeros(M, M)
        Z2 = spzeros(L, M)
        J = sparse(
            [V A' C'
             A Z1 Z2'
             C Z2 S])
    end =#
    #construct the jacobian
    J = [V A' C'
        A Z1 Z2'
        C Z2 S]
    #get the indices for the entries of S
    idx = (N + M) .+ (1:L)
    idxS = Base._sub2ind(size(J), idx, idx)
    return J, idxS
end


function muStep(Soln, step, alpha)
    #calculate the value of z's/L given a step in the proposed z and s directions
    (; z, s) = Soln
    dz = step.z * alpha
    ds = step.s * alpha
    return ((z + dz)' * (s + ds)) / length(s)
end


function residualNorm(res)
    (; rV, rA, rC) = res
    vec = [rV[:]; rA[:]; rC[:]]
    return norm(vec, Inf)   #residual vector norms
end


function rSfix!(res, stepAff, shift)
    #adds to the rS component of the residuals a term =  dZ*dS*1 + shift*1
    dz = stepAff.z
    ds = stepAff.s
    res.rS .+= dz .* ds .+ shift    #update the residuals
    return nothing
end


function searchDirection(F, Q, Soln, res)
    #Solve the Newton system for a given set of residuals, using the current Jacobian factorization
    (; z, s) = Soln
    (; rV, rA, rC, rS) = res
    rC1 = rC - (rS ./ z)    #eliminate the rS terms
    rhs = [rV; rA; rC1] #put them all together
    lhs = F \ rhs #solve it
    #parse the solution (including any post-solving) back into a *copy* of the variables
    (; N, M, L) = Q
    dx = -lhs[1:N]
    dy = lhs[(1:M).+N]
    dz = -lhs[(1:L).+(N+M)]
    #post-solve solve for the ds
    ds = -(rS + s .* dz) ./ z
    return Solution(dx, dy, dz, ds)    #the output should have the same structure as the Soln
end


function stepBound(Soln, step)
    #calculate the maximum allowable step in the proposed direction in (0 1]
    d = [step.z; step.s]   #the proposed z and s directions
    t = [Soln.z; Soln.s]
    am = 1.0
    id = findall(d .< 0)    #leave d>=0 alone
    if length(id) > 0
        a = t[id] ./ d[id]
        am = min(1.0, -maximum(a))
    end
    return am
end

