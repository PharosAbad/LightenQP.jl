# Problem-Algorithm-Solver pattern


"""

        optionsQP(; kwargs...)       The default Settings is set by Float64 type
        optionsQP{T<:AbstractFloat}(; kwargs...)

kwargs are from the fields of optionsQP{T<:AbstractFloat} for Float64 and BigFloat

    maxIter::Int64         #700
    scaleStep::T        #0.99   a crude step scaling factor (using Mehrotra's heuristic maybe better)
    tolMu::T        #1e-14  violation of the complementarity condition
    tolR::T         #1e-14  norm(resid) <= tolR * norm(OOQP)
    minPhi::T       #1e7    phi_min_history, not a foolproof test

see [`ooqp-userguide.pdf`](http://www.cs.wisc.edu/~swright/ooqp/ooqp-userguide.pdf) or [`Working with the QP Solver`](https://github.com/emgertz/OOQP/blob/master/doc-src/ooqp-userguide/ooqp4qpsolver.tex)
"""
struct optionsQP{T<:AbstractFloat}
    maxIter::Int64    #100
    scaleStep::T   # 0.99
    tolMu::T   #1e-7
    tolR::T   #1e-7
    minPhi::T  #1e10
end

optionsQP(; kwargs...) = optionsQP{Float64}(; kwargs...)

function optionsQP{Float64}(; maxIter=700,
    scaleStep=0.99,
    tolMu=2^-47,    #1e-14,  #2^-26,   #1e-7,
    tolR=2^-37,    #1e-14,  #2^-26,   #1e-7,
    minPhi=2^23)
    optionsQP{Float64}(maxIter, scaleStep, tolMu, tolR, minPhi)
end

function optionsQP{BigFloat}(; maxIter=700,
    scaleStep=0.99,
    tolMu=2^-87,
    tolR=2^-77,
    minPhi=2^23)
    optionsQP{BigFloat}(maxIter, scaleStep, tolMu, tolR, minPhi)
end


"""
    
        OOQP(V, q::T; A=A, b=b, C=C, g=g) where T

define the following convex quadratic programming problems (called OOQP)

```math
        min   (1/2)x′Vx+q′x
        s.t.   Ax=b ∈ R^{M}
               Cx≤g ∈ R^{L}
```
default values: A = ones(1,N), b = [1],  C = -I, g = zeros(N). Which define a portfolio optimization without short-sale 

For portfolio optimization

    OOQP(V, q)                      : for no short-sale 
    OOQP(V, q, u)                   : for bounds 0 <= x <= u, and thus A = ones(1,N), b = [1],  C = [-I; I], g = [zeros(N); u]
    OOQP(V, A, C, q, b, g, d, u)    : for OOQP + bounds d <= x <= u

See [`Documentation for LightenQP.jl`](https://github.com/PharosAbad/LightenQP.jl/wiki)

See also [`solveOOQP`](@ref), [`solutionQP`](@ref), [`mpcQP`](@ref), [`optionsQP`](@ref)
"""
struct OOQP{T<:AbstractFloat}
    V::Matrix{T}
    A::Matrix{T}
    C::Matrix{T}
    q::Vector{T}
    b::Vector{T}
    g::Vector{T}
    N::Int32
    M::Int32
    L::Int32
end

OOQP(args...) = OOQP{Float64}(args...)

function OOQP(V, q;
    A=ones(1, length(q)),
    b=ones(1),
    C=-Matrix(I, length(q), length(q)),
    g=zeros(length(q)))

    T = typeof(q).parameters[1]
    N::Int32 = length(q)
    (N, N) == size(V) || throw(DimensionMismatch("incompatible dimension: V"))

    qq = copy(vec(q))     #make sure vector and a new copy
    Vs = convert(Matrix{T}, (V + V') / 2)   #make sure symmetric

    #remove Inf bounds
    g = vec(g)
    ik = findall(.!isinf.(g))
    gb = g[ik]
    Cb = C[ik, :]

    M::Int32 = length(b)
    L::Int32 = length(gb)
    (M, N) == size(A) || throw(DimensionMismatch("incompatible dimension: A"))
    (L, N) == size(Cb) || throw(DimensionMismatch("incompatible dimension: C"))

    OOQP{T}(Vs,
        convert(Matrix{T}, copy(A)),   #make a copy, just in case it is modified somewhere
        convert(Matrix{T}, Cb),
        qq,
        convert(Vector{T}, copy(vec(b))),
        convert(Vector{T}, gb), N, M, L)
end

function OOQP(V, q, u)
    T = typeof(q).parameters[1]
    N::Int32 = length(q)
    (N, N) == size(V) || throw(DimensionMismatch("incompatible dimension: V"))
    A = ones(T, 1, N)
    b = ones(T, 1)
    iu = findall(u .< Inf)
    C = [-Matrix{T}(I, N, N); Matrix{T}(I, N, N)[iu, :]]
    g = [zeros(T, N); u[iu]]
    #L = N + length(iu)
    #M = 1
    #OOQP{T}(V, A, C, q, b, g, N, M, L)
    OOQP(V, q; A=A, b=b, C=C, g=g)
end

function OOQP(V, A, C, q, b, g, d, u)
    T = typeof(q).parameters[1]
    N::Int32 = length(q)
    (N, N) == size(V) || throw(DimensionMismatch("incompatible dimension: V"))
    id = findall(d .> -Inf)
    iu = findall(u .< Inf)
    ig = findall(g .< Inf)
    Ce = [C[ig, :]; -Matrix{T}(I, N, N)[id, :]; Matrix{T}(I, N, N)[iu, :]]
    ge = [g[ig]; -d[id]; u[iu]]
    #L = length(ig) + length(id) + length(iu)
    #M = length(b)
    #OOQP{T}(V, A, Ce, q, b, ge, N, M, L)
    OOQP(V, q; A=A, b=b, C=Ce, g=ge)
end

"""
    
    struct solutionQP

Solution strcture to the following convex quadratic programming problems (called OOQP)

```math
        min   (1/2)x′Vx+q′x
        s.t.   Ax=b ∈ R^{M}
               Cx≤g ∈ R^{L}
```

    x   : primal solution
    y   : equality multipliers, dual variables
    z   : inequality multipliers, dual variables
    s   : slack variables

See [`Documentation for LightenQP.jl`](https://github.com/PharosAbad/LightenQP.jl/wiki)

See also [`OOQP`](@ref), [`solveOOQP`](@ref), [`mpcQP`](@ref)
"""
struct solutionQP{T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    s::Vector{T}
end

function solutionQP(Q::OOQP{T}) where {T}
    (; N, M, L) = Q
    x = zeros(T, N)
    y = zeros(T, M)
    z = ones(T, L)
    s = ones(T, L)
    solutionQP(x, y, z, s)
end

struct residQP{T<:AbstractFloat}
    rV::Vector{T}
    rA::Vector{T}
    rC::Vector{T}
    rS::Vector{T}
end

function residQP(Q::OOQP{T}) where {T}
    (; N, M, L) = Q
    rV = zeros(T, N)
    rA = zeros(T, M)
    rC = zeros(T, L)
    rS = zeros(T, L)
    residQP(rV, rA, rC, rS)
end
