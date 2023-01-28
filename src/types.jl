# Problem-Algorithm-Solver pattern

"""

        OOQP(V, A, C, q, b, g)
        OOQP(V, q::T; A=A, b=b, C=C, g=g) where T

define the following convex quadratic programming problems (called OOQP)

```math
        min   (1/2)x′Vx+q′x
        s.t.   Ax=b ∈ R^{M}
               Cx≤g ∈ R^{L}
```
default values for OOQP(V, q; kwargs...): A = ones(1,N), b = [1],  C = -I, g = zeros(N). Which define a portfolio optimization without short-sale 

For portfolio optimization

    OOQP(V, q)                      : for no short-sale 
    OOQP(V, q, u)                   : for bounds 0 <= x <= u, and thus A = ones(1,N), b = [1],  C = [-I; I], g = [zeros(N); u]

See [`Documentation for LightenQP.jl`](https://github.com/PharosAbad/LightenQP.jl/wiki)

See also [`solveOOQP`](@ref), [`Solution`](@ref), [`mpcQP`](@ref), [`Settings`](@ref)
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
    OOQP(V, q; A=A, b=b, C=C, g=g)
end

function OOQP(V, A, C, q, b, g)
    OOQP(V, q; A=A, b=b, C=C, g=g)
end

#OOQP + bounds d <= x <= u
function OOQP(V, A, C, q, b, g, d, u)
    T = typeof(q).parameters[1]
    N::Int32 = length(q)
    (N, N) == size(V) || throw(DimensionMismatch("incompatible dimension: V"))
    id = findall(d .> -Inf)
    iu = findall(u .< Inf)
    ig = findall(g .< Inf)
    Ce = [C[ig, :]; -Matrix{T}(I, N, N)[id, :]; Matrix{T}(I, N, N)[iu, :]]
    ge = [g[ig]; -d[id]; u[iu]]
    OOQP(V, q; A=A, b=b, C=Ce, g=ge)
end




"""

        Settings(Q::OOQP; kwargs...)        The default Settings to given OOQP
        Settings(; kwargs...)       The default Settings is set by Float64 type
        Settings{T<:AbstractFloat}(; kwargs...)

kwargs are from the fields of Settings{T<:AbstractFloat} for Float64 and BigFloat

    maxIter::Int64      #777
    scaleStep::T        #0.99   a crude step scaling factor (using Mehrotra's heuristic maybe better)
    tol::T              #2^-26 ≈ 1.5e-8   general (not use in OOQP solver)
    tolMu::T            #2^-47 ≈ 7.1e-15  violation of the complementarity condition
    tolR::T             #2^-37 ≈ 7.3e-12  norm(resid) <= tolR * norm(OOQP)
    minPhi::T           #2^23 = 8388608 ≈ 1e7    phi_min_history, not a foolproof test

see [`ooqp-userguide.pdf`](http://www.cs.wisc.edu/~swright/ooqp/ooqp-userguide.pdf) or [`Working with the QP Solver`](https://github.com/emgertz/OOQP/blob/master/doc-src/ooqp-userguide/ooqp4qpsolver.tex)
"""
struct Settings{T<:AbstractFloat}
    maxIter::Int64  #777
    scaleStep::T    # 0.99
    tol::T          #2^-26
    tolMu::T        #2^-47
    tolR::T         #2^-37
    minPhi::T       #2^23
end

Settings(; kwargs...) = Settings{Float64}(; kwargs...)

function Settings{Float64}(; maxIter=777,
    scaleStep=0.99,
    tol=2^-26,
    tolMu=2^-52,
    tolR=2^-47,
    minPhi=2^23)
    Settings{Float64}(maxIter, scaleStep, tol, tolMu, tolR, minPhi)
end

function Settings{BigFloat}(; maxIter=777,
    scaleStep=0.99,
    tol=2^-76,
    tolMu=2^-87,
    tolR=2^-77,
    minPhi=2^23)
    Settings{BigFloat}(maxIter, scaleStep, tol, tolMu, tolR, minPhi)
end

function Settings(Q::OOQP{T}; kwargs...) where {T}
    Settings{T}(; kwargs...)
end




"""
    
    struct Solution

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
struct Solution{T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    s::Vector{T}
end

function Solution(Q::OOQP{T}) where {T}
    (; N, M, L) = Q
    x = zeros(T, N)
    y = zeros(T, M)
    z = ones(T, L)
    s = ones(T, L)
    Solution(x, y, z, s)
end




struct Residuals{T<:AbstractFloat}
    rV::Vector{T}
    rA::Vector{T}
    rC::Vector{T}
    rS::Vector{T}
end

function Residuals(Q::OOQP{T}) where {T}
    (; N, M, L) = Q
    rV = zeros(T, N)
    rA = zeros(T, M)
    rC = zeros(T, L)
    rS = zeros(T, L)
    Residuals(rV, rA, rC, rS)
end
