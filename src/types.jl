

"""

        Options(; kwargs...)       The default Settings is set by Float64 type
        Options{T<:AbstractFloat}(; kwargs...)

kwargs are from the fields of Options{T<:AbstractFloat} for Float64 and BigFloat

    maxIter::Int64         #700
    gamma::T        #0.99
    tolMu::T        #1e-14
    tolR::T         #1e-14
    minPhi::T       #1e7

"""
struct Options{T<:AbstractFloat}
    maxIter::Int64    #100
    gamma::T   # 0.99
    tolMu::T   #1e-7
    tolR::T   #1e-7
    minPhi::T  #1e10
end

Options(; kwargs...) = Options{Float64}(; kwargs...)

function Options{Float64}(; maxIter=700,
    gamma=0.99,
    tolMu=1e-14,  #2^-26,   #1e-7,
    tolR=1e-14,  #2^-26,   #1e-7,
    minPhi=1e7)
    Options{Float64}(maxIter, gamma, tolMu, tolR, minPhi)
end

function Options{BigFloat}(; maxIter=700,
    gamma=0.99,
    tolMu=1e-17,
    tolR=1e-17,
    minPhi=1e21)
    Options{BigFloat}(maxIter, gamma, tolMu, tolR, minPhi)
end

#=
function Options(P::Problem; kwargs...)
    Options{typeof(P).parameters[1]}(; kwargs...)
end
=#
"""
    
        mQP(V, q::T; A, b, C, g) where T

define the following convex quadratic programming problems (QP)

```math
        min   (1/2)x′Vx+q′x
        s.t.   Ax=b ∈ R^{M}
               Cx≤g ∈ R^{L}
```

For portfolio optimization

    mQP(V, q)   for no short-sale: A = ones(1,N), b = [1],  C = -I, g = zeros(N)
    mQP(V, q, u)    for bounds 0<= x <= u, and thus A = ones(1,N), b = [1],  C = [-I; I], g = [zeros(N); u]

See [`Documentation for EasyQP.jl`](https://github.com/PharosAbad/EasyQP.jl/wiki)

See also [`mpcQP`](@ref)
"""
struct mQP{T<:AbstractFloat}
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

mQP(args...) = mQP{Float64}(args...)

#=
function mQP(P::Problem{T}) where {T}
    (; E, V, u, d, G, g, A, b, N, M, J) = P
    iu = findall(u .< Inf)
    C = [G; -Matrix{T}(I, N, N); Matrix{T}(I, N, N)[iu, :]]
    gq = [g; -d; u[iu]]
    L = J + N + length(iu)
    mQP(V, A, C, -E, b, gq, N, M, L)
end
=#
function mQP(V, q;
    A = ones(1, length(q)),
    b = ones(1),
    C = -Matrix(I, length(q), length(q)),
    g = zeros(length(q)))

    T = typeof(q).parameters[1]
    N::Int32 = length(q)
    (N, N) == size(V) || throw(DimensionMismatch("incompatible dimension: V"))
    
    qq = copy(vec(q))     #make sure vector and a new copy
    Vs = convert(Matrix{T}, (V+V')/2)   #make sure symmetric

    #remove Inf bounds
    g = vec(g)
    ik = findall(.!isinf.(g))
    gb = g[ik]
    Cb = C[ik,:]
    
    M::Int32 = length(b)
    L::Int32 = length(gb)    
    (M, N) == size(A) || throw(DimensionMismatch("incompatible dimension: A"))
    (L, N) == size(Cb) || throw(DimensionMismatch("incompatible dimension: C"))

    mQP{T}(Vs,
        convert(Matrix{T}, copy(A)),   #make a copy, just in case it is modified somewhere
        convert(Matrix{T}, Cb),
        qq,
        convert(Vector{T}, copy(vec(b))),
        convert(Vector{T}, gb), N, M, L)
end

function mQP(V, q, u)
    T = typeof(q).parameters[1]
    N::Int32 = length(q)
    (N, N) == size(V) || throw(DimensionMismatch("incompatible dimension: V"))
    A = ones(T, 1, N)
    b = ones(T, 1)
    iu = findall(u .< Inf)
    C = [-Matrix{T}(I, N, N); Matrix{T}(I, N, N)[iu, :]]
    g = [zeros(T, N); u[iu]]
    L = N + length(iu)
    M = 1
    mQP{T}(V, A, C, q, b, g, N, M, L)
end


struct Variables{T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    s::Vector{T}
end

function Variables(Q::mQP{T}) where {T}
    (; N, M, L) = Q
    x = zeros(T, N)
    y = zeros(T, M)
    z = ones(T, L)
    s = ones(T, L)
    Variables(x, y, z, s)
end

struct Residuals{T<:AbstractFloat}
    rV::Vector{T}
    rA::Vector{T}
    rC::Vector{T}
    rS::Vector{T}
end

function Residuals(Q::mQP{T}) where {T}
    (; N, M, L) = Q
    rV = zeros(T, N)
    rA = zeros(T, M)
    rC = zeros(T, L)
    rS = zeros(T, L)
    Residuals(rV, rA, rC, rS)
end
