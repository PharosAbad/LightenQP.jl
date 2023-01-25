#The general quadratic programming formulation recognized by LightenQP

"""

        mpcQP(V, q, A, b, C, g, d, u, h; settings)    :OOQP + 'd≤x≤u' + 'h≤Cx'
        mpcQP(V, q, A, b, C, g, d, u; settings)       :OOQP + 'd≤x≤u'
        mpcQP(V, q, C, g, d, u, h; settings)          :OOQP + 'd≤x≤u' + 'h≤Cx' - 'Ax=b'
        mpcQP(V, q, A, b, C, g; settings)             :OOQP
        mpcQP(V, q, d, u; settings)                   :OOQP + 'd≤x≤u' - 'Ax=b, Cx≤g'
        mpcQP(O::OOQP; settings)                      :OOQP

wrapping `solveOOQP` for quadratic programming problems: `mpcQP(V, q, A, b, C, g, d, u, h)` for (OOQP + 'd≤x≤u' + 'h≤Cx')

```math
    min	(1/2)x′Vx+x′q
    s.t.	Ax=b∈R^{M}
	        h≤Cx≤g∈R^{L}
        	d≤x≤u∈R^{N}
```

`mpcQP(V, q, A, b, C, g, d, u)` for equality and inequality constraints, and lower and upper bounds (OOQP + 'd≤x≤u')

```math
    min	(1/2)x′Vx+x′q
    s.t.	Ax=b∈R^{M}
	        Cx≤g∈R^{L}
	        d≤x≤u∈R^{N}
```

`mpcQP(V, q, C, g, d, u, h)` for only inequality constraints, and lower and upper bounds (OOQP + 'd≤x≤u' + 'h≤Cx' - 'Ax=b')

```math
    min	(1/2)x′Vx+x′q
    s.t.	h≤Cx≤g∈R^{L}
        	d≤x≤u∈R^{N}
```


`mpcQP(V, q, d, u)` for only lower and upper bounds (OOQP + 'd≤x≤u' - 'Ax=b, Cx≤g')

```math
    min	(1/2)x′Vx+x′q
    s.t.	d≤x≤u∈R^{N}
```
See [`Documentation for LightenQP.jl`](https://github.com/PharosAbad/LightenQP.jl/wiki)

See also [`OOQP`](@ref), [`solveOOQP`](@ref), [`Solution`](@ref), [`Settings`](@ref)
"""
function mpcQP(V::Matrix{T}, q::Vector{T}, A::Matrix{T}, b::Vector{T}, C::Matrix{T}, g::Vector{T},
    d::Vector{T}, u::Vector{T}, h::Vector{T}; settings=Settings{T}()) where {T}

    id = findall(d .> -Inf)
    iu = findall(u .< Inf)
    ih = findall(h .> -Inf)
    ig = findall(g .< Inf)
    N = length(q)
    N == length(d) || throw(DimensionMismatch("incompatible dimension: d"))
    N == length(u) || throw(DimensionMismatch("incompatible dimension: u"))
    Cg = [C[ig, :]; -Matrix{T}(I, N, N)[id, :]; Matrix{T}(I, N, N)[iu, :]; -C[ih, :]]
    gg = [g[ig]; -d[id]; u[iu]; -h[ih]]
    O = OOQP(V, q; A=A, b=b, C=Cg, g=gg)
    return solveOOQP(O; settings=settings)
end


#OOQP + 'd≤x≤u'
function mpcQP(V::Matrix{T}, q::Vector{T}, A::Matrix{T}, b::Vector{T}, C::Matrix{T}, g::Vector{T},
    d::Vector{T}, u::Vector{T}; settings=Settings{T}()) where {T}

    id = findall(d .> -Inf)
    iu = findall(u .< Inf)
    ig = findall(g .< Inf)
    N = length(q)
    N == length(d) || throw(DimensionMismatch("incompatible dimension: d"))
    N == length(u) || throw(DimensionMismatch("incompatible dimension: u"))
    Cg = [C[ig, :]; -Matrix{T}(I, N, N)[id, :]; Matrix{T}(I, N, N)[iu, :]]
    gg = [g[ig]; -d[id]; u[iu]]
    O = OOQP(V, q; A=A, b=b, C=Cg, g=gg)
    return solveOOQP(O; settings=settings)
end


#OOQP + 'd≤x≤u' + 'h≤Cx' - 'Ax=b'
function mpcQP(V::Matrix{T}, q::Vector{T}, C::Matrix{T}, g::Vector{T},
    d::Vector{T}, u::Vector{T}, h::Vector{T}; settings=Settings{T}()) where {T}

    id = findall(d .> -Inf)
    iu = findall(u .< Inf)
    ih = findall(h .> -Inf)
    ig = findall(g .< Inf)
    N = length(q)
    N == length(d) || throw(DimensionMismatch("incompatible dimension: d"))
    N == length(u) || throw(DimensionMismatch("incompatible dimension: u"))
    Cg = [C[ig, :]; -Matrix{T}(I, N, N)[id, :]; Matrix{T}(I, N, N)[iu, :]; -C[ih, :]]
    gg = [g[ig]; -d[id]; u[iu]; -h[ih]]
    O = OOQP(V, q; A=zeros(T, 0, N), b=zeros(T, 0), C=Cg, g=gg)
    return solveOOQP(O; settings=settings)
end


#OOQP
function mpcQP(V::Matrix{T}, q::Vector{T}, A::Matrix{T}, b::Vector{T}, C::Matrix{T}, g::Vector{T}; settings=Settings{T}()) where {T}
    #alias for solveOOQP(V, q, A, b, C, g; settings=settings)
    O = OOQP(V, q; A=A, b=b, C=C, g=g)
    return solveOOQP(O; settings=settings)
end

function mpcQP(O::OOQP{T}; settings=Settings{T}()) where {T}
    #alias for solveOOQP(O; settings=settings)
    return solveOOQP(O; settings=settings)
end


#OOQP + 'd≤x≤u' - 'Ax=b, Cx≤g'
function mpcQP(V::Matrix{T}, q::Vector{T}, d::Vector{T}, u::Vector{T}; settings=Settings{T}()) where {T}
    id = findall(d .> -Inf)
    iu = findall(u .< Inf)
    N = length(q)
    N == length(d) || throw(DimensionMismatch("incompatible dimension: d"))
    N == length(u) || throw(DimensionMismatch("incompatible dimension: u"))
    C = [-Matrix{T}(I, N, N)[id, :]; Matrix{T}(I, N, N)[iu, :]]
    g = [-d[id]; u[iu]]
    O = OOQP(V, q; A=zeros(T, 0, N), b=zeros(T, 0), C=C, g=g)
    return solveOOQP(O; settings=settings)
end

