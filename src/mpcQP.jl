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





"""
        mpcLP(q, A, b, C, g, d, u; settings, min=true)
        mpcLP(q, A, b, C, g; settings, min=true)
        mpcLP(O::OOQP; settings, min=true)

`mpcLP(O::OOQP; settings)`: wrapping `solveOOQP` for linear programming problems (set `V=0` in `OOQP`). If `min=false`, we maximize the objective function

`mpcLP(q, A, b, C, g, d, u)` for equality and inequality constraints, and lower and upper bounds

```math
    min     x′q
    s.t.    Ax=b∈R^{M}
            Cx≤g∈R^{L}
            d≤x≤u∈R^{N}
```

`mpcLP(q, A, b, C, g)` for equality and inequality constraints

```math
    min     x′q
    s.t.    Ax=b∈R^{M}
            Cx≤g∈R^{L}
```

See [`Documentation for LightenQP.jl`](https://github.com/PharosAbad/LightenQP.jl/wiki)

See also [`OOQP`](@ref), [`solveOOQP`](@ref), [`Solution`](@ref), [`Settings`](@ref)
"""
function mpcLP(q::Vector{T}, A::Matrix{T}, b::Vector{T}, C::Matrix{T}, g::Vector{T}; settings=Settings{T}(), min=true) where {T}
    N = length(q)
    sgn = min == true ? 1 : -1
    O = OOQP(zeros(T, N, N), sgn * q; A=A, b=b, C=C, g=g)
    return solveOOQP(O; settings=settings)
end

function mpcLP(O::OOQP{T}; settings=Settings{T}(), min=true) where {T}
    (; A, C, q, b, g, N) = O
    sgn = min == true ? 1 : -1
    O0 = OOQP(zeros(T, N, N), sgn * q; A=A, b=b, C=C, g=g)
    return solveOOQP(O0; settings=settings)
end

#OOQP + 'd≤x≤u'
function mpcLP(q::Vector{T}, A::Matrix{T}, b::Vector{T}, C::Matrix{T}, g::Vector{T},
    d::Vector{T}, u::Vector{T}; settings=Settings{T}(), min=true) where {T}

    id = findall(d .> -Inf)
    iu = findall(u .< Inf)
    ig = findall(g .< Inf)
    N = length(q)
    N == length(d) || throw(DimensionMismatch("incompatible dimension: d"))
    N == length(u) || throw(DimensionMismatch("incompatible dimension: u"))
    Cg = [C[ig, :]; -Matrix{T}(I, N, N)[id, :]; Matrix{T}(I, N, N)[iu, :]]
    gg = [g[ig]; -d[id]; u[iu]]
    sgn = min == true ? 1 : -1
    O = OOQP(zeros(T, N, N), sgn * q; A=A, b=b, C=Cg, g=gg)
    return solveOOQP(O; settings=settings)
end


"""
        x, status = MVPortfolio(O::OOQP, mu; settings, mu::T=-Inf)
        x, status = MVPortfolio(O::OOQP; settings, L::T=-Inf)

find the minimum variance portfolio: See [`Portfolio Selection · LightenQP`](https://github.com/PharosAbad/LightenQP.jl/wiki/User-Guides#portfolio-selection-1)

    mu=-Inf         :MVP(L=0),  LVEP (Lowest Variance Efficient Portfolio)
    mu=+Inf         :MVP(L=+Inf), HVEP (Highest Variance Efficient Portfolio)
    mu=mu0          :MVP(mu=mu0), the minimum variance portfolio at mu=mu0
    L=-Inf         :MVP(L=0),  LVEP (Lowest Variance Efficient Portfolio)
    L=+Inf         :MVP(L=+Inf), HVEP (Highest Variance Efficient Portfolio)
    L=L0           :MVP(L=L0), the minimum variance portfolio at L=L0

See also [`OOQP`](@ref), [`solveOOQP`](@ref), [`Solution`](@ref), [`Settings`](@ref)
"""
function MVPortfolio(O::OOQP{T}, mu; settings=Settings{T}()) where {T}
    #MVP(mu=mu)
    (; V, A, C, q, b, g, N, M, L) = O
    if mu == -Inf  #LVEP (Lowest Variance Efficient Portfolio)
        qq = zeros(T, N)
        Q = OOQP{T}(V, A, C, qq, b, g, N, M, L)
        return solveOOQP(Q; settings=settings)
    end
    mu1 = mu
    if mu == Inf   #HVEP (Highest Variance Efficient Portfolio)
        #find the Highest mu
        x, status = mpcLP(q, A, b, C, g; settings=settings, min=false)
        if status == 0
            error("HVEP: infeasible")
        elseif status < 0
            error("HVEP: not converged")
        end
        mu1 = x.x' * q
    end
    #@ given mu1
    qq = zeros(T, N)
    Aq = [A; q']
    bq = [b; mu1]
    M += 1
    Q = OOQP{T}(V, Aq, C, qq, bq, g, N, M, L)
    return solveOOQP(Q; settings=settings)
end

function MVPortfolio(O::OOQP{T}; settings=Settings{T}(), L::T=-Inf) where {T}
    #MVP(L=L)
    (; V, A, C, q, b, g, N, M) = O
    if L == -Inf  #LVEP (Lowest Variance Efficient Portfolio)
        qq = zeros(T, N)
        Q = OOQP{T}(V, A, C, qq, b, g, N, M, O.L)
        return solveOOQP(Q; settings=settings)
    elseif L == Inf   #HVEP (Highest Variance Efficient Portfolio)
        #find the Highest mu
        x, status = mpcLP(q, A, b, C, g; settings=settings, min=false)
        if status == 0
            error("HVEP: infeasible")
        elseif status < 0
            error("HVEP: not converged")
        end
        mu1 = x.x' * q
        qq = zeros(T, N)
        Aq = [A; q']
        bq = [b; mu1]
        M += 1
        Q = OOQP{T}(V, Aq, C, qq, bq, g, N, M, O.L)
        return solveOOQP(Q; settings=settings)
    end
    #@ given L
    qq = -L * q
    Q = OOQP{T}(V, A, C, qq, b, g, N, M, O.L)
    return solveOOQP(Q; settings=settings)

end

function ePortfolio0(O::OOQP{T}; settings=Settings{T}(), mu::T=-Inf, Le::T=0.0) where {T}
    #mu=-Inf => @ given Le; mu=+Inf => HVEP; mu=mu0 => @ given mu0
    (; V, A, C, q, b, g, N, M, L) = O
    #=Aq = A
    bq = b
    qq = zeros(T, N)
    if mu == Inf  #EfficientFrontier @ L=1
        qq = -q
    elseif mu != -Inf   #given mu
        Aq = [A; q']
        bq = [b; mu]
        M += 1
    end =#
    if mu == -Inf  #EfficientFrontier by given Le
        qq = -Le * q
        Q = OOQP{T}(V, A, C, qq, b, g, N, M, L)
        return solveOOQP(Q; settings=settings)
    end

    mu1 = mu
    if mu == Inf   #HVEP (Highest Variance Efficient Portfolio)
        #find the Highest mu
        x, status = mpcLP(q, A, b, C, g; settings=settings, min=false)
        if status == 0
            error("HVEP: infeasible")
        elseif status < 0
            error("HVEP: not converged")
        end
        mu1 = x.x' * q
    end
    #@ given mu1
    qq = zeros(T, N)
    Aq = [A; q']
    bq = [b; mu1]
    M += 1
    Q = OOQP{T}(V, Aq, C, qq, bq, g, N, M, L)
    return solveOOQP(Q; settings=settings)
end
