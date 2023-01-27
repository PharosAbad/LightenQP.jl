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

        x, status = fPortfolio(O::OOQP; settings, L::T=0.0)
        x, status = fPortfolio(O::OOQP, mu; settings, check=true)

find the minimum variance portfolio: See [`Portfolio Selection · LightenQP`](https://github.com/PharosAbad/LightenQP.jl/wiki/User-Guides#portfolio-selection-1)

    L=-Inf          :FP(L=-Inf), LMFP (Lowest Mean Frontier Portfolio)
    L=+Inf          :FP(L=+Inf), HMFP (Highest Mean Frontier Portfolio) HVEP (Highest Variance Efficient Portfolio)
    L=L0            :FP(L=L0), the frontier (minimum variance) portfolio at L=L0. L=0, LVEP (Lowest Variance Efficient Portfolio, also called GMVP, Global Minimum Variance Portfolio)
    mu=-Inf         :FP(L=-Inf), LMFP (Lowest Mean Frontier Portfolio)
    mu=+Inf         :FP(L=+Inf), HMFP (Highest Mean Frontier Portfolio) == HVEP (Highest Variance Efficient Portfolio)
    mu=mu0          :FP(mu=mu0), the frontier (minimum variance) portfolio at mu=mu0

if `check=false`, we do not check if mu is feasible or not (between lowest and highest mean)

See also [`OOQP`](@ref), [`solveOOQP`](@ref), [`Solution`](@ref), [`Settings`](@ref)
"""
function fPortfolio(O::OOQP{T}, mu::T; settings=Settings{T}(), check=true) where {T}
    #FP(mu=mu)
    (; V, A, C, q, b, g, N, M, L) = O

    mu1 = mu
    if check
        #make sure mu is feasible, otherwise, change mu to be the highest or lowest
        #HMFP (Highest Mean Frontier Portfolio)        
        xH, status = mpcLP(q, A, b, C, g; settings=settings, min=false)  #find the Highest mu
        if status == 0
            error("mu for Highest Mean Frontier Portfolio: infeasible")
        elseif status < 0
            error("mu for Highest Mean Frontier Portfolio: not converged")
        end
        muH = xH.x' * q
        if mu1 > muH
            mu1 = muH
            if isfinite(mu)
                @warn "mu is higher than the highest mu_H, compute at mu_H"
            end
        else
            #LMFP (Lowest Mean Frontier Portfolio)
            xL, status = mpcLP(q, A, b, C, g; settings=settings) #find the Lowest mu
            if status == 0
                error("mu for Lowest Mean Frontier Portfolio: infeasible")
            elseif status < 0
                error("mu for Lowest Mean Frontier Portfolio: not converged")
            end
            muL = xL.x' * q
            if mu1 < muL
                mu1 = muL
                if isfinite(mu)
                    @warn "mu is lower than the lowest mu_L, compute at mu_L"
                end
            end
        end
    end
    #@ given mu1
    qq = zeros(T, N)
    Aq = [A; q']
    bq = [b; mu1]
    M += 1
    Q = OOQP{T}(V, Aq, C, qq, bq, g, N, M, L)
    return solveOOQP(Q; settings=settings)
end


function fPortfolio0(O::OOQP{T}, mu; settings=Settings{T}()) where {T}
    #FP(mu=mu)
    (; V, A, C, q, b, g, N, M, L) = O
    #@ given mu
    qq = zeros(T, N)
    Aq = [A; q']
    bq = [b; mu]
    M += 1
    Q = OOQP{T}(V, Aq, C, qq, bq, g, N, M, L)
    return solveOOQP(Q; settings=settings)
end


function fPortfolio(O::OOQP{T}; settings=Settings{T}(), L::T=0.0) where {T}
    #FP(L=L)
    (; V, A, C, q, b, g, N, M) = O
    if isfinite(L)  #@ given L
        qq = -L * q
        Q = OOQP{T}(V, A, C, qq, b, g, N, M, O.L)
        return solveOOQP(Q; settings=settings)
    end
    if L == -Inf  #LMFP (Lowest Mean Frontier Portfolio)
        x, status = mpcLP(q, A, b, C, g; settings=settings) #find the Lowest mu
        if status == 0
            error("mu for Lowest Mean Frontier Portfolio: infeasible")
        elseif status < 0
            error("mu for Lowest Mean Frontier Portfolio: not converged")
        end
    elseif L == Inf   #HMFP (Highest Mean Frontier Portfolio)
        x, status = mpcLP(q, A, b, C, g; settings=settings, min=false)  #find the Highest mu
        if status == 0
            error("mu for Highest Mean Frontier Portfolio: infeasible")
        elseif status < 0
            error("mu for Highest Mean Frontier Portfolio: not converged")
        end
    end
    mu = x.x' * q
    qq = zeros(T, N)
    Aq = [A; q']
    bq = [b; mu]
    M += 1
    Q = OOQP{T}(V, Aq, C, qq, bq, g, N, M, O.L)
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
