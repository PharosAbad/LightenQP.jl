#Speed and Accuracy: Quadratic pragramming
#compare: LightenQP, OSQP, Clarabel


using EfficientFrontier, LinearAlgebra
using TranscodingStreams, CodecXz, Serialization, Downloads

if length(filter((x) -> x == :uOSQP, names(Main, imported=true))) == 0
    include("./uOSQP.jl")
    using .uOSQP
end

if length(filter((x) -> x == :uClarabel, names(Main, imported=true))) == 0
    include("./uClarabel.jl")
    using .uClarabel
end

function testData(ds::Symbol)
    if ds == :Ungil
        E, V = EfficientFrontier.EVdata(:Ungil, false)
        A = [1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
            1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
        b = [1.0; 0.25]
        G = [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0]
        G[1, :] = -G[1, :]
        g = [-0.3; 0.6]
        d = vec([-0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.1 -0.1 -0.1 -0.1])
        u = vec([0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.1 0.1 0.1 0.3 0.3 0.3 0.3])

        P = Problem(E, V, u, d, G, g, A, b)
    elseif ds == :SP500
        xzFile = "/tmp/sp500.jls.xz"
        if !isfile(xzFile)
            Downloads.download("https://github.com/PharosAbad/PharosAbad.github.io/raw/master/files/sp500.jls.xz", xzFile)
        end
        io = open(xzFile)
        io = TranscodingStream(XzDecompressor(), io)
        E = deserialize(io)
        V = deserialize(io)
        close(io)
        N = length(E)
        u = fill(3 / 32, N)

        pd = true
        if pd
            N = 263
            ip = 1:N
            V = V[ip, ip]
            E = E[ip]
            u = fill(3 / 32, N)
        end
        P = Problem(E, V, u)
    else
        error("Unknown dataset")
    end
    return P
end


function SpeedAccuracy(aEF, P, QPsolver, M=16)
    V = P.V
    N = length(aEF.mu) - 1
    T = zeros(N, M)     #time used, for speed
    A = zeros(N, M)     #Accuracy
    O = zeros(N, M)     #Objective function
    S = trues(N, M)     #Solution status
    for k in 1:N
        for m = 1:M
            mu = ((M + 1 - m) * aEF.mu[k] + (m - 1) * aEF.mu[k+1]) / M
            z = ePortfolio(aEF, mu)
            if QPsolver == :LightenQP
                ts = @elapsed y, status = fPortfolio(P, mu; check=false)
                st = status > 0
            elseif QPsolver == :OSQP
                ts = @elapsed y = OpSpQP(P, mu)
                st = y.info.status_val == 1
            elseif QPsolver == :Clarabel
                ts = @elapsed y = ClarabelQP(P, mu)
                st = Int(y.status) == 1
            else
                error("Unknown QP solver")
            end
            S[k, m] = st
            T[k, m] = ts
            A[k, m] = norm(y.x - z, Inf)
            O[k, m] = sqrt(y.x' * V * y.x) - sqrt(z' * V * z)
        end
    end

    return T, A, O, S
end


function cmpSA(ds::Symbol)
    P = testData(ds)
    println("--- Starting EfficientFrontier ---")
    t0 = time()
    ts = @elapsed aCL = EfficientFrontier.ECL(P)
    aEF = eFrontier(aCL, P)
    t1 = time()
    println("1st run, EfficientFrontier:  ", t1 - t0, "  seconds", "\n   aCL:  ", ts, "  seconds")
    t0 = time()
    ts = @elapsed aCL = EfficientFrontier.ECL(P)
    aEF = eFrontier(aCL, P)
    t1 = time()
    println("2nd run, EfficientFrontier:  ", t1 - t0, "  seconds", "\n   aCL:  ", ts, "  seconds")

    QPsolver = :LightenQP
    Tl, Al, Ol, Sl = SpeedAccuracy(aEF, P, QPsolver)
    QPsolver = :OSQP
    To, Ao, Oo, So = SpeedAccuracy(aEF, P, QPsolver)
    QPsolver = :Clarabel
    Tc, Ac, Oc, Sc = SpeedAccuracy(aEF, P, QPsolver)

    redirect_stdio(stdout="stdout.txt") do
        #status
        println("\n------- Solution status ------- LightenQP/OSQP/Clarabel")
        show(stdout, "text/plain", Sl); println("")
        show(stdout, "text/plain", So); println("")
        show(stdout, "text/plain", Sc); println("")
        #Accuracy
        println("\n------- Accuracy -------LightenQP/OSQP/Clarabel   ", round.([norm(Al, Inf), norm(Ao, Inf), norm(Ac, Inf)], sigdigits=3))
        show(stdout, "text/plain", round.(Al, sigdigits=3)); println("")
        show(stdout, "text/plain", round.(Ao, sigdigits=3)); println("")
        show(stdout, "text/plain", round.(Ac, sigdigits=3)); println("")
        #Speed
        println("\n--- Speed (time span, smaller for faster speed) ---LightenQP/OSQP/Clarabel   ", round.([norm(Tl, Inf), norm(To, Inf), norm(Tc, Inf)], sigdigits=3))
        show(stdout, "text/plain", round.(Tl, sigdigits=3)); println("")
        show(stdout, "text/plain", round.(To, sigdigits=3)); println("")
        show(stdout, "text/plain", round.(Tc, sigdigits=3)); println("")
        #Objective function
        println("\n--- Objective function value (diff in sd, not variance) ---LightenQP/OSQP/Clarabel   ", round.([norm(Ol, Inf), norm(Oo, Inf), norm(Oc, Inf)], sigdigits=3))
        show(stdout, "text/plain", round.(Ol, sigdigits=3)); println("")
        show(stdout, "text/plain", round.(Oo, sigdigits=3)); println("")
        show(stdout, "text/plain", round.(Oc, sigdigits=3)); println("")
    end
    #redirect_stdio()
    #return Sl, So, Sc, Al, Ao, Ac, Tl, To, Tc, Ol, Oo, Oc
    return nothing
end


#Sl, So, Sc, Al, Ao, Ac, Tl, To, Tc, Ol, Oo, Oc = cmpSA(:Ungil)
#cmpSA(:Ungil)
#Sl, So, Sc, Al, Ao, Ac, Tl, To, Tc, Ol, Oo, Oc = cmpSA(:SP500)
cmpSA(:SP500)

nothing