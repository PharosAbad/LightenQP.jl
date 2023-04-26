#SP500 dataset, lower bound d = 0, upper bound  u =3/32=9.375%
# User Guides: https://github.com/PharosAbad/LightenQP.jl/wiki/User-Guides#portfolio-selection-1

using LightenQP

using TranscodingStreams, CodecXz, Serialization, Downloads

function main()

    #download the online data
    xzFile = joinpath(tempdir(),"sp500.jls.xz")  #xzFile = "/tmp/sp500.jls.xz"
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

    #= Q = OOQP(V, zeros(N), u)
    ts = @elapsed x, status = solveOOQP(Q)
    println("--- LightenQP Method @LVEP (Lowest Variance Efficient Portfolio, also called GMVP, Global Minimum Variance Portfolio):  ", ts, "  seconds")

    Q1 = OOQP(V, -E, u)
    ts = @elapsed x1, status1 = solveOOQP(Q1)
    println("--- LightenQP Method @L=1:  ", ts, "  seconds")    =#

    #v1.0.1
    Q = OOQP(V, E, u)
    ts = @elapsed xH, statusH = fPortfolio(Q, Inf)       #HMFP (Highest Mean Frontier Portfolio)
    println("HMEP (Highest Mean Efficient Portfolio) --- fPortfolio @ max mu:  ", ts, "  seconds")
    ts = @elapsed xL, statusL = fPortfolio(Q, -Inf)      #LMFP (Lowest Mean Frontier Portfolio)
    println("LMEP (Lowest Mean Efficient Portfolio)  --- fPortfolio @ min mu:  ", ts, "  seconds")
    ts = @elapsed x0, status0 = fPortfolio(Q, 0.0)
    println("LMEP (Lowest Mean Efficient Portfolio, also called GMVP, Global Minimum Variance Portfolio) --- fPortfolio @ L=0:  ", ts, "  seconds")
    ts = @elapsed x2, status2 = fPortfolio(Q, 2.0)
    println(" --- fPortfolio @ L=2.0:  ", ts, "  seconds")
end

main()
nothing

