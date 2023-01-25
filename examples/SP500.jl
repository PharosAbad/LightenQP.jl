#SP500 dataset, lower bound d = 0, upper bound  u =3/32=9.375%

using LightenQP

using TranscodingStreams, CodecXz, Serialization, Downloads

#download the online data
xzFile = Downloads.download("https://github.com/PharosAbad/PharosAbad.github.io/raw/master/files/sp500.jls.xz")
io = open(xzFile)
io = TranscodingStream(XzDecompressor(), io)
E = deserialize(io)
V = deserialize(io)
close(io)

N = length(E)
u = fill(3 / 32, N)

Q = OOQP(V, zeros(N), u)
ts = @elapsed x, status = solveOOQP(Q)
println("--- LightenQP Method @LVEP (Lowest Variance Efficient Portfolio, also called GMVP, Global Minimum Variance Portfolio):  ", ts, "  seconds")

Q1 = OOQP(V, -E, u)
ts = @elapsed x1, status1 = solveOOQP(Q1)
println("--- LightenQP Method @L=1:  ", ts, "  seconds")


