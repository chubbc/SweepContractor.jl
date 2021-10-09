using Pkg; Pkg.add(url="https://github.com/chubbc/SweepContractor.jl")
using SweepContractor

L=16; d=2

LTN = LabelledTensorNetwork{Tuple{Int,Int}}()
for i∈1:L, j∈1:L
    adj=Tuple{Int,Int}[];
    i>1 && push!(adj,(i-1,j))
    j>1 && push!(adj,(i,j-1))
    i<L && push!(adj,(i+1,j))
    j<L && push!(adj,(i,j+1))
    LTN[i,j] = Tensor(adj, randn(d*ones(Int,length(adj))...), i, j)
end

println("example_2d_open")
for χ ∈ d.^(4:9)
    sweep = sweep_contract(LTN, χ, 2*χ; fast=true)
    println("χ=$χ:\t", ldexp(sweep...))
end
