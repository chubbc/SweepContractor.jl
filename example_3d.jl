using Pkg; Pkg.add(url="https://github.com/chubbc/SweepContractor.jl")
using SweepContractor

L=3; d=2

LTN = LabelledTensorNetwork{Tuple{Int,Int,Int}}()
for i∈1:L, j∈1:L, k∈1:L
    adj=Tuple{Int,Int,Int}[];
    i>1 && push!(adj,(i-1,j,k))
    i<L && push!(adj,(i+1,j,k))
    j>1 && push!(adj,(i,j-1,k))
    j<L && push!(adj,(i,j+1,k))
    k>1 && push!(adj,(i,j,k-1))
    k<L && push!(adj,(i,j,k+1))
    LTN[i,j,k] = Tensor(
        adj,
        randn(d*ones(Int,length(adj))...),
        i+0.01*randn(),
        j+0.01*randn()
    )
end

println("example_3d")
for χ∈d.^(4:9)
    sweep=sweep_contract(LTN, χ, 2*χ)
    println("χ=$χ:\t",ldexp(sweep...))
end
