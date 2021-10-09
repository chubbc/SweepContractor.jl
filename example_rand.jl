using Pkg; Pkg.add(url="https://github.com/chubbc/SweepContractor.jl")
using SweepContractor

n=8; d=2

TN=TensorNetwork(undef,n)
for i=1:n
    TN[i]=Tensor(
        setdiff(1:n,i),
        randn(d*ones(Int,n-1)...),
        randn(),
        randn()
    )
end

println("example_rand")
for χ∈d.^(4:9)
    sweep=sweep_contract(TN,χ,2*χ;connected=false)
    println("χ=$χ:\t",ldexp(sweep...))
end
