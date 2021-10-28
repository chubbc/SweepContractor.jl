using SweepContractor

L=6; d=2

LTN = LabelledTensorNetwork{Tuple{Int,Int}}()
for i∈1:L, j∈1:L
    adj=[
        (mod1(i-1,L),mod1(j,L)),
        (mod1(i+1,L),mod1(j,L)),
        (mod1(i,L),mod1(j-1,L)),
        (mod1(i,L),mod1(j+1,L))
    ]
    LTN[i,j] = Tensor(adj, randn(d,d,d,d), i+0.01*rand(), j+0.01*rand())
end

println("example_2d_periodic")
for χ ∈ d.^(4:9)
    sweep = sweep_contract(LTN,χ, 2*χ)
    println("χ=$χ:\t",ldexp(sweep...))
end
