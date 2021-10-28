using SweepContractor
using Test

@testset "ABCD" begin
    LTN = LabelledTensorNetwork{Char}()
    LTN['A'] = Tensor(['D','B'], [i^2-2j for i=0:2, j=0:2], 0, 1)
    LTN['B'] = Tensor(['A','D','C'], [-3^i*j+k for i=0:2, j=0:2, k=0:2], 0, 0)
    LTN['C'] = Tensor(['B','D'], [j for i=0:2, j=0:2], 1, 0)
    LTN['D'] = Tensor(['A','B','C'], [i*j*k for i=0:2, j=0:2, k=0:2], 1, 1)

    brute = 0.0
    for ab=1:3, ad=1:3, bc=1:3, bd=1:3, cd=1:3
        brute += LTN['A'].arr[ad,ab] * LTN['B'].arr[ab,bd,bc] * LTN['C'].arr[bc,cd] *
            LTN['D'].arr[ad,bd,cd]
    end

    sweep = sweep_contract(LTN, 2, 4; fast=true)
    @test ldexp(sweep...) == brute
end


@testset "2d periodic" begin
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

    last_val = 0.0
    this_val = 0.0
    for χ ∈ d.^(4:9)
        sweep = sweep_contract(LTN,χ, 2*χ)
        last_val = this_val
        this_val = sweep[1]
    end
    @test abs(last_val - this_val) < 1e-10
end
