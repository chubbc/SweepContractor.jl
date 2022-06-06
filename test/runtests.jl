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
        global brute += LTN['A'].arr[ad,ab] * LTN['B'].arr[ab,bd,bc] * LTN['C'].arr[bc,cd] *
            LTN['D'].arr[ad,bd,cd]
    end

    sweep = sweep_contract(LTN, 2, 4; fast=true)
    
    (mps,dangexp) = sweep_contract_dangling(LTN, 2, 4; fast=true)
    dang=0.0
    for i1=1:2, i2=1:2, da=1:3, db=1:3, dc=1:3
        global dang+=mps[1][1,da,i1]*mps[2][i1,db,i2]*mps[3][i2,dc,1]*LTN['D'].arr[da,db,dc]
    end
    
    @test ldexp(sweep...) == brute
    @test ldexp(sweep...) ≈ ldexp(dang,dangexp)
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
