using SweepContractor

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

println("example_ABCD")
println("brute = $brute")
println("sweep = $(ldexp(sweep...))")
