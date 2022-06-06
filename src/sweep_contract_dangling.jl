"""
    sweep_contract_dangling(LTN::LabelledTensorNetwork, χ, τ;
        fast=false, valid=false, planar=false, connected=false, report=false)
    sweep_contract_dangling(TN::TensorNetwork, χ, τ;
        fast=false, valid=false, planar=false, connected=false, report=false)

A modified version of `sweep_contract` which can be used to evaluate open tensor networks
which possess dangling legs. This is done by halting the sweep procedure before the final
tensor is contracted and returning the MPS approximating the network so far. To do this a
dummy tensor must be present in the network which is above the rest of the network, i.e. has
a higher y coordinate.

Similar to sweep_contract underflow/overflow issues are avoided by return the MPS in a
floating point format. The output is returned as a tuple `(mps::MPS, i::Int64)` where `mps`
has a 2-norm in [1,2), representing the MPS `mps*2^i`.

For other details see the documentation of `sweep_contract`.
"""
sweep_contract_dangling(LTN::LabelledTensorNetwork, χ::Int, τ::Int;
    fast=false, valid=false, planar=false, connected=false, report=false) =
sweep_contract_dangling!(deepcopy(LTN), χ, τ;
    fast=fast, planar=planar, connected=connected, report=report)

sweep_contract_dangling(TN::TensorNetwork, χ::Int, τ::Int;
    fast=false, valid=false, planar=false, connected=false, report=false) =
sweep_contract_dangling!(deepcopy(TN), χ, τ;
    fast=fast, planar=planar, connected=connected, report=report)

"""
    sweep_contract_dangling!(LTN::LabelledTensorNetwork, χ, τ;
        fast=false, valid=false, planar=false, connected=false, report=false)
    sweep_contract_dangling!(TN::TensorNetwork, χ, τ;
        fast=false, valid=false, planar=false, connected=false, report=false)

The mutating form of `sweep_contract_dangling`.
"""
sweep_contract_dangling!(LTN::LabelledTensorNetwork, χ::Int, τ::Int;
    fast=false, valid=false, planar=false, connected=false, report=false) =
sweep_contract_dangling!(delabel(LTN), χ, τ;
        fast=fast, planar=planar, connected=connected, report=report)

function sweep_contract_dangling!(TN::TensorNetwork, χ::Int, τ::Int;
    fast=false,valid=false,planar=false,connected=false,report=false)::Tuple{MPS,Int}
    if !fast
        valid || checkvalid(TN)
        planar || planarise!(TN)
        connected || connect!(hull!(TN))
    end

    sort!(TN)

    resexp = 0
    count = 0

    MPS_t = [ones(1,1,1)]
    MPS_i = Int[]

    for (i,t) ∈ enumerate(TN)
        if i==length(TN)
            break;
        end
        ind_up = Int[]
        ind_do = Int[]
        for n ∈ t.adj
            if TN[n]>t
                push!(ind_up, n)
            elseif TN[n]<t
                push!(ind_do, n)
            else
                throw(InvalidTNError("Overlapping tensors"))
            end
        end
        sort!(ind_up, by=λ->atan(TN[λ].x-t.x,TN[λ].y-t.y))
        sort!(ind_do, by=λ->atan(TN[λ].x-t.x,t.y-TN[λ].y))
        σ = permutebetween(t.adj, [ind_do; ind_up])
        t.arr = permutedims(t.arr, σ)
        s = size(t.arr)
        t.arr = reshape(t.arr,(prod(s[1:length(ind_do)]),s[length(ind_do)+1:end]...))

        if isempty(MPS_i)
            MPS_t = splitMPStensor(MPS_t[1][1]*reshape(t.arr,(size(t.arr)...,1)))
            MPS_i = ind_up
        else
            lo = findfirst(isequal(i), MPS_i)
            hi = findlast(isequal(i), MPS_i)

            isnothing(lo) && throw(InvalidTNError("Disconnected TN"))

            X::Array{Float64} = MPS_t[lo]
            for j ∈ lo+1:hi
                finalsize = (size(X,1),size(X,2)*size(MPS_t[j],2),size(MPS_t[j],3))
                X = reshape(X,(size(X,1)*size(X,2),size(X,3)))*
                    reshape(MPS_t[j],(size(MPS_t[j],1),size(MPS_t[j],2)*size(MPS_t[j],3)))
                X = reshape(X,finalsize)
            end
            X = permutedims(X,[1,3,2])
            M = reshape(t.arr,(size(t.arr,1),prod(size(t.arr)[2:end])))
            X = reshape(
                reshape(X,(size(X,1)*size(X,2),size(X,3)))*M,
                (size(X,1),size(X,2),size(M,2))
            )
            X = permutedims(X,[1,3,2])
            X = reshape(X,(size(X,1),size(t.arr)[2:end]...,size(X,3)))

            MPS_i = [MPS_i[1:lo-1]; ind_up; MPS_i[hi+1:end]]
            if ndims(X)!=2
                MPS_t = [MPS_t[1:lo-1]; splitMPStensor(X); MPS_t[hi+1:end]]
            elseif isempty(MPS_i)
                MPS_t=[reshape([X[1]],(1,1,1))]
            elseif lo>1
                s = size(MPS_t[lo-1])
                MPS_t[lo-1] = reshape(
                    reshape(MPS_t[lo-1],(s[1]*s[2],s[3]))*X,
                    (s[1],s[2],size(X,2))
                )
                MPS_t = [MPS_t[1:lo-1]; MPS_t[hi+1:end]]
            else
                s = size(MPS_t[hi+1])
                MPS_t[hi+1] = reshape(
                    X*reshape(MPS_t[hi+1],(s[1],s[2]*s[3])),
                    (size(X,1),s[2],s[3])
                )
                MPS_t = [MPS_t[1:lo-1]; MPS_t[hi+1:end]]
            end

            if any(size.(MPS_t,3).>τ)
                count += 1
                truncMPS!(MPS_t, χ)
                h = Int(floor(log2(LinearAlgebra.norm(MPS_t[1]))))
                resexp += h
                MPS_t[1] /= exp2(h)
            end
        end
    end

    report && println("Number of truncations: $count")

    truncMPS!(MPS_t, χ)
    h = Int(floor(log2(LinearAlgebra.norm(MPS_t[1]))))
    resexp += h
    MPS_t[1] /= exp2(h)

    return (MPS_t,resexp);
end
