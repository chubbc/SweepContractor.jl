const MPSTensor = Array{Float64,3}
const MPS = Vector{MPSTensor}

# Split a single multi-site tensor into single-site MPS tensors in the dumb way
function splitMPStensor(T::Array{Float64})
    v = MPS(undef,ndims(T)-2);
    (l,r) = (2,ndims(T)-1)
    s = collect(size(T))
    while r>l
        # Calculate the bond dimensions of sweeping tensor either way and minimise
        L = s[l-1]*s[l]
        R = s[r]*s[r+1]
        if L <= R
            v[l-1] = reshape(Matrix{Float64}(LinearAlgebra.I,L,L),(s[l-1],s[l],L));
            s[l] *= s[l-1]
            l += 1
        else
            v[r-1] = reshape(Matrix{Float64}(LinearAlgebra.I,R,R),(R,s[r],s[r+1]));
            s[r] *= s[r+1]
            r -= 1
        end
    end
    v[l-1] = reshape(T,(s[l-1],s[l],s[l+1]))
    return v
end

# Truncates down the bond dimension of an MPS, performing a (lossy) compression
function truncMPS!(M::MPS, χ::Int64)
    # Put the MPS in canonical form using the QR decomposition, sweeping left-to-right
    for i ∈ 1:length(M)-1
        X = reshape(M[i],(size(M[i],1)*size(M[i],2),size(M[i],3)))
        q,r = LinearAlgebra.qr(X)
        if size(r,1)==size(r,2)
            M[i] = reshape(Matrix(q),size(M[i]))
            LinearAlgebra.lmul!(LinearAlgebra.UpperTriangular(r),
                reshape(M[i+1],(size(M[i+1],1),size(M[i+1],2)*size(M[i+1],3))))
        else
            M[i] = reshape(Matrix(q),(size(M[i],1),size(M[i],2),size(r,1)))
            M[i+1] = reshape(r*reshape(M[i+1], (size(M[i+1],1),
                size(M[i+1],2)*size(M[i+1],3))), (size(r,1),size(M[i+1],2),size(M[i+1],3)))
        end
    end
    # Perform the bond truncation using the SVD decomposition, sweeping right-to-left
    for i ∈ length(M):-1:2
        X = reshape(M[i],(size(M[i],1),size(M[i],2)*size(M[i],3)))
        # In some rare cases the default svd can fail to converge
        try
            F = LinearAlgebra.svd!(X);
            (u,s,v) = (F.U,F.S,F.V)
        catch _
            F = LinearAlgebra.svd!(X; alg=LinearAlgebra.QRIteration())
            (u,s,v) = (F.U,F.S,F.V)
        end
        b = min(length(s),χ)
        u = u[:,1:b]
        s = s[1:b]
        v = v[:,1:b]'
        M[i] = reshape(v,(b,size(M[i],2),size(M[i],3)));
        X = reshape(M[i-1],(size(M[i-1],1)*size(M[i-1],2),size(M[i-1],3)))*u
        LinearAlgebra.rmul!(X,LinearAlgebra.Diagonal(s));
        M[i-1] = reshape(X,(size(M[i-1],1),size(M[i-1],2),b));
    end
    return M
end

# Find the permutation transformation between two vectors
function permutebetween(from, to)
    σf = sortperm(from)
    σt = sortperm(to)
    arr = Vector{Int}(undef,length(from))
    for i ∈ eachindex(from)
        arr[σt[i]]=σf[i]
    end
    return arr
end

"""
    sweep_contract(LTN::LabelledTensorNetwork, χ, τ;
        fast=false, valid=false, planar=false, connected=false, report=false)
    sweep_contract(TN::TensorNetwork, χ, τ;
        fast=false, valid=false, planar=false, connected=false, report=false)

Returns the contraction of the `TensorNetwork TN`, or the `LabelledTensorNetwork LTN` using
the sweepline contraction algorithm of `arXiv:2101.04125`. The MPS is truncated down to a
bond dimension of `χ` whenever any bond dimension exceeds `τ`.

By default the network is checked for validity, planarised, and sweep-connected, where
necessary. The keyword flags `valid`, `planar`, and `connected` can be used to skip these,
or the flag `fast` can be used to skip them all. If these flags are enabled then contraction
may fail on poorly formed networks.

To avoid underflow/overflow issues the contraction value of the network is returned as a
tuple `(f::Float64, i::Int64)` where `1≦f<2` or `f` is `0`, representing a value of `f*2^i`.
The function `ldexp` can be used to convert this back to a Float64.

`sweep_contract` is non-mutating and acts upon a deep copy of the network, where possible
use the more efficient mutating version `sweep_contract!`.
"""
sweep_contract(LTN::LabelledTensorNetwork, χ::Int, τ::Int;
    fast=false, valid=false, planar=false, connected=false, report=false) =
sweep_contract!(deepcopy(LTN), χ, τ;
    fast=fast, planar=planar, connected=connected, report=report)

sweep_contract(TN::TensorNetwork, χ::Int, τ::Int;
    fast=false, valid=false, planar=false, connected=false, report=false) =
sweep_contract!(deepcopy(TN), χ, τ;
    fast=fast, planar=planar, connected=connected, report=report)

"""
    sweep_contract!(LTN::LabelledTensorNetwork, χ, τ;
        fast=false, valid=false, planar=false, connected=false, report=false)
    sweep_contract!(TN::TensorNetwork, χ, τ;
        fast=false, valid=false, planar=false, connected=false, report=false)

The mutating form of `sweep_contract`.
"""
sweep_contract!(LTN::LabelledTensorNetwork, χ::Int, τ::Int;
    fast=false, valid=false, planar=false, connected=false, report=false) =
sweep_contract!(delabel(LTN), χ, τ;
        fast=fast, planar=planar, connected=connected, report=report)

function sweep_contract!(TN::TensorNetwork, χ::Int, τ::Int;
    fast=false,valid=false,planar=false,connected=false,report=false)::Tuple{Float64,Int}
    if !fast
        valid || checkvalid(TN)
        planar || planarise!(TN)
        connected || connect!(hull!(TN))
    end

    sort!(TN)

    N = length(TN)

    resexp = 0
    count = 0

    MPS_t = [ones(1,1,1)]
    MPS_i = Int[]

    for (i,t) ∈ enumerate(TN)
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
                if LinearAlgebra.norm(MPS_t[1])==0
                    return (0.0,typemin(Int))
                end
                h = Int(floor(log2(LinearAlgebra.norm(MPS_t[1]))))
                resexp += h
                MPS_t[1] /= exp2(h)
            end
        end
    end

    report && println("Number of truncations: $count")

    res = MPS_t[1][1];
    if res == 0.0
        return (0.0, typemin(Int64));
    end
    h = Int(floor(log2(abs(res))));
    return (res/exp2(h),resexp+h);
end
                                    
