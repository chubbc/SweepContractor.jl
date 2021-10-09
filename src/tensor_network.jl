"""
    Tensor{T}

A tensor in a network with index labels of type T.

# Fields
- `adj::Vector{T}`: the labels of the adjacent tensors
- `arr::Array{Float64}`: the contents of the tensor
- `x::Float64`, `y::Float64`: position coordinates of the tensor

The order with which the neighbours are labelled in `adj` may be arbitrary, but must be
compatible with the labelling of the indices of `arr`.
"""
mutable struct Tensor{T}
    adj::Vector{T}
    arr::Array{Float64}
    x::Float64
    y::Float64

    # Native float constructors
    Tensor{T}(adj::Vector{T}, arr::Array{Float64}, x::Float64, y::Float64) where T =
        new{T}(adj, arr, x, y)
    Tensor(adj::Vector{T}, arr::Array{Float64}, x::Float64, y::Float64) where T =
        new{T}(adj, arr, x, y)

    # Promoted constructors
    Tensor{T}(adj::Vector{T}, arr::Array{<:Real}, x::Real, y::Real) where T =
        new{T}(adj, float(arr), float(x), float(y))
    Tensor(adj::Vector{T}, arr::Array{<:Real}, x::Real, y::Real) where T =
        new{T}(adj, float(arr), float(x), float(y))
end

"""
    TensorNetwork

A tensor network with edges labelled by consecutive integers. Alias for
`Vector{Tensor{Int}}`.
"""
const TensorNetwork = Vector{Tensor{Int}}
labels(TN::TensorNetwork) = keys(TN)
tensors(TN::TensorNetwork) = values(TN)

"""
    LabelledTensorNetwork{T}

A tensor network with tensor labels of type `T`. Alias for `Dict{T,Tensor{T}}`. Due to the
potential performacne overheads associated with dealing with arbitrarily labelled tensors
the alternative `TensorNetwork` type should be used in performance sensitive circumstances.
"""
const LabelledTensorNetwork = Dict{T,Tensor{T}} where T
labels(LTN::LabelledTensorNetwork) = keys(LTN)
tensors(LTN::LabelledTensorNetwork) = values(LTN)

# Convert LabelledTensorNetwork into TensorNetwork for easier manipulation
function delabel(LTN::LabelledTensorNetwork{T}) where T
    N = length(LTN)
    # Construct an (arbitrary) mapping from labels to indices
    delabelling = Dict{T,Int}()
    sizehint!(delabelling, N)
    for (i,l) ∈ enumerate(labels(LTN))
        delabelling[l] = i
    end
    # Construct a TensorNetwork with this labelling
    TN = TensorNetwork(undef,N)
    for (l,t) ∈ LTN
        TN[delabelling[l]] = Tensor{Int}(map(λ->delabelling[λ],t.adj), t.arr, t.x, t.y)
    end
    return TN
end

struct InvalidTNError <: Exception
    str::String
end

Base.showerror(io::IO, e::InvalidTNError) = print(io, "Invalid tensor network: ", e.str)

# Check the validity of a tensor network and the tensors therein. If invalid then
# InvalidTNError is thrown, and otherwise nothing is returned
function checkvalid(LTN::LabelledTensorNetwork)
    for l ∈ labels(LTN)
        # Check that the graph degree matches the order of the tensor
        ndims(LTN[l].arr)!=length(LTN[l].adj) && throw(InvalidTNError("graph degree != tensor order"))
        # Check for multi-edges
        !allunique(LTN[l].adj) && throw(InvalidTNError("multi-edge"))
        for (i,k) ∈ enumerate(LTN[l].adj)
            # Check for self-edges
            l==k && throw(InvalidTNError("self-edge"))
            l>k && continue
            # Find edge back
            j = findfirst(isequal(l),LTN[k].adj)
            isnothing(j) && throw(InvalidTNError("directed edge"))
            # Make sure tensor dimensions are consistent along edge
            size(LTN[l].arr,i)!=size(LTN[k].arr,j) && throw(InvalidTNError("dimension mismatch"))
        end
    end
    return nothing
end

function checkvalid(TN::TensorNetwork)
    for u ∈ eachindex(TN)
        ndims(TN[u].arr)!=length(TN[u].adj) && throw(InvalidTNError("graph degree != tensor order"))
        !allunique(TN[u].adj) && throw(InvalidTNError("multi-edge"))
        for (i,v) ∈ enumerate(TN[u].adj)
            u==v && throw(InvalidTNError("self-edge"))
            u>v && continue
            j = findfirst(isequal(u),TN[v].adj)
            isnothing(j) && throw(InvalidTNError("directed edge"))
            size(TN[u].arr,i)!=size(TN[v].arr,j) && throw(InvalidTNError("dimension mismatch"))
        end
    end
    return nothing
end
