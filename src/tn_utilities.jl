# Total ordering on tensors which defines the sweep direction
const TensorOrder = Base.By(λ->(λ.y,λ.x))
isless(A::Tensor, B::Tensor) = lt(TensorOrder, A, B)
sortperm(TN::TensorNetwork) = Base.sortperm(TN, order=TensorOrder)
issorted(TN::TensorNetwork) = Base.issorted(TN, order=TensorOrder)

# Sorts the tensor network according to TensorOrder and updates the adjacency labels
function sort!(TN::TensorNetwork)
    issorted(TN) && return TN
    σ = sortperm(TN)
    permute!(TN, σ)
    τ = invperm(σ)
    for t ∈ TN
        t.adj = τ[t.adj]
    end
    return TN
end

struct edge
    lo::Int
    hi::Int
end

# Returns *twice* the perpendicular distance, as the extra division is unnecessary
perpdist(TN::TensorNetwork, p::Int, e::edge) =
    (TN[e.hi].x-TN[e.lo].x)*(TN[e.hi].y-2*TN[p].y+TN[e.lo].y)-
    (TN[e.hi].y-TN[e.lo].y)*(TN[e.hi].x-2*TN[p].x+TN[e.lo].x)

# Is a point p to the 'left' of an edge e?
toleft(TN::TensorNetwork, p::Int, e::edge) = perpdist(TN,p,e)<0

# Do two edges non-trivially intersect?
intersect(TN::TensorNetwork, a::edge, b::edge) =
    a.lo!=b.lo && a.lo!=b.hi && a.hi!=b.lo && a.hi!=b.hi &&
    perpdist(TN,a.lo,b)*perpdist(TN,a.hi,b)<0 && perpdist(TN,b.lo,a)*perpdist(TN,b.hi,a)<0

# A (partial) order on edges relevant to a Bentley-Ottmann sweep
function edgelt(TN::TensorNetwork, a::edge, b::edge)
    TN[a.lo]>TN[b.lo] && return toleft(TN,a.lo,b)
    TN[a.lo]<TN[b.lo] && return !toleft(TN,b.lo,a)
    θa = atan(TN[a.hi].x-TN[a.lo].x, TN[a.hi].y-TN[a.lo].y)
    θb = atan(TN[b.hi].x-TN[b.lo].x, TN[b.hi].y-TN[b.lo].y)
    return θa<θb
end

# Adds a dim=1 edge to the network
function addedge!(TN::TensorNetwork, u::Int, v::Int)
    u∈TN[v].adj && return
    push!(TN[u].adj, v)
    TN[u].arr = reshape(TN[u].arr, (size(TN[u].arr)...,1))
    push!(TN[v].adj, u)
    TN[v].arr = reshape(TN[v].arr, (size(TN[v].arr)...,1))
    return nothing
end

# Given that an intersection exists, what is its position?
function intersectpos(TN::TensorNetwork, a::edge, b::edge)
    Aa = TN[a.hi].y - TN[a.lo].y
    Ba = TN[a.hi].x - TN[a.lo].x
    Ca = TN[a.hi].y*TN[a.lo].x - TN[a.lo].y*TN[a.hi].x

    Ab = TN[b.hi].y - TN[b.lo].y
    Bb = TN[b.hi].x - TN[b.lo].x
    Cb = TN[b.hi].y*TN[b.lo].x - TN[b.lo].y*TN[b.hi].x

    return (Bb*Ca-Ba*Cb, Ab*Ca-Aa*Cb) ./ (Aa*Bb-Ab*Ba)
end

# Add dim=1 edges corresponding to the left hull of the network. Together with connect! this
# allows the imposition of sweep-connectivity on an arbitrary network
function hull!(TN::TensorNetwork)
    sort!(TN)
    # Left hull
    H = Int[]
    for h ∈ eachindex(TN)
        push!(H, h)
        while length(H)>2 && toleft(TN, H[end], edge(H[end-2], H[end-1]))
            H = H[[1:end-2;end]]
        end
    end
    for i ∈ 2:length(H)
        addedge!(TN, H[i-1], H[i])
    end
    return TN
end

# Adds swap tensors as appropriate to planarise an arbitrary network. Can throw
# InvalidTNError if the graph is not appropriately connected. This is based off a
# Bentley-Ottmann Algorithm approach, such as can be seen in https://youtu.be/be5y0BVQ5kg
function planarise!(TN::TensorNetwork)
    # Check the intersection of neighbouring edges, labelled r-1 and r, in the AVLTree B. If
    # an intersection is found, also updates the network, Q and B
    function checkintersection(r::Int)
        # If there is no intersection then return
        1<r<=length(B) || return nothing
        L = B[r-1]
        R = B[r]
        intersect(TN, L, R) || return nothing
        # Otherwise we need to include a new swap tensor
        len += 1;
        # As we are constantly adding to TN, resize it only infrequently to avoid overheads
        # from memory management
        len>length(TN) && resize!(TN,2*length(TN))
        bL = size(TN[L.lo].arr, findfirst(isequal(L.hi), TN[L.lo].adj))
        bR = size(TN[R.lo].arr, findfirst(isequal(R.hi), TN[R.lo].adj))
        TN[len] = Tensor{Int}(
            [L.lo, R.lo, L.hi, R.hi],
            reshape(Matrix{Float64}(LinearAlgebra.I,bL*bR,bL*bR), (bL,bR,bL,bR)),
            intersectpos(TN, L, R)...
        )
        # Add the new tensor to the queue of points that need to be analysed
        DataStructures.enqueue!(Q, len, (TN[len].y,TN[len].x));
        # Disconnect all of the relevant tensors, connecting them through the swap
        replace!(TN[L.lo].adj, L.hi=>len)
        replace!(TN[L.hi].adj, L.lo=>len)
        replace!(TN[R.lo].adj, R.hi=>len)
        replace!(TN[R.hi].adj, R.lo=>len)
        # Remove the old edges from B and include the new ones
        delete!(B, L)
        delete!(B, R)
        push!(B, edge(L.lo, len))
        push!(B, edge(R.lo, len))
    end

    len = length(TN)

    # Self-balancing tree containing the edges cut by the current sweepline
    B::AVLTree{edge} = AVLTree{edge}(Lt((a,b)->edgelt(TN,a,b)))
    # Queue of points above the sweepline yet to be analysed
    Q = PriorityQueue{Int64,Tuple{Float64,Float64}}()
    # Populate the queue
    for l ∈ eachindex(TN)
        DataStructures.enqueue!(Q, l, (TN[l].y,TN[l].x))
    end
    # Loop over points in the queue until empty
    while !isempty(Q)
        q = DataStructures.dequeue!(Q)
        # Loop over all 'downward' edges of q
        for n ∈ TN[q].adj
            TN[n]>TN[q] && continue
            # For each such edge, remove it, and then check whether the now-neighbouring
            # edges in B are intersecting. If the tree throws an error this is because of a
            # failure of the partial ordering on edges, suggesting overlapping edges
            e = edge(n, q)
            try
                r = sorted_rank(B, e)
                delete!(B, e)
                checkintersection(r)
            catch _
                throw(InvalidTNError("overlapping edges"))
            end
        end
        # Loop over all 'upward' edges of q
        for n ∈ TN[q].adj
            TN[n]<TN[q] && continue
            e = edge(q, n)
            # Push the new edge into the tree, once again checking for any intersections
            # with its neighbours therein
            push!(B, e)
            try
                r = sorted_rank(B, e)
                checkintersection(r)
                checkintersection(r+1)
            catch _
                throw(InvalidTNError("overlapping edges"))
            end
        end
    end
    # Truncate off any unnecessary memory that was included to avoid memory overheads
    resize!(TN, len)
    return TN
end

# Sweep-connects a tensor network. This assumes that all of the edges along the left hull
# are included and that the graph is planar, as are imposed by !hull and !connect
# respectively. As with planarise! this algorithm is also based on a Bentley-Ottmann-style
# sweepline algorithm. sweep. Instead of looking for intersections, this looks for tensors
# with no 'downard' edges (other than the bottommost tensor), and connects them to a tensor
# at a Steiner point immediate to the left.
function connect!(TN::TensorNetwork)
    len = length(TN)

    sort!(TN)

    # Standard Bentley-Ottmann setup as in planarise!
    B::AVLTree{edge} = AVLTree{edge}(Lt((a,b)->edgelt(TN,a,b)))
    Q = PriorityQueue{Int64,Tuple{Float64,Float64}}()
    for l ∈ eachindex(TN)
        DataStructures.enqueue!(Q, l, (TN[l].y,TN[l].x));
    end
    while !isempty(Q)
        q = DataStructures.dequeue!(Q)
        # Consider only points where are not sweep-connected
        steiner = q>1
        for n ∈ TN[q].adj
            TN[n]>TN[q] && continue
            steiner = false
            delete!(B, edge(n, q))
        end
        if steiner
            # Find the edge L which is to the immediate left of the point q
            if isempty(TN[q].adj)
                e = edge(q, q)
                push!(B, e)
                r = sorted_rank(B, e)
                L = B[r-1]
                delete!(B,e)
            else
                e = edge(q, first(TN[q].adj))
                push!(B, e)
                r = sorted_rank(B, e)
                L = B[r-1]
            end
            if TN[L.lo].y == TN[q].y
                # If there is a tensor at the steiner point, directly connect them
                addedge!(FTN, L.lo, q)
            else
                # Otherwise include a new point
                Al = TN[L.hi].y - TN[L.lo].y
                Bl = TN[L.hi].x - TN[L.lo].x
                Cl = TN[L.lo].x*TN[L.hi].y - TN[L.lo].y*TN[L.hi].x
                x = (Cl + Bl * TN[q].y) / Al
                len += 1;
                len>length(TN) && resize!(TN, 2*length(TN))

                b = size(TN[L.lo].arr, findfirst(isequal(L.hi), TN[L.lo].adj))
                TN[len]=Tensor{Int}(
                    [L.lo, L.hi],
                    Matrix{Float64}(LinearAlgebra.I, b, b),
                    x, TN[q].y
                )
                addedge!(TN, len, q)
                replace!(TN[L.lo].adj, L.hi=>len)
                replace!(TN[L.hi].adj, L.lo=>len)
                delete!(B, L)
                push!(B, edge(len,L.hi))
            end
        end
        for n ∈ TN[q].adj
            TN[n]<TN[q] && continue
            push!(B, edge(q, n))
        end
    end
    resize!(TN, len)
    return TN
end
