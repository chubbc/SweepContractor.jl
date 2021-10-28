heapleft(i::Integer) = 2i
heapright(i::Integer) = 2i + 1
heapparent(i::Integer) = div(i, 2)
#import OrderedCollections
abstract type AbstractSemiToken end
struct IntSemiToken <: AbstractSemiToken
    address::Int
end

using DataStructures: DataStructures, PriorityQueue
