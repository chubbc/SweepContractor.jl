module SweepContractor
    import Base: show, print, summary
    export Tensor, TensorNetwork, LabelledTensorNetwork
    # Define the types
    include("tensor_network.jl")

    import Base: By, Lt, Ordering, Forward, ForwardOrdering, ReverseOrdering, lt
    include("sorted_data_structs/data_structures.jl")
    include("sorted_data_structs/balanced_tree.jl")
    include("sorted_data_structs/avl_tree.jl")
    include("sorted_data_structs/sorted_dict.jl")
    include("sorted_data_structs/priorityqueue.jl")
    import LinearAlgebra
    import Base: issorted, sortperm, sort!, isless, intersect
    # Utilities to make sure networks are suitably formatted for sweep contraction
    include("tn_utilities.jl")

    export sweep_contract!, sweep_contract
    # The sweep contraction algorithm
    include("sweep_contract.jl")
end
