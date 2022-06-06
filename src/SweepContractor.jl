module SweepContractor
    import Base: show, print, summary
    export Tensor, TensorNetwork, LabelledTensorNetwork
    # Define the types
    include("tensor_network.jl")

    import Base: By, Lt, Ordering, Forward, ForwardOrdering, ReverseOrdering, lt
    include("sorted_data_structs/data_structures.jl")
    include("sorted_data_structs/avl_tree.jl")
    import LinearAlgebra
    import Base: issorted, sortperm, sort!, isless, intersect
    # Utilities to make sure networks are suitably formatted for sweep contraction
    include("tn_utilities.jl")

    export sweep_contract!, sweep_contract
    export sweep_contract_dangling, sweep_contract_dangling!
    # The sweep contraction algorithm
    include("sweep_contract.jl")
    include("sweep_contract_dangling.jl")
end
