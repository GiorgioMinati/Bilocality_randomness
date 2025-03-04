using LinearAlgebra


## Utils

function resum(a; dims=tuple())
    if isempty(dims)
        return sum(a)
    else
        s = sum(a, dims=dims)
        d = size(s)
        drop = tuple(filter(i->d[i]<=1, 1:length(d))...)
        return dropdims(s, dims=drop)
    end
end

function partial_trace(m::Matrix{T}, tensor_struct::Vector{Int}, trace_on::Vector{Int}) where T <: Number
    
    # Need to reverse the order to reshape a kron product
    reverse!(tensor_struct)    
    #trace_mask = map(i->(i in trace_on), 1:length(tensor_struct))
    #reverse!(trace_mask)
    trace_mask = map(i->(i in trace_on), length(tensor_struct):-1:1)
    
    shape = (tensor_struct..., tensor_struct...)
    arr = reshape(m, shape)
    
    traced_struct = filter(d->d>0, .!trace_mask .* tensor_struct)
    traced_shape = tuple(traced_struct..., traced_struct...)
    traced_arr = zeros(traced_shape)
    
    for trace_ids in Iterators.product(map(x -> x>0 ? (1:x) : 0, trace_mask .* tensor_struct)...)
        idx = [i>0 ? i : (1:d) for (i,d) in zip(trace_ids, tensor_struct)]
        traced_arr .+= arr[[idx; idx]...]
    end
    
    traced_mat_dim = prod(traced_struct)
    return reshape(traced_arr, (traced_mat_dim, traced_mat_dim))
end

function eye(d)
    return I+zeros((d,d))
end

# States and measurements
# ------------------------

# Definitions for some specific states

bell_base = [
    1 0  0  1
    1 0  0 -1
    0 1  1  0
    0 1 -1  0
    ]*1.0 .+ 0.0*im

σ = [
    [0 1; 1 0],
    [0 -im; im 0],
    [1 0; 0 -1],
]

function bell_state(id::Int; θ::Float64=π/4, φ::Float64=0.0)
    """Bell states with a bit of rotation"""
    θ = θ - ((id-1)%2)*π/2
    p = (id-1) ÷ 2
    ψ = [(1-p)*cos(θ), p*cos(θ), p*exp(im*φ)*sin(θ), (1-p)*exp(im*φ)*sin(θ)]
    return ψ * ψ'
end

function rotbell_state(id::Int; qb::Int=1, θ::Float64=π/4, φ::Float64=0.0)
    """Bell states with a bit of rotation θ, and rotated on one of the two parties by φ"""
    ψ = bell_base[id,:] .* [cos(θ), cos(θ), sin(θ), sin(θ)]
    if qb == 1
        ψ = kron(cos(φ)*eye(2) + im*sin(φ)*σ[2], eye(2)) * ψ
    else
        ψ = kron(eye(2), cos(φ)*eye(2) + im*sin(φ)*σ[2]) * ψ
    end
    return ψ * ψ'
end

function andtangled_state(;θ::Float64=π/5, φ::Float64=0.0)
    ψ = cos(θ)*[0 1 exp(im*φ) 0]/√2 + sin(θ)*[1 0 0 0]
    return ψ' * ψ
end

# Measurements

function state2proj(ρ, outcome::Int)
    if outcome > 2 || outcome < 1
        throw("Invalid outcome value for a projector")
    end
    return outcome == 1 ? ρ : I-ρ
end

function bell_proj(id::Int, outcome::Int; θ::Float64=π/4)
    """Projection on Bell states with a bit of rotation"""
    return state2proj(bell_state(id, θ=θ), outcome)
end

function cyclic_obs(E::Matrix{T}, party::Int; num_parties=3::Int) where T <: Number
    """Generates the measurement for `party` in a n-partite cyclic scenario.
    E should be a d²×d² matrix (for some integer d) so that the subspaces for each party 
    are assumed to be equal to d.
    """
    d = Int(√(size(E)[1]))
    E_tensor = reshape(
        kron(E, eye(d^(2*(num_parties-1)))),
            tuple((d for _=1:num_parties*4)...))
    
    n = collect(1:num_parties*2)
    s = 2*party - 1
    dim_shift = [n[s+1:end]; n[1:s]]
    perm_E_tensor = permutedims(E_tensor, [dim_shift; dim_shift .+ (num_parties*2)])
    En = reshape(perm_E_tensor, (d^(num_parties*2), d^(num_parties*2)))
    return En
end

function cyclic_obs(Es::Vector{Matrix{T}}) where T <: Number
    """Generates the measurements for all parties in a n-partite cyclic scenario.
    E should be a vector of d²×d² matrices (for some integer d) so that the subspaces for each party 
    are assumed to be equal to d.
    """
    d = size(Es[1])
    for E in Es[2:end]
        if size(E) != d
            throw("Subspaces for parties are assumed to be of the same dimension")
        end
    end
    N = length(Es)
    return prod([cyclic_obs(Es[k], k, num_parties = N) for k=1:N])
end

# Elegant Joint Measurements
function EJM4(;η=0.0)
    
    e1 = 1/√8 * [-1-im 0 -2im -1+im]
    e2 = 1/√8 * [1-im 2im 0 1+im]
    e3 = 1/√8 * [-1+im 2im 0 -1-1im]
    e4 = 1/√8 * [1+im 0 -2im 1-im] 
    
    
    E1 = (1-η) * (e1' * e1) + (η/4) * eye(4)
    E2 = (1-η) * (e2' * e2) + (η/4) * eye(4)
    E3 = (1-η) * (e3' * e3) + (η/4) * eye(4)
    E4 = (1-η) * (e4' * e4) + (η/4) * eye(4)
    return [E1, E2, E3, E4]
end

function EJM2(;η=0.0, cg=2)
    
    e1 = 1/√8 * [-1-im 0 -2im -1+im]
    e2 = 1/√8 * [1-im 2im 0 1+im]
    e3 = 1/√8 * [-1+im 2im 0 -1-1im]
    e4 = 1/√8 * [1+im 0 -2im 1-im] 
    
    E1 = (1-η) * (e1' * e1) + (η/4) * eye(4)
    E2 = (1-η) * (e2' * e2) + (η/4) * eye(4)
    E3 = (1-η) * (e3' * e3) + (η/4) * eye(4)
    E4 = (1-η) * (e4' * e4) + (η/4) * eye(4)
    
    if cg == 2
        return [E1 + E2, E3 + E4]
    elseif cg == 3
        return [E1 + E3, E2 + E4]
    elseif cg == 4 
        return [E1 + E4, E2 + E3]
    else
        @error "Parameter `cg` can only be 2,3 or 4."
    end
end

# Build distributions
# -------------------

function split_states(ρs, intervention; dim=2)
    """Split the states `ρs[i]` with i ∈ `intervention` in the vector `ρs`
    and return the global state.
    """
    split_ρs = []
    for (i,ρ) in enumerate(ρs)
        if i in intervention
            split_ρ = kron(partial_trace(ρs[i], [dim, dim], [1, ]),
                           partial_trace(ρs[i], [dim, dim], [2, ]))
            push!(split_ρs, split_ρ)
        else
            push!(split_ρs, ρ)
        end
    end
    return kron(split_ρs...)
end

function p_cycle(
        ρs::Vector{Matrix{T}}, 
        Es::Vector{Vector{Matrix{T}}}; 
        intervention = nothing
    ) where T <: Number    
    
    """Generate observational and interventional distribution for the n-cycle scenario.
    Interventions are specified by tuples of sources to be cut. For example (1,)
    means an intervention cutting the source 1 between AB (cyclic scenario A-1-B-2-C-3-..).
    For example in the triangle the order would be C-β-A-γ-B-α-C
    If a list of intervention is specified all are generated.
    """
    
    if isnothing(intervention)
        full_states = [kron(ρs...),]
    elseif typeof(intervention) <: Tuple{Vararg{Int}}
        full_states = [kron(ρs...), split_states(ρs, intervention),]
    elseif typeof(intervention) <: Tuple{Vararg{Tuple{Vararg{Int}}}}
        full_states = [kron(ρs...); [split_states(ρs, i) for i in intervention]]
    else
        throw("Invalid intervention format")        
    end

    ps = []
    num_outs = (length(e) for e in Es)
    for ρ in full_states
        p = zeros(num_outs...)
        for outs in Iterators.product((1:n for n in num_outs)...)
            ABC = cyclic_obs([Es[i][o] for (i,o) in enumerate(outs)])
            p[outs...] = real(tr(ABC*ρ))
        end
        push!(ps, p)
    end
    return ps
end

# Overloaded for python interface
#function p_cycle(ρs::Vector{Matrix{T}}, Es::PyList{Any}; intervention = nothing) where T <: Number 
#    Es = [Vector{Matrix{ComplexF64}}(E) for E in Es]
#    return p_cycle(ρs, Es; intervention = intervention)  
#end
#
#function p_cycle(ρs::PyList{Any}, Es::Vector{Vector{Matrix{T}}}; intervention = nothing) where T <: Number 
#    ρs = [Matrix{ComplexF64}(ρ) for ρ in ρs]
#    return p_cycle(ρs, Es; intervention = intervention)  
#end
#
#function p_cycle(ρs::PyList{Any}, Es::PyList{Any}; intervention = nothing)
#    ρs = [Matrix{ComplexF64}(ρ) for ρ in ρs]
#    Es = [Vector{Matrix{ComplexF64}}(E) for E in Es]
#    return p_cycle(ρs, Es; intervention = intervention)  
#end

function p_Δ222_bell(; intervention = nothing,
        state_ids=(1,1,1),      obs_ids=(1,1,1), 
        state_θs=(π/4,π/4,π/4), obs_θs=(π/4,π/4,π/4)
    )
    """Generate observational and interventional distribution for the 222 triangle scenario
    using Bell states and measurements.
    Interventions use the same syntax as the `p_cycle` function.
    """
    ρs = [bell_state(k, θ=θ) for (k, θ) in zip(state_ids, state_θs)]
    Es = [[bell_proj(i, o, θ=θ) for o=1:2] for (i,θ) in zip(obs_ids, obs_θs)]
    return p_cycle(ρs, Es, intervention=intervention)
end

function p_Δ222_rotbell(; intervention = nothing,
        state_ids=(1,1,1),       
        state_θs=(π/4,π/4,π/4), 
        obs_ids=(1,1,1),
        obs_θs=(π/4,π/4,π/4),
        obs_φs=(0,0,0),
        obs_qbs=(1,1,1)
    )
    """Generate observational and interventional distribution for the 222 triangle scenario
    using Bell states and rotated Bell projectors.
    Interventions use the same syntax as the `p_cycle` function.
    """
    ρs = [bell_state(k, θ=θ) for (k, θ) in zip(state_ids, state_θs)]
    Es = [[state2proj(rotbell_state(i, θ=θ, φ=φ, qb=qb), o) for o=1:2] 
            for (i,θ,φ,qb) in zip(obs_ids, obs_θs, obs_φs, obs_qbs)]
    return p_cycle(ρs, Es, intervention=intervention)
end

function p_Δ222_and(; intervention = nothing,
        state_ids=(1,1,1), state_θs=(π/4,π/4,π/4),
        kA=1, θA=π/4, φA=π/4,
        kB=1, θB=π/4, φB=π/8,
        θC=π/5, φC=0.0
    )
    ρs = [bell_state(k, θ=θ) for (k, θ) in zip(state_ids, state_θs)]
    A = [state2proj(rotbell_state(kA, θ=θA, φ2=φA), o) for o=1:2]
    B = [state2proj(rotbell_state(kB, θ=θB, φ1=φB), o) for o=1:2]
    C = [state2proj(andtangled_state(θ=θC, φ=φC), o) for o=1:2]
    return p_cycle(ρs, [A, B, C], intervention=intervention)
end

function p_Δ444_EJM(ρs; ηs=(0.0,0.0,0.0), intervention=nothing)
    """Generate observational and interventional distribution for the EJM in the 
    triangle scenario
    Interventions use the same syntax as the `p_cycle` function.
    """
    Es = [EJM4(η=ηs[1]), EJM4(η=ηs[2]), EJM4(η=ηs[3])]
    return p_cycle(ρs, Es, intervention=intervention)
end

function p_Δ222_EJM(ρs; cgs=(2,2,2), ηs=(0.0,0.0,0.0), intervention=nothing)
    """Generate observational and interventional distribution for the 222 
    coarse grained EJM in the triangle scenario.
    Interventions use the same syntax as the `p_cycle` function.
    """
    Es = [EJM2(η=ηs[1], cg=cgs[1]), EJM2(η=ηs[2], cg=cgs[2]), EJM2(η=ηs[3], cg=cgs[3])]
    return p_cycle(ρs, Es, intervention=intervention)
end

function p_Δ222_EJM(; intervention=nothing,
        state_ids=(1,1,1), state_θs=(π/4,π/4,π/4),
        cgs=(2,2,2), ηs=(0,0,0))
    """Generate observational and interventional distribution for the 222 
    coarse grained EJM on Bell states in the triangle scenario.
    Interventions use the same syntax as the `p_cycle` function.
    """
    ρs = [bell_state(k, θ=θ) for (k, θ) in zip(state_ids, state_θs)]
    return p_Δ222_EJM(ρs; cgs=cgs, ηs=ηs, intervention=intervention)
end

## PGKTG distribution
function ppk(c1,c2,c3)
    p = zeros(2,2,2)
    for a=0:1, b=0:1, c=0:1
        p[a+1, b+1, c+1] = 1/8*(1 + c1*(a+b+c) + c2*(a*b + a*c + b*c) + c3*a*b*c)
    end
    return p
end

function ppk_int(c1,c2,c3)
    # TODO
    p = zeros(2,2,2)
    for a=0:1, b=0:1, c=0:1
        p[a+1, b+1, c+1] = 1/8*(1 + c1*(a+b+c) + c2*(a*b + a*c + b*c) + c3*a*b*c)
    end
    return p
end

## Some useful standard distributions

# Maximally mixed
pmix = ones(2,2,2)/8;

# 3-GHZ
pghz3 = zeros(2,2,2)
pghz3[1,1,1] = 1/2
pghz3[2,2,2] = 1/2

# 2-GHZ
pghz2 = zeros(2,2,2)
pghz2[1,1,1] = 1/4
pghz2[2,2,2] = 1/4 
pghz2[1,1,2] = 1/4
pghz2[2,2,1] = 1/4

# Evans distributions
pP_fun = (a,b,c; v=1, pβ=1/2)-> 1/8*(1 + v/√2 *(-1)^(a+b) * (1-2*pβ*c)) 
pP = (v, pβ) -> [pP_fun(a,b,c, v=v, pβ=pβ) for a=0:1, b=0:1, c=0:1]

function pQEvans(a,b,c; v=1, η=1, θ=π/8)
    if b==0
        return 1/16*(1 + η*(((1+v)^2)/4)*sin(2*θ)*(-1)^(a+c))
    else 
        return 1/16*(3 + v^2*(-1)^(a+c) + v*cos(2*θ)*((-1)^c-(-1)^a))
    end
end