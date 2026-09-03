#####
##### Where one chain starts
#####

"""
$(TYPEDEF)

Where a single chain starts.

$(FIELDS)
"""
struct InitChainConfig{T<:Real,P<:AbstractVector{<:Real},M<:AbstractVector{<:Real}}
    "Step size the chain starts from."
    step_size::T
    "Position the chain starts from."
    position::P
    "Diagonal of the mass matrix the chain starts from."
    mass::M
end

"""
$(SIGNATURES)

Construct the starting point of a single chain. The values are taken as given; an
[`InitConfig`](@ref) is what validates them.
"""
InitChainConfig(; step_size, position, mass) = InitChainConfig(step_size, position, mass)

#####
##### Where every chain starts
#####

"""
$(TYPEDEF)

Where every chain starts: one step size, one position and one diagonal mass matrix per chain.

$(FIELDS)
"""
struct InitConfig{T<:Real,P<:AbstractVector{<:Real},M<:AbstractVector{<:Real}}
    "Step size for each chain."
    step_sizes::Vector{T}
    "Starting position for each chain."
    positions::Vector{P}
    "Diagonal of the starting mass matrix for each chain."
    masses::Vector{M}
end

"""
$(SIGNATURES)

Construct the starting points of `num_chains` chains over `dims` dimensions.

Each of the three settings takes either one value shared by every chain or a collection
holding one value per chain: `step_size` a number or a collection of numbers, `position` and
`mass` a vector or a collection of vectors. A value shared by every chain is copied, so each
chain gets its own array to move from. Left unset, chains start at the origin under unit
mass.

Positions drawn at random come from [`random_positions`](@ref), masses estimated from the
target from [`estimate_masses`](@ref), and step sizes found by probing the target from
[`adapt_step_sizes`](@ref).
"""
function InitConfig(;
    num_chains::Integer,
    dims::Integer,
    step_size=0.1,
    position=nothing,
    mass=nothing,
)
    validate_nonnegative(num_chains, "num_chains")
    validate_nonnegative(dims, "dims")
    return InitConfig(
        step_sizes_per_chain(step_size, num_chains),
        positions_per_chain(something(position, zeros(dims)), num_chains, dims),
        masses_per_chain(something(mass, ones(dims)), num_chains, dims),
    )
end

step_sizes_per_chain(step_size::Real, num_chains::Integer) =
    (validate_finite_positive(step_size, "step size"); fill(float(step_size), num_chains))

function step_sizes_per_chain(step_sizes, num_chains::Integer)
    xs = collect(step_sizes)
    validate_size(xs, num_chains, "step_sizes", "num_chains")
    validate_finite_positive(xs, "step_size")
    return xs
end

function positions_per_chain(
    position::AbstractVector{<:Real},
    num_chains::Integer,
    dims::Integer,
)
    validate_size(position, dims, "position", "dims")
    validate_finite(position, "position")
    return [copy(position) for _ in 1:num_chains]
end

function positions_per_chain(positions, num_chains::Integer, dims::Integer)
    ps = collect(positions)
    validate_size(ps, num_chains, "positions", "num_chains")
    validate_finite(ps, "positions")
    for p in ps
        validate_size(p, dims, "position", "dims")
    end
    return ps
end

function masses_per_chain(mass::AbstractVector{<:Real}, num_chains::Integer, dims::Integer)
    validate_size(mass, dims, "masses", "dims")
    validate_finite_positive(mass, "masses")
    return [copy(mass) for _ in 1:num_chains]
end

function masses_per_chain(masses, num_chains::Integer, dims::Integer)
    ms = collect(masses)
    validate_size(ms, num_chains, "masses", "num_chains")
    validate_finite_positive(ms, "masses")
    for m in ms
        validate_size(m, dims, "all masses", "dims")
    end
    return ms
end

"""
$(SIGNATURES)

Return the number of chains `cfg` starts.
"""
num_chains(cfg::InitConfig) = length(cfg.step_sizes)

"""
$(SIGNATURES)

Return the dimensionality of the positions `cfg` starts from, or zero if it starts no chains.
"""
dims(cfg::InitConfig) = isempty(cfg.positions) ? 0 : length(first(cfg.positions))

"""
$(SIGNATURES)

Return where chain `n` starts.
"""
init_chain_config(cfg::InitConfig, n::Integer) =
    InitChainConfig(cfg.step_sizes[n], cfg.positions[n], cfg.masses[n])

"""
$(SIGNATURES)

Draw a starting position for each of `num_chains` chains over `dims` dimensions, every
coordinate an independent normal draw of standard deviation `scale`. Chains started far
apart make it visible when they fail to reach the same place.
"""
function random_positions(
    rng::AbstractRNG,
    num_chains::Integer,
    dims::Integer;
    scale::Real=1.0,
)
    validate_finite_positive(scale, "init_scale")
    return [scale .* randn(rng, dims) for _ in 1:num_chains]
end

"""
$(SIGNATURES)

Estimate a diagonal mass matrix at each of `positions` from the gradient of the target there,
taking the absolute gradient and pulling it towards one: weight `smoothing` on one and
`1 - smoothing` on the absolute gradient, so that a coordinate whose gradient happens to
vanish still gets a usable scale.

With `average = true` every chain is given the same mass matrix, the geometric mean of the
per-chain estimates, so that a chain that started somewhere unrepresentative does not keep
its own scaling.

Follows Seyboldt, Carlson and Carpenter (2026), "Preconditioning Hamiltonian Monte Carlo by
minimizing Fisher divergence", [arXiv:2603.18845](https://arxiv.org/abs/2603.18845v1).
"""
function estimate_masses(logp_grad!, positions; smoothing, average::Bool=false)
    validate_probability(smoothing, "mass_smoothing")
    masses = [(1 - smoothing) .* abs.(grad(logp_grad!, θ)) .+ smoothing for θ in positions]
    (average && !isempty(masses)) || return masses
    geometric_mean = exp.(sum(m -> log.(m), masses) ./ length(masses))
    return [copy(geometric_mean) for _ in eachindex(masses)]
end

"""
$(SIGNATURES)

Return `cfg` with each chain's step size replaced by one that [`adapt_step`](@ref) found
acceptable from that chain's starting position and mass matrix, searching from the step size
the chain already has.
"""
function adapt_step_sizes(rng::AbstractRNG, logp_grad!, cfg::InitConfig)
    step_sizes = [
        adapt_step(rng, logp_grad!, cfg.positions[c], cfg.masses[c], cfg.step_sizes[c])
        for c in eachindex(cfg.step_sizes, cfg.positions, cfg.masses)
    ]
    return InitConfig(step_sizes, cfg.positions, cfg.masses)
end

#####
##### Warmup
#####

"""
$(TYPEDEF)

How warmup tunes the step size and the diagonal mass matrix.

Warmup stops once every chain's step size and mass matrix are within
`step_size_converge_tol` and `mass_converge_tol` of the cross-chain geometric mean, but never
before `min_iter` iterations and never after `max_iter`.

$(FIELDS)
"""
struct WarmupConfig
    "Iterations warmup runs before it is allowed to stop."
    min_iter::Int
    "Iterations after which warmup stops whether or not the chains agree."
    max_iter::Int
    "How far a chain's step size may sit from the cross-chain mean and still count as agreed."
    step_size_converge_tol::Float64
    "How far a chain's mass matrix may sit from the cross-chain mean and still count as agreed."
    mass_converge_tol::Float64
    "Observations the initial mass matrix is weighted as though it came from."
    mass_init_count::Float64
    "Added to the estimated mass matrix so that a near-zero variance cannot divide."
    mass_additive_smoothing::Float64
    "Macro steps per trajectory the minimum micro step count is aimed at."
    max_macro_steps_target::Float64
    "Acceptance rate the step size is driven towards."
    step_accept_rate_target::Float64
    "Size of the step size's move per iteration, before the decay schedule is applied."
    step_learning_rate::Float64
    "Weight the step-size optimiser's running mean gradient keeps on its history."
    step_gradient_decay::Float64
    "Weight the step-size optimiser's running mean squared gradient keeps on its history."
    step_sq_gradient_decay::Float64
    "Added to the step-size optimiser's root mean squared gradient so a tiny one cannot divide."
    step_stabilization::Float64
    "Exponent by which the step-size learning rate decays with the iteration count."
    step_learn_rate_decay::Float64
    "Iterations between a chain publishing its statistics for the convergence check."
    publish_stride::Int
    "Iterations between a chain's task yielding to the scheduler."
    yield_period::Int

    function WarmupConfig(
        min_iter,
        max_iter,
        step_size_converge_tol,
        mass_converge_tol,
        mass_init_count,
        mass_additive_smoothing,
        max_macro_steps_target,
        step_accept_rate_target,
        step_learning_rate,
        step_gradient_decay,
        step_sq_gradient_decay,
        step_stabilization,
        step_learn_rate_decay,
        publish_stride,
        yield_period,
    )
        min_iter, max_iter = Int(min_iter), Int(max_iter)
        publish_stride, yield_period = Int(publish_stride), Int(yield_period)
        validate_iter_range(min_iter, max_iter)
        validate_finite_positive(step_size_converge_tol, "step_size_converge_tol")
        validate_finite_positive(mass_converge_tol, "mass_converge_tol")
        validate_finite_positive(mass_init_count, "mass_init_count")
        validate_finite_positive(mass_additive_smoothing, "mass_additive_smoothing")
        validate_finite_positive(max_macro_steps_target, "max_macro_steps_target")
        validate_probability(step_accept_rate_target, "step_accept_rate_target")
        validate_finite_positive(step_learning_rate, "step_learning_rate")
        validate_probability(step_gradient_decay, "step_gradient_decay")
        validate_probability(step_sq_gradient_decay, "step_sq_gradient_decay")
        validate_finite_positive(step_stabilization, "step_stabilization")
        validate_probability(step_learn_rate_decay, "step_learn_rate_decay")
        validate_positive(publish_stride, "publish_stride")
        validate_positive(yield_period, "yield_period")
        return new(
            min_iter,
            max_iter,
            step_size_converge_tol,
            mass_converge_tol,
            mass_init_count,
            mass_additive_smoothing,
            max_macro_steps_target,
            step_accept_rate_target,
            step_learning_rate,
            step_gradient_decay,
            step_sq_gradient_decay,
            step_stabilization,
            step_learn_rate_decay,
            publish_stride,
            yield_period,
        )
    end
end

"""
$(SIGNATURES)

Construct a warmup configuration. Every default is the one the `walnutpie` C++ package uses.
"""
function WarmupConfig(;
    min_iter=50,
    max_iter=1000,
    step_size_converge_tol=0.1,
    mass_converge_tol=1.0,
    mass_init_count=4.0,
    mass_additive_smoothing=1e-5,
    max_macro_steps_target=15.0,
    step_accept_rate_target=0.8,
    step_learning_rate=0.05,
    step_gradient_decay=0.8,
    step_sq_gradient_decay=0.9,
    step_stabilization=1e-4,
    step_learn_rate_decay=0.5,
    publish_stride=5,
    yield_period=32,
)
    return WarmupConfig(
        min_iter,
        max_iter,
        step_size_converge_tol,
        mass_converge_tol,
        mass_init_count,
        mass_additive_smoothing,
        max_macro_steps_target,
        step_accept_rate_target,
        step_learning_rate,
        step_gradient_decay,
        step_sq_gradient_decay,
        step_stabilization,
        step_learn_rate_decay,
        publish_stride,
        yield_period,
    )
end

"""
$(SIGNATURES)

Throw an `ArgumentError` unless `min_iter` and `max_iter` are a usable range.
"""
function validate_iter_range(min_iter::Integer, max_iter::Integer)
    validate_nonnegative(min_iter, "min_iter")
    validate_nonnegative(max_iter, "max_iter")
    min_iter <= max_iter && return nothing
    throw(ArgumentError("min_iter must be <= max_iter"))
end

"""
$(SIGNATURES)

Construct the step-size optimiser `warmup` describes, starting from `step_size`.
"""
Adam(warmup::WarmupConfig, step_size::Real) = Adam(;
    step_size,
    accept_rate_target=warmup.step_accept_rate_target,
    learning_rate=warmup.step_learning_rate,
    gradient_decay=warmup.step_gradient_decay,
    sq_gradient_decay=warmup.step_sq_gradient_decay,
    stabilization=warmup.step_stabilization,
    learn_rate_decay=warmup.step_learn_rate_decay,
)

#####
##### Sampling
#####

"""
$(TYPEDEF)

How the sampler builds trajectories and when it stops drawing.

Sampling stops once the cross-chain agreement measure on the log density falls below
`rhat_converge_tol`, but never before `min_iter` draws and never after `max_iter`.

$(FIELDS)
"""
struct SamplingConfig
    "Draws taken before sampling is allowed to stop."
    min_iter::Int
    "Draws after which sampling stops whether or not the chains agree."
    max_iter::Int
    "How many times a trajectory may be doubled in length."
    max_trajectory_doublings::Int
    "How many times the step size may be halved within one trajectory segment."
    max_step_halvings::Int
    "Energy a segment may lose before its step size is halved and it is simulated again."
    max_hamiltonian_error::Float64
    "Smallest number of micro steps a macro step is broken into."
    min_micro_steps::Int
    "Cross-chain agreement measure on the log density below which sampling may stop."
    rhat_converge_tol::Float64

    function SamplingConfig(
        min_iter,
        max_iter,
        max_trajectory_doublings,
        max_step_halvings,
        max_hamiltonian_error,
        min_micro_steps,
        rhat_converge_tol,
    )
        min_iter, max_iter = Int(min_iter), Int(max_iter)
        max_trajectory_doublings = Int(max_trajectory_doublings)
        max_step_halvings, min_micro_steps = Int(max_step_halvings), Int(min_micro_steps)
        validate_iter_range(min_iter, max_iter)
        validate_nonnegative(max_trajectory_doublings, "max_trajectory_doublings")
        validate_nonnegative(max_step_halvings, "max_step_halvings")
        validate_finite_positive(max_hamiltonian_error, "max_hamiltonian_error")
        validate_positive(min_micro_steps, "min_micro_steps")
        validate_finite_gt1(rhat_converge_tol, "rhat_converge_tol")
        return new(
            min_iter,
            max_iter,
            max_trajectory_doublings,
            max_step_halvings,
            max_hamiltonian_error,
            min_micro_steps,
            rhat_converge_tol,
        )
    end
end

"""
$(SIGNATURES)

Construct a sampling configuration. Every default is the one the `walnutpie` C++ package
uses.
"""
function SamplingConfig(;
    min_iter=50,
    max_iter=1000,
    max_trajectory_doublings=5,
    max_step_halvings=5,
    max_hamiltonian_error=0.5,
    min_micro_steps=1,
    rhat_converge_tol=1.01,
)
    return SamplingConfig(
        min_iter,
        max_iter,
        max_trajectory_doublings,
        max_step_halvings,
        max_hamiltonian_error,
        min_micro_steps,
        rhat_converge_tol,
    )
end

#####
##### The whole run
#####

"""
$(TYPEDEF)

Everything a Walnuts run needs: where the chains start, how warmup tunes them, and how they
then draw.

$(FIELDS)
"""
struct WalnutsConfig{I<:InitConfig}
    "Where every chain starts."
    init::I
    "How warmup tunes the step size and mass matrix."
    warmup::WarmupConfig
    "How the sampler builds trajectories and when it stops."
    sampling::SamplingConfig
end

"""
$(SIGNATURES)

Construct the configuration for a whole run.
"""
WalnutsConfig(; init, warmup=WarmupConfig(), sampling=SamplingConfig()) =
    WalnutsConfig(init, warmup, sampling)
