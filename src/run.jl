#####
##### What a chain publishes
#####

"""
$(TYPEDEF)

What a warming-up chain publishes about its tuning.

$(FIELDS)
"""
struct WarmupSnapshot{V<:AbstractVector{<:Real}}
    "Warmup iterations the chain has taken."
    iteration::Int
    "Logarithm of the chain's step size."
    log_step_size::Float64
    "Logarithm of the diagonal of the chain's mass matrix."
    log_mass::V
    "Diagonal of the chain's mass matrix."
    mass::V
end

"""
$(SIGNATURES)

Take a snapshot of the tuning `a` has reached after `iteration` warmup iterations.
"""
function WarmupSnapshot(a::AdaptiveWalnuts, iteration::Integer)
    log_mass_a = log_mass(a)
    return WarmupSnapshot(iteration, log_step_size(a), log_mass_a, exp.(log_mass_a))
end

"""
$(TYPEDEF)

What a drawing chain publishes about the log densities of its draws.

$(FIELDS)
"""
struct SamplingSnapshot
    "Mean of the log densities."
    mean::Float64
    "Sample variance of the log densities."
    variance::Float64
    "Draws the chain has taken."
    count::Int
end

"""
$(SIGNATURES)

Take a snapshot of the log densities `acc` has accumulated.
"""
SamplingSnapshot(acc::WelfordAccumulator) =
    SamplingSnapshot(acc.mean, sample_variance(acc), acc.n)

#####
##### Running chains against a controller
#####

"Iterations between a drawing chain's task yielding to the scheduler."
const SAMPLING_YIELD_PERIOD = 1024

"""
$(SIGNATURES)

Run one task per chain and let `control` decide when they stop. Chains are numbered from
one, so anything `work!` or `control` indexes by chain number must be one-based.

`work!(m, publish, stop)` runs chain `m`, calling `publish(snapshot)` with each snapshot of
type `S` it wants read and returning once `stop[]` is set. `control(channel, stop)` reads
`(chain, snapshot)` pairs off `channel` and returns the result of the run; it may set `stop[]`
itself, and the channel closes once every chain has finished, so it is never left waiting on
a snapshot that will not come.

The chains are stopped and waited for however `control` leaves. An exception raised by a
chain is re-thrown here, unless `control` raised one first.
"""
function run_chains(work!, control, num_chains::Integer, ::Type{S}) where {S}
    channel = Channel{Tuple{Int,S}}(Inf)
    stop = Threads.Atomic{Bool}(false)
    tasks = map(1:num_chains) do m
        Threads.@spawn work!(m, snapshot -> put!(channel, (m, snapshot)), stop)
    end
    # Closing the channel is what ends the controller's read loop when the chains finish of
    # their own accord. Failures are left for the caller to re-throw below.
    closer = Threads.@spawn begin
        for task in tasks
            try
                wait(task)
            catch
            end
        end
        close(channel)
    end
    result = try
        control(channel, stop)
    finally
        stop[] = true
        wait(closer)
    end
    foreach(wait, tasks)
    return result
end

"""
$(SIGNATURES)

Ask `interrupt` whether to stop, and if it says so, stop the chains and raise.
"""
function check_interrupt(interrupt, stop::Threads.Atomic{Bool})
    is_interrupted(interrupt) || return nothing
    stop[] = true
    throw(InterruptException())
end

#####
##### Warmup across chains
#####

"""
$(SIGNATURES)

Take warmup iterations with `a`, publishing a snapshot of its tuning every
`warmup.publish_stride` iterations, until `warmup.max_iter` iterations have been taken or the
controller sets `stop`. A final snapshot is published either way.
"""
function warmup_chain!(a::AdaptiveWalnuts, publish, stop, warmup::WarmupConfig)
    publish(WarmupSnapshot(a, 0))
    iteration = 1
    while iteration <= warmup.max_iter
        stop[] && break
        iteration % warmup.yield_period == 0 && yield()
        warmup!(a)
        iteration % warmup.publish_stride == 0 && publish(WarmupSnapshot(a, iteration))
        iteration += 1
    end
    publish(WarmupSnapshot(a, iteration - 1))
    return a
end

"""
$(SIGNATURES)

Read tuning snapshots until the chains agree, until they have each taken `warmup.max_iter`
iterations, or until interrupted, then stop them. Returns the cross-chain geometric mean of
the mass matrices and of the step sizes.

The chains agree when no chain's mass matrix lies further than `warmup.mass_converge_tol`
from the geometric mean in relative Euclidean distance, and no chain's step size exceeds the
geometric mean by more than `warmup.step_size_converge_tol` in relative terms. Agreement is
not looked for until every chain has taken `warmup.min_iter` iterations.
"""
function control_warmup(
    channel,
    stop,
    latest::AbstractVector,
    warmup::WarmupConfig,
    interrupt,
)
    Base.require_one_based_indexing(latest)
    num_chains = length(latest)
    max_draws = num_chains * warmup.max_iter
    geometric_means() = (
        exp.(sum(s -> s.log_mass, latest) ./ num_chains),
        exp(sum(s -> s.log_step_size, latest) / num_chains),
    )
    geom_mean_mass, geom_mean_step = geometric_means()
    for (m, snapshot) in channel
        latest[m] = snapshot
        check_interrupt(interrupt, stop)
        all(s -> s.iteration >= warmup.min_iter, latest) || continue
        geom_mean_mass, geom_mean_step = geometric_means()
        max_rel_diff_mass = maximum(s -> l2_rel_diff(s.mass, geom_mean_mass), latest)
        max_rel_diff_step =
            maximum(s -> (exp(s.log_step_size) - geom_mean_step) / geom_mean_step, latest)
        agreed =
            max_rel_diff_mass <= warmup.mass_converge_tol &&
            max_rel_diff_step <= warmup.step_size_converge_tol
        if agreed || sum(s -> s.iteration, latest) >= max_draws
            stop[] = true
            break
        end
    end
    return (geom_mean_mass, geom_mean_step)
end

"""
$(SIGNATURES)

Warm up every chain in `adapters` concurrently and return the cross-chain geometric mean of
the mass matrices and of the step sizes they settled on.
"""
function adapt!(adapters::AbstractVector, warmup::WarmupConfig, interrupt)
    Base.require_one_based_indexing(adapters)
    isempty(adapters) && return (Float64[], NaN)
    latest = [WarmupSnapshot(a, 0) for a in adapters]
    return run_chains(
        (m, publish, stop) -> warmup_chain!(adapters[m], publish, stop, warmup),
        (channel, stop) -> control_warmup(channel, stop, latest, warmup, interrupt),
        length(adapters),
        eltype(latest),
    )
end

#####
##### Sampling across chains
#####

"""
$(SIGNATURES)

Draw with `s`, publishing the running mean and variance of the log densities after each
draw, until `sampling.max_iter` draws have been taken or the controller sets `stop`.
"""
function sample_chain!(s::WalnutsSampler, publish, stop, sampling::SamplingConfig)
    logp_stats = WelfordAccumulator()
    iteration = 1
    while iteration <= sampling.max_iter
        stop[] && break
        iteration % SAMPLING_YIELD_PERIOD == 0 && yield()
        observe!(logp_stats, draw!(s))
        publish(SamplingSnapshot(logp_stats))
        iteration += 1
    end
    return s
end

"""
$(SIGNATURES)

Read draw statistics until the chains agree on the log density, until they have each taken
`sampling.max_iter` draws, or until interrupted, then stop them.

Agreement is measured by how much of the spread in the log density lies between the chains
rather than within them: one when they agree exactly, larger when they do not (aka the R-hat
statistic). It is reported through [`on_rhat`](@ref) and compared against
`sampling.rhat_converge_tol`, and is not looked for until every chain has taken
`sampling.min_iter` draws.
"""
function control_sampling(
    channel,
    stop,
    num_chains::Integer,
    sampling::SamplingConfig,
    handler,
    interrupt,
)
    latest = [SamplingSnapshot(NaN, NaN, 0) for _ in 1:num_chains]
    means = fill(NaN, num_chains)
    variances = fill(NaN, num_chains)
    max_draws = num_chains * sampling.max_iter
    for (m, snapshot) in channel
        latest[m] = snapshot
        check_interrupt(interrupt, stop)
        all(s -> s.count >= sampling.min_iter, latest) || continue
        for c in eachindex(latest, means, variances)
            means[c] = latest[c].mean
            variances[c] = latest[c].variance
        end
        rhat = sqrt(1 + var(means) / mean(variances))
        on_rhat(handler, rhat)
        if rhat <= sampling.rhat_converge_tol || sum(s -> s.count, latest) >= max_draws
            stop[] = true
            break
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Draw from every sampler in `samplers` concurrently until the chains agree.
"""
function sample!(
    samplers::AbstractVector,
    sampling::SamplingConfig,
    global_handler,
    interrupt,
)
    Base.require_one_based_indexing(samplers)
    isempty(samplers) && return nothing
    return run_chains(
        (m, publish, stop) -> sample_chain!(samplers[m], publish, stop, sampling),
        (channel, stop) -> control_sampling(
            channel,
            stop,
            length(samplers),
            sampling,
            global_handler,
            interrupt,
        ),
        length(samplers),
        SamplingSnapshot,
    )
end

#####
##### The whole run
#####

"""
$(SIGNATURES)

Warm up the chains `config` describes, then draw from them, and return the cross-chain
geometric mean of the mass matrices and of the step sizes warmup settled on.

`logp_grad!(∇, θ)` writes the gradient of the target into `∇` and returns its log density.
Chains run concurrently on separate tasks, so it must be safe to call from several tasks at
once, and `rngs` must hold one generator per chain rather than one shared between them.

Fixing the generators fixes the sequence of draws each chain makes, but not how many of them
it makes: the controller stops the chains when it observes that they agree, and how far each
has got by then depends on how the tasks were scheduled.

Each chain's handler in `chain_handlers` is told about that chain's warmup iterations
([`on_warmup`](@ref)), the tuning it settled on ([`on_warmup_complete`](@ref)), its draws
([`on_sample`](@ref)), and any exception the target raised ([`on_logp_exception`](@ref)).
`global_handler` is told how far the chains still disagree as sampling proceeds
([`on_rhat`](@ref)). `interrupt` is asked whether to stop ([`is_interrupted`](@ref)).
"""
function walnuts(
    rngs::AbstractVector,
    chain_handlers::AbstractVector,
    global_handler,
    interrupt,
    logp_grad!,
    config::WalnutsConfig,
)
    Base.require_one_based_indexing(rngs, chain_handlers)
    n = num_chains(config.init)
    validate_size(rngs, n, "rngs", "num_chains")
    validate_size(chain_handlers, n, "chain_handlers", "num_chains")
    adapters = [
        AdaptiveWalnuts(
            rngs[m],
            chain_handlers[m],
            logp_grad!,
            init_chain_config(config.init, m),
            config.warmup,
            config.sampling,
        ) for m in 1:n
    ]
    result = adapt!(adapters, config.warmup, interrupt)
    sample!([sampler(a) for a in adapters], config.sampling, global_handler, interrupt)
    return result
end
