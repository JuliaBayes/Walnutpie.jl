#####
##### The handler interface
#####
#
# A handler is any object with methods for the events it cares about. Nothing dispatches on a
# handler type, so no abstract type is introduced and no method is required: each fallback
# below ignores its event, and a handler defines only the methods it wants.

"""
$(SIGNATURES)

Report a post-warmup draw `θ` and its log density. Handlers add a method for their own type;
the fallback ignores the draw.
"""
on_sample(handler, θ, logp) = nothing

"""
$(SIGNATURES)

Report a warmup iteration: the position drawn, its log density, and the step size and
diagonal inverse mass matrix the iteration used. Handlers add a method for their own type;
the fallback ignores the iteration.
"""
on_warmup(handler, θ, logp, step_size, inv_mass) = nothing

"""
$(SIGNATURES)

Report the step size and diagonal inverse mass matrix warmup settled on. Handlers add a
method for their own type; the fallback ignores the report.
"""
on_warmup_complete(handler, step_size, inv_mass) = nothing

"""
$(SIGNATURES)

Report how far the chains still disagree about the log density, where one means they agree
exactly and a larger number means they do not (aka the R-hat statistic). Handlers add a
method for their own type; the fallback ignores the report.
"""
on_rhat(handler, rhat) = nothing

"""
$(SIGNATURES)

Return whether the run should stop now. Callbacks add a method for their own type; the
fallback never interrupts.

Both the warmup and the sampling controller ask each time a chain publishes, and raise an
`InterruptException` when the answer is `true`. A run is therefore interruptible between
iterations rather than during one, so a single very slow evaluation of the target is seen
through to its end.
"""
is_interrupted(callback) = false

"""
$(SIGNATURES)

Return the target `f` wraps, without the exception trapping.
"""
target(f::NoExceptLogpGrad) = f.logp_grad!

"""
$(TYPEDEF)

Ignores the acceptance estimates a trajectory reports, for the sampler whose step size is
already settled.
"""
struct NoStepSizeAdapter end

"""
$(SIGNATURES)

Discard an acceptance estimate.
"""
observe!(::NoStepSizeAdapter, α::Real) = nothing

#####
##### Drawing with settled tuning
#####

"""
$(TYPEDEF)

Draws from a target with a fixed step size and mass matrix, forming a Markov chain whose
stationary distribution is the target.

Call [`draw!`](@ref) for each draw. Warmup hands one of these over through
[`sampler`](@ref); constructing one directly draws with the tuning it is given.

$(FIELDS)
"""
mutable struct WalnutsSampler{
    R<:AbstractRNG,
    F,
    H,
    V<:AbstractVector{<:Real},
    M<:AbstractVector{<:Real},
}
    "Source of the momentum draws and the jump decisions."
    const rng::R
    "Told about each draw through [`on_sample`](@ref)."
    const handler::H
    "The target, wrapped so that an exception it raises does not end the run."
    const logp_grad!::NoExceptLogpGrad{F,H}
    "The current position."
    θ::V
    "Diagonal of the inverse mass matrix."
    const inv_mass::M
    "Square roots of the diagonal of the mass matrix, which scale the momentum draws."
    const chol_mass::M
    "Size of the micro steps a macro step starts out taking."
    const step_size::Float64
    "How many times a trajectory may be doubled."
    const max_trajectory_doublings::Int
    "How many times the step size may be halved within one macro step."
    const max_step_halvings::Int
    "Smallest number of micro steps a macro step is broken into."
    const min_micro_steps::Int
    "Energy a macro step may lose before its step size is halved."
    const max_hamiltonian_error::Float64
end

"""
$(SIGNATURES)

Construct a sampler drawing from `logp_grad!` starting at `θ`, under the diagonal inverse
mass matrix `inv_mass`.

Every tuning value is checked here rather than at the first draw, so that a mistake surfaces
before the run rather than after it.
"""
function WalnutsSampler(
    rng::AbstractRNG,
    handler,
    logp_grad!,
    θ::AbstractVector{<:Real},
    inv_mass::AbstractVector{<:Real};
    step_size,
    max_trajectory_doublings,
    max_step_halvings,
    min_micro_steps,
    max_hamiltonian_error,
)
    validate_positive(inv_mass, "inv_mass")
    validate_positive(step_size, "step_size")
    validate_positive(Int(max_trajectory_doublings), "max_trajectory_doublings")
    validate_positive(Int(max_step_halvings), "max_step_halvings")
    validate_positive(Int(min_micro_steps), "min_micro_steps")
    validate_positive(max_hamiltonian_error, "max_hamiltonian_error")
    inv_mass = float.(inv_mass)
    return WalnutsSampler(
        rng,
        handler,
        NoExceptLogpGrad(logp_grad!, handler),
        float.(θ),
        inv_mass,
        inv.(sqrt.(inv_mass)),
        step_size,
        max_trajectory_doublings,
        max_step_halvings,
        min_micro_steps,
        max_hamiltonian_error,
    )
end

"""
$(SIGNATURES)

Advance the chain by one draw, report it through [`on_sample`](@ref), and return its log
density.
"""
function draw!(s::WalnutsSampler)
    ms = MacroStep(
        s.inv_mass,
        s.step_size,
        s.max_step_halvings,
        s.min_micro_steps,
        s.max_hamiltonian_error,
    )
    θ, _, logp, _ = transition(
        s.rng,
        s.logp_grad!,
        NoStepSizeAdapter(),
        ms,
        s.chol_mass,
        s.θ,
        s.max_trajectory_doublings,
    )
    s.θ = θ
    on_sample(s.handler, θ, logp)
    return logp
end

#####
##### Estimating the mass matrix
#####

"""
$(TYPEDEF)

Estimates a diagonal inverse mass matrix from the draws and from the gradients of the target
at them, forgetting the past at a rate that slows as warmup proceeds.

Either quantity alone estimates the scale of a coordinate: for a Gaussian target the draws
have the variance of the coordinate and the gradients its inverse. Taking the square root of
their ratio — the geometric mean of the one estimate and the other — is more robust than
either, since a coordinate the chain has barely explored is misjudged by the draws but not by
the gradients.

$(FIELDS)
"""
struct MassEstimator{V<:AbstractVector{<:Real}}
    "Observations the initial mass matrix is weighted as though it came from."
    init_count::Float64
    "Running mean and variance of the draws."
    draws::OnlineMoments{V}
    "Running mean and variance of the gradients."
    gradients::OnlineMoments{V}
end

"""
$(SIGNATURES)

Construct an estimator started from the mass matrix `init` gives, weighted as though it came
from `warmup.mass_init_count` observations.
"""
function MassEstimator(warmup::WarmupConfig, init::InitChainConfig)
    init_mean = zero(float.(init.position))
    init_count = warmup.mass_init_count
    return MassEstimator(
        init_count,
        OnlineMoments(; init_weight=init_count, init_mean, init_variance=inv.(init.mass)),
        OnlineMoments(; init_weight=init_count, init_mean, init_variance=init.mass),
    )
end

"""
$(SIGNATURES)

Fold the draw `θ` and the gradient `∇` of the target there, made at `iteration`, into `est`,
and return `est`.

The weight the past keeps rises towards one with the iteration count, so the estimate settles
as warmup proceeds instead of chasing the latest draws forever.
"""
function observe!(
    est::MassEstimator,
    θ::AbstractVector,
    ∇::AbstractVector,
    iteration::Integer,
)
    discount_factor = 1 - 1 / (est.init_count + iteration)
    observe!(est.draws, discount_factor, θ)
    observe!(est.gradients, discount_factor, ∇)
    return est
end

"""
$(SIGNATURES)

Return the diagonal of the estimated inverse mass matrix.
"""
inv_mass_estimate(est::MassEstimator) =
    sqrt.(variance(est.draws) ./ variance(est.gradients))

#####
##### Estimating how finely to step
#####

"""
$(TYPEDEF)

Raises the smallest number of micro steps a macro step is broken into, so that trajectories
average a target number of macro steps.

A macro step made of more micro steps covers more simulated time, so a trajectory needs fewer
of them; the count to aim for is therefore the observed average divided by the target. The
estimate never falls below the floor it was constructed with. It starts from one observation
of two macro steps, which keeps the first few estimates from swinging on a single trajectory.

$(FIELDS)
"""
mutable struct MinMicroStepsEstimator
    "Macro steps per trajectory the estimate aims at."
    const target::Float64
    "Smallest number of micro steps the estimate may report."
    const lower_bound::Int
    "Macro steps observed in total."
    total::Float64
    "Trajectories observed."
    count::Float64
end

"""
$(SIGNATURES)

Construct an estimator aiming at `target` macro steps per trajectory and never reporting
fewer than `lower_bound` micro steps.
"""
MinMicroStepsEstimator(target::Real, lower_bound::Integer) =
    MinMicroStepsEstimator(target, lower_bound, 2.0, 1.0)

"""
$(SIGNATURES)

Fold a trajectory of `macro_steps` macro steps into `est` and return `est`.
"""
function observe!(est::MinMicroStepsEstimator, macro_steps::Integer)
    est.total += macro_steps
    est.count += 1
    return est
end

"""
$(SIGNATURES)

Return the smallest number of micro steps a macro step should be broken into.
"""
min_micro_steps(est::MinMicroStepsEstimator) = max(
    est.lower_bound,
    round(Int, (est.total / est.count) / est.target, RoundNearestTiesAway),
)

#####
##### Drawing while tuning
#####

"""
$(TYPEDEF)

Draws while tuning the step size and the diagonal mass matrix, re-estimating both every
iteration and forgetting the past at a rate that slows as it goes.

Its draws do not come from the target and are not valid for inference. Call
[`warmup!`](@ref) once per warmup iteration, then [`sampler`](@ref) for a
[`WalnutsSampler`](@ref) that holds the tuning fixed and does form a Markov chain.

$(FIELDS)
"""
mutable struct AdaptiveWalnuts{R<:AbstractRNG,F,H,V<:AbstractVector{<:Real},M}
    "Source of the momentum draws and the jump decisions."
    const rng::R
    "Told about each warmup iteration through [`on_warmup`](@ref)."
    const handler::H
    "The target, wrapped so that an exception it raises does not end the run."
    const logp_grad!::NoExceptLogpGrad{F,H}
    "How warmup tunes the step size and mass matrix."
    const warmup::WarmupConfig
    "How trajectories are built."
    const sampling::SamplingConfig
    "The current position."
    θ::V
    "Warmup iterations taken so far."
    iteration::Int
    "Drives the step size towards the target acceptance rate."
    const step_size_optimiser::Adam
    "Estimates the diagonal inverse mass matrix."
    const mass_estimator::MassEstimator{M}
    "Estimates how finely a macro step should be broken up."
    const micro_steps_estimator::MinMicroStepsEstimator
end

"""
$(SIGNATURES)

Construct a warming-up sampler for one chain, starting where `init` says.
"""
function AdaptiveWalnuts(
    rng::AbstractRNG,
    handler,
    logp_grad!,
    init::InitChainConfig,
    warmup::WarmupConfig,
    sampling::SamplingConfig,
)
    return AdaptiveWalnuts(
        rng,
        handler,
        NoExceptLogpGrad(logp_grad!, handler),
        warmup,
        sampling,
        float.(init.position),
        0,
        Adam(warmup, init.step_size),
        MassEstimator(warmup, init),
        MinMicroStepsEstimator(warmup.max_macro_steps_target, sampling.min_micro_steps),
    )
end

"""
$(SIGNATURES)

Return the step size `a` has tuned to.
"""
step_size(a::AdaptiveWalnuts) = step_size(a.step_size_optimiser)

"""
$(SIGNATURES)

Return the logarithm of the step size `a` has tuned to.
"""
log_step_size(a::AdaptiveWalnuts) = log_step_size(a.step_size_optimiser)

"""
$(SIGNATURES)

Return the diagonal of the inverse mass matrix `a` has tuned to.
"""
inv_mass(a::AdaptiveWalnuts) = inv_mass_estimate(a.mass_estimator)

"""
$(SIGNATURES)

Return the logarithm of the diagonal of the mass matrix `a` has tuned to.
"""
log_mass(a::AdaptiveWalnuts) = -log.(inv_mass(a))

"""
$(SIGNATURES)

Take one warmup iteration and return `a`.

The trajectory is built with the current tuning; the step-size optimiser is fed the
acceptance estimate from each coarsest macro step along it, the draw and the gradient there
go into the mass estimate, and the number of macro steps the trajectory took goes into the
estimate of how finely to step.
"""
function warmup!(a::AdaptiveWalnuts)
    inv_M = inv_mass(a)
    chol_M = sqrt.(inv.(inv_M))
    ms = MacroStep(
        inv_M,
        step_size(a),
        a.sampling.max_step_halvings,
        min_micro_steps(a.micro_steps_estimator),
        a.sampling.max_hamiltonian_error,
    )
    θ, ∇, logp, depth = transition(
        a.rng,
        a.logp_grad!,
        a.step_size_optimiser,
        ms,
        chol_M,
        a.θ,
        a.sampling.max_trajectory_doublings,
    )
    a.θ = θ
    observe!(a.mass_estimator, θ, ∇, a.iteration)
    observe!(a.micro_steps_estimator, 1 << depth)
    on_warmup(a.handler, θ, logp, step_size(a), inv_M)
    a.iteration += 1
    return a
end

"""
$(SIGNATURES)

Report the tuning warmup settled on through [`on_warmup_complete`](@ref) and return a
[`WalnutsSampler`](@ref) holding it fixed, carrying on from where warmup left off.
"""
function sampler(a::AdaptiveWalnuts)
    ϵ = step_size(a)
    inv_M = inv_mass(a)
    on_warmup_complete(a.handler, ϵ, inv_M)
    return WalnutsSampler(
        a.rng,
        a.handler,
        target(a.logp_grad!),
        a.θ,
        inv_M;
        step_size=ϵ,
        max_trajectory_doublings=a.sampling.max_trajectory_doublings,
        max_step_halvings=a.sampling.max_step_halvings,
        min_micro_steps=min_micro_steps(a.micro_steps_estimator),
        max_hamiltonian_error=a.sampling.max_hamiltonian_error,
    )
end
