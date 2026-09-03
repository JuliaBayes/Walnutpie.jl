#####
##### Trajectory segments
#####

"""
$(TYPEDEF)

A stretch of a Hamiltonian trajectory, holding only what extending it and combining it with
its neighbours require: its two endpoints, the state selected from it so far, and the log of
the summed joint density of every state along it.

Each endpoint carries a position, a momentum, the gradient of the target log density there,
and the joint log density of the position and the momentum. The gradients are kept rather
than recomputed, since recomputing one means calling the target again.

The weight the segment carries is `logp`, the log of the summed joint densities along it,
which is what the two update rules compare when deciding whether to jump to it.

A segment is never written to once built. Combining two segments takes the arrays of their
outer endpoints as they stand, so segments share arrays and writing through one would corrupt
the others.

$(FIELDS)
"""
struct SpanW{V<:AbstractVector{<:Real}}
    "Position at the earliest endpoint."
    θ_bk::V
    "Momentum at the earliest endpoint."
    ρ_bk::V
    "Gradient of the target log density at the earliest endpoint."
    ∇_bk::V
    "Joint log density of the position and momentum at the earliest endpoint."
    logp_bk::Float64
    "Position at the latest endpoint."
    θ_fw::V
    "Momentum at the latest endpoint."
    ρ_fw::V
    "Gradient of the target log density at the latest endpoint."
    ∇_fw::V
    "Joint log density of the position and momentum at the latest endpoint."
    logp_fw::Float64
    "The state selected from the segment, which is what a draw is taken from."
    θ_select::V
    "Gradient of the target log density at the selected state."
    ∇_select::V
    "Log density of the selected position, without its momentum."
    logp_pos_select::Float64
    "Log of the summed joint densities of every state along the segment."
    logp::Float64
end

"""
$(SIGNATURES)

Construct the segment holding the single state `(θ, ρ)`, which is necessarily both of its
endpoints and its selected state. `logp_pos` is the log density of `θ` alone and `logp_joint`
that of `θ` together with `ρ`.
"""
SpanW(θ, ρ, ∇, logp_pos::Real, logp_joint::Real) =
    SpanW(θ, ρ, ∇, logp_joint, θ, ρ, ∇, logp_joint, θ, ∇, logp_pos, logp_joint)

"""
$(SIGNATURES)

Construct the segment spanning `span_bk` and the segment `span_fw` that adjoins it later in
time, taking the outer endpoint of each. The selected state and the weight are the caller's,
since choosing between the two selected states is what [`combine`](@ref) does.
"""
SpanW(
    span_bk::SpanW,
    span_fw::SpanW,
    θ_select,
    ∇_select,
    logp_pos_select::Real,
    logp::Real,
) = SpanW(
    span_bk.θ_bk,
    span_bk.ρ_bk,
    span_bk.∇_bk,
    span_bk.logp_bk,
    span_fw.θ_fw,
    span_fw.ρ_fw,
    span_fw.∇_fw,
    span_fw.logp_fw,
    θ_select,
    ∇_select,
    logp_pos_select,
    logp,
)

"""
$(SIGNATURES)

Return `(x₁, x₂)` stepping forward in time and `(x₂, x₁)` stepping backward, so that a
routine written for one time direction reads the same in the other.
"""
order(::Forward, x₁, x₂) = (x₁, x₂)
order(::Backward, x₁, x₂) = (x₂, x₁)

"""
$(SIGNATURES)

Return the endpoint of `span` that a step in direction `dir` leaves from, as
`(θ, ρ, ∇, logp_joint)`.
"""
endpoint(::Forward, span::SpanW) = (span.θ_fw, span.ρ_fw, span.∇_fw, span.logp_fw)
endpoint(::Backward, span::SpanW) = (span.θ_bk, span.ρ_bk, span.∇_bk, span.logp_bk)

"""
$(SIGNATURES)

Return `ϵ` signed for a step in direction `dir`. Simulating backward in time is simulating
forward with a negative step.
"""
signed_step(::Forward, ϵ::Real) = ϵ
signed_step(::Backward, ϵ::Real) = -ϵ

"""
$(SIGNATURES)

Return whether the trajectory formed by `span₁` and `span₂`, ordered in time by `dir`, has
doubled back on itself.

Let `δ` be the separation between the trajectory's two ends, measured under the inverse mass
matrix `inv_M`. The trajectory has doubled back once the momentum at either end points away
from `δ`, since carrying on from there would retrace ground the trajectory already covers.
"""
function uturn(dir::Direction, span₁::SpanW, span₂::SpanW, inv_M::AbstractVector)
    span_bk, span_fw = order(dir, span₁, span₂)
    θ_start, ρ_start = span_bk.θ_bk, span_bk.ρ_bk
    θ_end, ρ_end = span_fw.θ_fw, span_fw.ρ_fw
    ix = eachindex(θ_start, θ_end, ρ_start, ρ_end, inv_M)
    δ(i) = inv_M[i] * (θ_end[i] - θ_start[i])
    return sum(i -> ρ_end[i] * δ(i), ix) < 0 || sum(i -> ρ_start[i] * δ(i), ix) < 0
end

#####
##### Simulating a macro step
#####

"""
$(TYPEDEF)

How one macro step of a trajectory is simulated.

A macro step is `min_micro_steps` leapfrog steps of size `step_size`. When it loses more
energy than `max_error`, the step size is halved and the number of steps doubled — covering
the same interval of simulated time more finely — up to `max_step_halvings` times. This is
what lets one trajectory take fine steps through a region of high curvature and coarse ones
elsewhere.

$(FIELDS)
"""
struct MacroStep{V<:AbstractVector{<:Real}}
    "Diagonal of the inverse mass matrix."
    inv_M::V
    "Size of the micro steps a macro step starts out taking."
    step_size::Float64
    "How many times the step size may be halved."
    max_step_halvings::Int
    "Smallest number of micro steps a macro step is broken into."
    min_micro_steps::Int
    "Energy a macro step may lose before its step size is halved."
    max_error::Float64
end

"""
$(SIGNATURES)

Take one leapfrog step of size `ϵ` from `θ`, `ρ` and `∇`, overwriting all three with the
state reached, and return the log density of the position reached.

This is the innermost loop of the sampler and allocates nothing, which the loops below are
written out to keep: in-place broadcast emits a branch that copies a source array when it
cannot rule out that the source aliases the destination, and that branch is enough for
`AllocCheck.check_allocs` to report the routine as allocating.
"""
function leapfrog!(
    logp_grad!,
    θ::AbstractVector,
    ρ::AbstractVector,
    ∇::AbstractVector,
    inv_M::AbstractVector,
    ϵ::Real,
)
    half_ϵ = ϵ / 2
    for i in eachindex(ρ, ∇)
        ρ[i] += half_ϵ * ∇[i]
    end
    for i in eachindex(θ, inv_M, ρ)
        θ[i] += ϵ * inv_M[i] * ρ[i]
    end
    logp_pos = logp_grad!(∇, θ)
    for i in eachindex(ρ, ∇)
        ρ[i] += half_ϵ * ∇[i]
    end
    return logp_pos
end

"""
$(SIGNATURES)

Take `num_steps` leapfrog steps of size `ϵ` from `θ`, `ρ` and `∇`, overwriting all three with
the state reached, and return whether the joint log density there is within `max_error` of
`logp_joint`.
"""
function within_tolerance!(
    logp_grad!,
    θ::AbstractVector,
    ρ::AbstractVector,
    ∇::AbstractVector,
    inv_M::AbstractVector,
    ϵ::Real,
    num_steps::Integer,
    max_error::Real,
    logp_joint::Real,
)
    logp_pos = float(logp_joint)
    for _ in 1:num_steps
        logp_pos = leapfrog!(logp_grad!, θ, ρ, ∇, inv_M, ϵ)
    end
    return abs((logp_pos + logp_momentum(ρ, inv_M)) - logp_joint) <= max_error
end

"""
$(SIGNATURES)

Return whether `num_steps` is the number of micro steps a simulation would have settled on
had it started from the state the macro step ended at, `(θ, ρ, ∇)`, and run the other way.

How finely a macro step is stepped depends on the trajectory it is taken along, so the chain
only targets the intended distribution if the reverse move would have made the same choice.
It would have, exactly when every coarser simulation back from the endpoint *fails* to stay
within `max_error` — a coarser one that succeeded is one the reverse move would have stopped
at instead. A macro step of a single micro step has nothing coarser to try and always
reverses.
"""
function reversible(
    logp_grad!,
    θ::AbstractVector,
    ρ::AbstractVector,
    ∇::AbstractVector,
    inv_M::AbstractVector,
    ϵ::Real,
    num_steps::Integer,
    min_micro_steps::Integer,
    max_error::Real,
    logp_joint::Real,
)
    num_steps == 1 && return true
    θ_back, ρ_back, ∇_back = similar(θ), similar(ρ), similar(∇)
    while num_steps >= 2 * min_micro_steps
        copyto!(θ_back, θ)
        for i in eachindex(ρ_back, ρ)
            ρ_back[i] = -ρ[i]
        end
        copyto!(∇_back, ∇)
        num_steps ÷= 2
        ϵ *= 2
        within_tolerance!(
            logp_grad!,
            θ_back,
            ρ_back,
            ∇_back,
            inv_M,
            ϵ,
            num_steps,
            max_error,
            logp_joint,
        ) && return false
    end
    return true
end

"""
$(SIGNATURES)

Take a macro step from the endpoint of `span` that `dir` leaves from, writing the state
reached into `θ_next`, `ρ_next` and `∇_next`, and return
`(ok, logp_pos_next, logp_joint_next)`.

`ok` is false when the step could not be made accurate enough within `ms.max_step_halvings`
halvings, or when the step it settled on would not have been reversed. The acceptance
estimate from the coarsest attempt is reported to `adapter` through
[`observe!`](@ref), which is the only acceptance signal warmup's step-size optimiser gets.
"""
function macro_step!(
    dir::Direction,
    logp_grad!,
    adapter,
    ms::MacroStep,
    span::SpanW,
    θ_next::AbstractVector,
    ρ_next::AbstractVector,
    ∇_next::AbstractVector,
)
    θ, ρ, ∇, logp = endpoint(dir, span)
    inv_M = ms.inv_M
    ϵ = signed_step(dir, ms.step_size)
    num_steps = ms.min_micro_steps
    logp_pos_next = -Inf
    logp_joint_next = -Inf
    for _ in 1:(ms.max_step_halvings)
        copyto!(θ_next, θ)
        copyto!(ρ_next, ρ)
        copyto!(∇_next, ∇)
        for _ in 1:num_steps
            logp_pos_next = leapfrog!(logp_grad!, θ_next, ρ_next, ∇_next, inv_M, ϵ)
        end
        logp_joint_next = logp_pos_next + logp_momentum(ρ_next, inv_M)
        energy_error = abs(logp - logp_joint_next)
        num_steps == ms.min_micro_steps && observe!(adapter, exp(-energy_error))
        if energy_error <= ms.max_error
            ok = reversible(
                logp_grad!,
                θ_next,
                ρ_next,
                ∇_next,
                inv_M,
                ϵ,
                num_steps,
                ms.min_micro_steps,
                ms.max_error,
                logp_joint_next,
            )
            return (ok, logp_pos_next, logp_joint_next)
        end
        num_steps *= 2
        ϵ /= 2
    end
    return (false, logp_pos_next, logp_joint_next)
end

#####
##### Building a trajectory
#####

"""
$(SIGNATURES)

Return the denominator `rule` measures a new segment's weight against, given the old
segment's weight and the total across both. The two update rules differ in nothing else.
"""
log_denominator(::Metropolis, logp_old::Real, logp_total::Real) = logp_old
log_denominator(::Barker, logp_old::Real, logp_total::Real) = logp_total

"""
$(SIGNATURES)

Return `span_old` and `span_new` joined into one segment, ordered in time by `dir`, having
chosen which of the two selected states the joined segment carries.

The new segment is jumped to with probability given by its weight over what `rule` measures
that weight against.
"""
function combine(
    rng::AbstractRNG,
    rule::Update,
    dir::Direction,
    span_old::SpanW,
    span_new::SpanW,
)
    logp_total = logaddexp(span_old.logp, span_new.logp)
    log_jump_prob = span_new.logp - log_denominator(rule, span_old.logp, logp_total)
    selected = log(rand(rng)) < log_jump_prob ? span_new : span_old
    span_bk, span_fw = order(dir, span_old, span_new)
    return SpanW(
        span_bk,
        span_fw,
        selected.θ_select,
        selected.∇_select,
        selected.logp_pos_select,
        logp_total,
    )
end

"""
$(SIGNATURES)

Return the single-state segment one macro step from `span` in direction `dir`, or `nothing`
if that step could not be taken accurately and reversibly.
"""
function build_leaf(dir::Direction, logp_grad!, adapter, ms::MacroStep, span::SpanW)
    θ_next = similar(span.θ_fw)
    ρ_next = similar(θ_next)
    ∇_next = similar(θ_next)
    ok, logp_pos, logp_joint =
        macro_step!(dir, logp_grad!, adapter, ms, span, θ_next, ρ_next, ∇_next)
    ok || return nothing
    return SpanW(θ_next, ρ_next, ∇_next, logp_pos, logp_joint)
end

"""
$(SIGNATURES)

Return a segment of `2^depth` macro steps extending `span` in direction `dir`, or `nothing`
if any step failed or the two halves of a doubling turned back on each other.

The two halves are combined under the Barker rule, which weighs the new half against the
total across both rather than against the old half alone.
"""
function build_span(
    rng::AbstractRNG,
    dir::Direction,
    logp_grad!,
    adapter,
    ms::MacroStep,
    span::SpanW,
    depth::Integer,
)
    depth == 0 && return build_leaf(dir, logp_grad!, adapter, ms, span)
    span₁ = build_span(rng, dir, logp_grad!, adapter, ms, span, depth - 1)
    span₁ === nothing && return nothing
    span₂ = build_span(rng, dir, logp_grad!, adapter, ms, span₁, depth - 1)
    span₂ === nothing && return nothing
    uturn(dir, span₁, span₂, ms.inv_M) && return nothing
    return combine(rng, Barker(), dir, span₁, span₂)
end

"""
$(SIGNATURES)

Double `span` in direction `dir` and return `(span, stop)`, where `stop` says the trajectory
has nowhere left to go: the new half could not be built, or the doubled trajectory turns back
on itself.

The doubling is combined into the trajectory under the Metropolis rule, which weighs the new
half against the trajectory so far.
"""
function extend(
    rng::AbstractRNG,
    dir::Direction,
    logp_grad!,
    adapter,
    ms::MacroStep,
    span::SpanW,
    depth::Integer,
)
    next = build_span(rng, dir, logp_grad!, adapter, ms, span, depth)
    next === nothing && return (span, true)
    made_uturn = uturn(dir, span, next, ms.inv_M)
    return (combine(rng, Metropolis(), dir, span, next), made_uturn)
end

"""
$(SIGNATURES)

Draw the next state of the Markov chain from `θ`, returning
`(θ_next, ∇_next, logp_next, depth)`: the position selected, the gradient of the target
there, its log density, and the number of doublings the trajectory took.

A momentum is drawn from the mass matrix — `chol_M` holds the square roots of its diagonal,
which is what scales a standard normal draw to it — and the trajectory is then doubled up to `max_depth` times, each doubling going forward or
backward with equal probability. Doubling stops early once the trajectory turns back on
itself or a segment cannot be built.
"""
function transition(
    rng::AbstractRNG,
    logp_grad!,
    adapter,
    ms::MacroStep,
    chol_M::AbstractVector,
    θ::AbstractVector,
    max_depth::Integer,
)
    ρ = randn!(rng, similar(θ, float(eltype(θ))))
    ρ .*= chol_M
    ∇ = similar(ρ)
    logp_pos = logp_grad!(∇, θ)
    span = SpanW(θ, ρ, ∇, logp_pos, logp_pos + logp_momentum(ρ, ms.inv_M))
    depth = 1
    while depth <= max_depth
        span, stop = if rand(rng, Bool)
            extend(rng, Forward(), logp_grad!, adapter, ms, span, depth - 1)
        else
            extend(rng, Backward(), logp_grad!, adapter, ms, span, depth - 1)
        end
        stop && break
        depth += 1
    end
    return (span.θ_select, span.∇_select, span.logp_pos_select, depth)
end
