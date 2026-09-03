#####
##### Trajectory tags
#####

"""
$(TYPEDEF)

Time direction of a Hamiltonian simulation. Subtypes are [`Forward`](@ref) and
[`Backward`](@ref), used as singletons so that the direction is resolved when the method is
compiled rather than branched on at each step.
"""
abstract type Direction end

"""
$(TYPEDEF)

Step forward in time.
"""
struct Forward <: Direction end

"""
$(TYPEDEF)

Step backward in time.
"""
struct Backward <: Direction end

"""
$(TYPEDEF)

Rule for deciding whether to jump to a newly built trajectory segment. Subtypes are
[`Barker`](@ref) and [`Metropolis`](@ref); they differ only in what the new segment's weight
is compared against.
"""
abstract type Update end

"""
$(TYPEDEF)

Jump with probability proportional to the new segment's share of the weight across both
segments.
"""
struct Barker <: Update end

"""
$(TYPEDEF)

Jump with probability given by the new segment's weight relative to the old segment's.
"""
struct Metropolis <: Update end

#####
##### Hamiltonian quantities
#####

"""
$(SIGNATURES)

Return the unnormalised log density of momentum `ρ` under a diagonal mass matrix whose
inverse has diagonal `inv_M`, which is the negative kinetic energy `-ρ'M⁻¹ρ/2`.
"""
function logp_momentum(ρ::AbstractVector, inv_M::AbstractVector)
    return -sum(i -> inv_M[i] * abs2(ρ[i]), eachindex(ρ, inv_M)) / 2
end

"""
$(SIGNATURES)

Return the change in the joint log density of position and momentum across a single leapfrog
step of size `ϵ` taken from `θ` with momentum `ρ`.

Exact Hamiltonian dynamics would leave the joint density unchanged, so the result measures
how much the discretisation has distorted it — zero for a perfect step, negative when the
step loses probability mass.

`logp_grad!` writes the gradient of the target into its first argument and returns the log
density at its second.
"""
function leapfrog_error(
    logp_grad!,
    θ::AbstractVector,
    ρ::AbstractVector,
    inv_M::AbstractVector,
    ϵ::Real,
)
    ∇ = similar(θ)
    logp = logp_grad!(∇, θ) + logp_momentum(ρ, inv_M)
    ρ′ = ρ .+ (ϵ / 2) .* ∇
    θ′ = θ .+ ϵ .* inv_M .* ρ′
    logp′ = logp_grad!(∇, θ′)
    ρ′ .+= (ϵ / 2) .* ∇
    return (logp′ + logp_momentum(ρ′, inv_M)) - logp
end

"""
$(SIGNATURES)

Return a step size for the target `logp_grad!` at position `θ` under the diagonal mass matrix
`M`, starting the search from `ϵ`.

A momentum is drawn once and held fixed. The step is doubled while a single leapfrog step
from it would be accepted with probability above 0.9, then multiplied by `√(1/2)` while that
probability is below 0.6, leaving it between the two. This refines the coarser doubling
heuristic used by NUTS, which stops at the first crossing.
"""
function adapt_step(
    rng::AbstractRNG,
    logp_grad!,
    θ::AbstractVector,
    M::AbstractVector,
    ϵ::Real,
)
    inv_M = inv.(M)
    ρ = randn!(rng, similar(M, float(eltype(M))))
    ρ .*= sqrt.(M)
    while leapfrog_error(logp_grad!, θ, ρ, inv_M, ϵ) > log(0.9)
        ϵ *= 2
    end
    while leapfrog_error(logp_grad!, θ, ρ, inv_M, ϵ) < log(0.6)
        ϵ *= sqrt(0.5)
    end
    return ϵ
end

#####
##### Calling the target
#####

"""
$(SIGNATURES)

Report that evaluating the target at `θ` threw `exception`. Handlers add a method for their
own type; the fallback ignores the report.
"""
on_logp_exception(handler, θ, exception) = nothing

"""
$(TYPEDEF)

Wraps a target so that an exception raised while evaluating it is reported to `handler` and
then treated as a position of zero probability, rather than ending the run. A target that is
undefined on part of the space is common enough that a chain wandering into one should
recover instead of dying.

$(FIELDS)
"""
struct NoExceptLogpGrad{F,H}
    "The wrapped target, called as `logp_grad!(∇, θ)`."
    logp_grad!::F
    "Told about each exception through [`on_logp_exception`](@ref)."
    handler::H
end

"""
$(SIGNATURES)

Write the gradient of the target at `θ` into `∇` and return its log density. If the target
throws, report it and return negative infinity with a zero gradient.

An `InterruptException` is re-thrown rather than swallowed, so that a run stays interruptible
even when the target is being called in a tight loop.
"""
function (f::NoExceptLogpGrad)(∇::AbstractVector, θ::AbstractVector)
    try
        return f.logp_grad!(∇, θ)
    catch exception
        exception isa InterruptException && rethrow()
        on_logp_exception(f.handler, θ, exception)
        fill!(∇, 0)
        return -Inf
    end
end

"""
$(SIGNATURES)

Return the gradient of the target at `θ`, discarding the log density.
"""
function grad(logp_grad!, θ::AbstractVector)
    ∇ = similar(θ)
    logp_grad!(∇, θ)
    return ∇
end

#####
##### Comparing vectors
#####

"""
$(SIGNATURES)

Return the Euclidean norm of the elementwise relative difference between `a` and the baseline
`b`, that is `norm((a - b) ./ b)`. Scaling both arguments by the same factor leaves the result
unchanged, which is what makes it usable as a convergence measure across chains on different
scales.
"""
function l2_rel_diff(a::AbstractVector, b::AbstractVector)
    return sqrt(sum(i -> abs2((a[i] - b[i]) / b[i]), eachindex(a, b)))
end
