#####
##### Running mean and variance of a stream of scalars
#####

"""
$(TYPEDEF)

Running mean and sample variance of a stream of numbers, updated one observation at a time
and holding only three numbers however long the stream runs (Welford's algorithm).

Accumulating the squared deviations from the *running* mean, rather than subtracting the
square of the mean from the mean of the squares, keeps the variance accurate when the numbers
are large relative to their spread.

$(FIELDS)
"""
mutable struct WelfordAccumulator
    "How many observations have been folded in."
    n::Int
    "Mean of the observations; zero before any are seen."
    mean::Float64
    "Sum of squared deviations from the running mean."
    sum_sq_dev::Float64
end

"""
$(SIGNATURES)

Construct an accumulator holding no observations.
"""
WelfordAccumulator() = WelfordAccumulator(0, 0.0, 0.0)

"""
$(SIGNATURES)

Fold `x` into `acc` and return `acc`.
"""
function observe!(acc::WelfordAccumulator, x::Real)
    acc.n += 1
    δ = x - acc.mean
    acc.mean += δ / acc.n
    acc.sum_sq_dev += δ * (x - acc.mean)
    return acc
end

"""
$(SIGNATURES)

Return the unbiased sample variance of the observations, dividing the sum of squared
deviations by one less than their count. Fewer than two observations leave it undefined, and
give not-a-number.
"""
function sample_variance(acc::WelfordAccumulator)
    acc.n > 1 || return NaN
    return acc.sum_sq_dev / (acc.n - 1)
end

"""
$(SIGNATURES)

Discard the observations, returning `acc` to the state it was constructed in.
"""
function reset!(acc::WelfordAccumulator)
    acc.n = 0
    acc.mean = 0.0
    acc.sum_sq_dev = 0.0
    return acc
end

#####
##### Running mean and variance of a stream of vectors, forgetting the past
#####

"""
$(TYPEDEF)

Running mean and variance of a stream of vectors, weighting an observation less the further
back it lies: the one from `k` updates ago carries weight `discount_factor^k`, where the
discount factor is given per update. A discount factor of 1 forgets nothing and gives the
ordinary running mean and variance; 0 forgets everything but the latest observation.

Both the storage and the cost of an update are proportional to the dimension, whatever the
length of the stream. The arithmetic follows [`WelfordAccumulator`](@ref), to which this
reduces when nothing is discounted.

$(FIELDS)
"""
mutable struct OnlineMoments{V<:AbstractVector{<:Real}}
    "Combined weight of the observations so far."
    weight::Float64
    "Weighted mean of the observations."
    mean::V
    "Weighted sum of squared deviations from the running mean, one per dimension."
    sum_sq_dev::V
end

"""
$(SIGNATURES)

Construct an accumulator over `dims` dimensions holding no observations, whose mean is zero
and whose variance is one.
"""
OnlineMoments(dims::Integer) = OnlineMoments(0.0, zeros(dims), zeros(dims))

"""
$(SIGNATURES)

Construct an accumulator started from `init_mean` and `init_variance`, weighted as though
they had been measured from `init_weight` observations.
"""
function OnlineMoments(; init_weight, init_mean, init_variance)
    validate_positive(init_weight, "init_weight")
    validate_same_size(init_mean, init_variance, "init_mean", "init_variance")
    weight = float(init_weight)
    mean = similar(init_mean, Float64)
    sum_sq_dev = similar(init_mean, Float64)
    mean .= init_mean
    sum_sq_dev .= weight .* init_variance
    return OnlineMoments(weight, mean, sum_sq_dev)
end

"""
$(SIGNATURES)

Fold `y` into `m`, having first discounted the weight of everything already there by
`discount_factor`, and return `m`. The new observation is given a weight of one.
"""
function observe!(m::OnlineMoments, discount_factor::Real, y::AbstractVector{<:Real})
    validate_probability_inclusive(discount_factor, "discount_factor")
    m.weight = discount_factor * m.weight + 1
    for i in eachindex(m.mean, m.sum_sq_dev, y)
        δ = y[i] - m.mean[i]
        m.mean[i] += δ / m.weight
        m.sum_sq_dev[i] = discount_factor * m.sum_sq_dev[i] + δ * (y[i] - m.mean[i])
    end
    return m
end

"""
$(SIGNATURES)

Return the weighted variance of the observations, dividing the weighted sum of squared
deviations by the total weight. With no weight behind it the accumulator has nothing to
report, and gives a variance of one in every dimension.
"""
function variance(m::OnlineMoments)
    m.weight > 0 || return fill!(similar(m.mean), 1)
    return m.sum_sq_dev ./ m.weight
end
