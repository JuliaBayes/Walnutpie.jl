#####
##### Argument validation
#####
#
# Each message below is asserted on by the configuration tests, so the exact
# wording is part of this module's interface, not a detail.

"""
$(SIGNATURES)

Throw an `ArgumentError` unless `x` has `n` elements. `var` names `x` and `target` names
where `n` came from; both appear in the message.
"""
function validate_size(x, n::Integer, var::AbstractString, target::AbstractString)
    length(x) == n && return nothing
    throw(ArgumentError("$var size must match $target: got $(length(x)), need $n"))
end

"""
$(SIGNATURES)

Throw an `ArgumentError` unless `x₁` and `x₂` have the same number of elements.
"""
function validate_same_size(x₁, x₂, name₁::AbstractString, name₂::AbstractString)
    length(x₁) == length(x₂) && return nothing
    throw(
        ArgumentError(
            "$name₁ and $name₂ must be the same size: $(length(x₁)) and $(length(x₂))",
        ),
    )
end

"""
$(SIGNATURES)

Throw an `ArgumentError` unless `x` is finite and greater than 1.
"""
function validate_finite_gt1(x::Real, var::AbstractString)
    (isfinite(x) && x > 1) && return nothing
    throw(ArgumentError("$var must be finite and > 1"))
end

"""
$(SIGNATURES)

Throw an `ArgumentError` unless `x` is finite and strictly positive.
"""
function validate_finite_positive(x::Real, var::AbstractString)
    (isfinite(x) && x > 0) && return nothing
    throw(ArgumentError("$var must be finite and > 0"))
end

"""
$(SIGNATURES)

Throw an `ArgumentError` unless every element of `xs` is finite and strictly positive.
Dispatch recurses, so a collection of collections is checked elementwise throughout.
"""
function validate_finite_positive(xs::AbstractArray, var::AbstractString)
    for x in xs
        validate_finite_positive(x, var)
    end
    return nothing
end

"""
$(SIGNATURES)

Throw an `ArgumentError` unless `x` is finite.
"""
function validate_finite(x::Real, var::AbstractString)
    isfinite(x) && return nothing
    throw(ArgumentError("$var must be finite"))
end

"""
$(SIGNATURES)

Throw an `ArgumentError` unless every element of `xs` is finite. Dispatch recurses, so a
collection of collections is checked elementwise throughout.
"""
function validate_finite(xs::AbstractArray, var::AbstractString)
    for x in xs
        validate_finite(x, var)
    end
    return nothing
end

"""
$(SIGNATURES)

Throw an `ArgumentError` unless `x` lies in ``(0, ∞)``. A not-a-number value fails, since
it does not compare greater than zero.
"""
function validate_positive(x::Real, name::AbstractString)
    (x > 0 && !isinf(x)) && return nothing
    throw(ArgumentError("$name must be in (0, inf)."))
end

"""
$(SIGNATURES)

Throw an `ArgumentError` unless the integer `x` is strictly positive. Integers cannot be
infinite, so this is the only bound to check.
"""
function validate_positive(x::Integer, name::AbstractString)
    x > 0 && return nothing
    throw(ArgumentError("$name must be in {1, 2, ... }"))
end

"""
$(SIGNATURES)

Throw an `ArgumentError` unless every element of `x` lies in ``(0, ∞)``.
"""
function validate_positive(x::AbstractArray, name::AbstractString)
    (all(>(0), x) && all(isfinite, x)) && return nothing
    throw(ArgumentError("$name must be in (0, inf)."))
end

"""
$(SIGNATURES)

Throw an `ArgumentError` unless `x` lies strictly between 0 and 1.
"""
function validate_probability(x::Real, name::AbstractString)
    (x > 0 && x < 1) && return nothing
    throw(ArgumentError("$name must be in (0, 1)"))
end

"""
$(SIGNATURES)

Throw an `ArgumentError` unless `x` lies in ``[0, 1]``, endpoints included.
"""
function validate_probability_inclusive(x::Real, name::AbstractString)
    (x >= 0 && x <= 1) && return nothing
    throw(ArgumentError("$name must be in [0, 1]"))
end

"""
$(SIGNATURES)

Throw an `ArgumentError` unless the integer `x` is zero or greater.
"""
function validate_nonnegative(x::Integer, name::AbstractString)
    x >= 0 && return nothing
    throw(ArgumentError("$name must be in {0, 1, ... }"))
end
