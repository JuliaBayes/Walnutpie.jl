#####
##### Step-size adaptation
#####

"""
$(TYPEDEF)

Drives a step size towards one whose proposals are accepted at a target rate, by stochastic
gradient descent on the logarithm of the step size (the Adam method of Kingma and Ba,
[arXiv:1412.6980](https://arxiv.org/abs/1412.6980)).

The quantity being minimised is half the squared distance between the observed and the target
acceptance rate, so the gradient descended is just the target minus the observation:
accepting too often pushes the step size up, accepting too rarely pulls it down. Each step is
divided by the square root of a running mean of squared gradients, which keeps the size of
the move steady however large the errors happen to be.

The learning rate is divided by the observation count raised to `learn_rate_decay`, which is
what makes the estimate settle rather than keep bouncing with every new observation. An
exponent of zero recovers the standard method, whose estimate never settles; any exponent
above zero and up to one converges (Zou, Shen, Jie, Zhang and Liu,
[arXiv:1811.09358](https://arxiv.org/abs/1811.09358)). [`WarmupConfig`](@ref) admits the
exponents strictly between zero and one.

$(FIELDS)
"""
mutable struct Adam
    "Logarithm of the current step size."
    log_step_size::Float64
    "Running mean of the gradient."
    grad_mean::Float64
    "Running mean of the squared gradient."
    sq_grad_mean::Float64
    "How many acceptance rates have been observed."
    t::Int
    "`gradient_decay` raised to `t`; corrects `grad_mean` for having started at zero."
    gradient_decay_pow::Float64
    "`sq_gradient_decay` raised to `t`; corrects `sq_grad_mean` for having started at zero."
    sq_gradient_decay_pow::Float64
    "Acceptance rate the step size is driven towards."
    const accept_rate_target::Float64
    "Size of the move made per observation, before the decay schedule is applied."
    const learning_rate::Float64
    "Weight the running mean of the gradient keeps on its history."
    const gradient_decay::Float64
    "Weight the running mean of the squared gradient keeps on its history."
    const sq_gradient_decay::Float64
    "Added to the square root of the mean squared gradient so that a tiny one cannot divide."
    const stabilization::Float64
    "Exponent by which the learning rate decays with the observation count."
    const learn_rate_decay::Float64
end

"""
$(SIGNATURES)

Construct a step-size optimiser starting from `step_size`. The tuning parameters are those a
[`WarmupConfig`](@ref) carries, and are expected to have been validated there.
"""
function Adam(;
    step_size,
    accept_rate_target,
    learning_rate,
    gradient_decay,
    sq_gradient_decay,
    stabilization,
    learn_rate_decay,
)
    return Adam(
        log(step_size),
        0.0,
        0.0,
        0,
        1.0,
        1.0,
        accept_rate_target,
        learning_rate,
        gradient_decay,
        sq_gradient_decay,
        stabilization,
        learn_rate_decay,
    )
end

"""
$(SIGNATURES)

Observe an acceptance rate `α` in ``(0, 1)`` and move the step size towards the target rate.
Returns `a`.
"""
function observe!(a::Adam, α::Real)
    a.t += 1
    a.gradient_decay_pow *= a.gradient_decay
    a.sq_gradient_decay_pow *= a.sq_gradient_decay

    ∇ = a.accept_rate_target - α
    a.grad_mean = a.gradient_decay * a.grad_mean + (1 - a.gradient_decay) * ∇
    a.sq_grad_mean = a.sq_gradient_decay * a.sq_grad_mean + (1 - a.sq_gradient_decay) * ∇^2

    grad_mean = a.grad_mean / (1 - a.gradient_decay_pow)
    sq_grad_mean = a.sq_grad_mean / (1 - a.sq_gradient_decay_pow)

    learning_rate = a.learning_rate / a.t^a.learn_rate_decay
    a.log_step_size -= learning_rate * grad_mean / (sqrt(sq_grad_mean) + a.stabilization)
    return a
end

"""
$(SIGNATURES)

Return the current step size estimate.
"""
step_size(a::Adam) = exp(a.log_step_size)

"""
$(SIGNATURES)

Return the logarithm of the current step size estimate.
"""
log_step_size(a::Adam) = a.log_step_size
