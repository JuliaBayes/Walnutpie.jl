module Walnutpie

using Compat: @compat
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF
using LogExpFunctions: logaddexp
using Random: AbstractRNG, randn!
using Statistics: mean, var

# Enough to configure and start a run. Every other part of the interface is reached through
# the module, since the names are ones a caller is likely to have of their own.
export InitConfig, SamplingConfig, WalnutsConfig, WarmupConfig, walnuts

include("validate.jl")
include("util.jl")
include("online_moments.jl")
include("adam.jl")
include("config.jl")
include("trajectory.jl")
include("sampler.jl")
include("run.jl")

# Part of the interface, but reached through the module rather than exported. `public` is a
# syntax error before Julia 1.11, which `@compat` covers by making it a no-op there.

# What a handler and an interrupt callback may define methods for.
@compat public on_sample,
on_warmup,
on_warmup_complete,
on_rhat,
on_logp_exception,
is_interrupted

# Building up the starting points of the chains.
@compat public InitChainConfig,
init_chain_config,
num_chains,
dims,
random_positions,
estimate_masses,
adapt_step_sizes

# Driving a single chain, for a caller who wants the loop rather than `walnuts`.
@compat public AdaptiveWalnuts, warmup!, sampler, WalnutsSampler, draw!, step_size, inv_mass

end
