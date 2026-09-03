module Walnutpie

using Compat: @compat
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF
using LogExpFunctions: logaddexp
using Random: AbstractRNG, randn!
using Statistics: mean, var

export InitConfig, SamplingConfig, WalnutsConfig, WarmupConfig, walnuts

include("validate.jl")
include("util.jl")
include("online_moments.jl")
include("adam.jl")
include("config.jl")
include("trajectory.jl")
include("sampler.jl")
include("run.jl")

@compat public on_sample,
on_warmup,
on_warmup_complete,
on_rhat,
on_logp_exception,
is_interrupted

@compat public InitChainConfig,
init_chain_config,
num_chains,
dims,
random_positions,
estimate_masses,
adapt_step_sizes

@compat public AdaptiveWalnuts, warmup!, sampler, WalnutsSampler, draw!, step_size, inv_mass

end
