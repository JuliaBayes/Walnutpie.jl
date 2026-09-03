using Walnutpie
# Public but unexported, so named here rather than at every use site. Anything not in this
# list and not exported is internal, and is written out as `Walnutpie.name` below.
using Walnutpie:
    InitChainConfig,
    adapt_step_sizes,
    dims,
    estimate_masses,
    init_chain_config,
    num_chains,
    random_positions,
    step_size
using Test
using Random: randn!
using StableRNGs: StableRNG
using Statistics: var

# gtest's EXPECT_DOUBLE_EQ admits a discrepancy of four units in the last place.
const ULP4 = 4eps(Float64)

const RNG_TEST_SEED = 12345
const TEST_SIZE = 100000

#####
##### Targets and handlers used across the tests
#####

# A standard normal target: log density -‖x‖²/2 with gradient -x.
function good_logp_grad!(∇, x)
    ∇ .= .-x
    return -sum(abs2, x) / 2
end

const THROWING_ERROR_MSG = "logp_grad failed"

throwing_logp_grad!(∇, x) = error(THROWING_ERROR_MSG)

mutable struct TattleTaleHandler
    latest_err::String
    latest_err_pos::Vector{Float64}
end

TattleTaleHandler() = TattleTaleHandler("", Float64[])

function Walnutpie.on_logp_exception(handler::TattleTaleHandler, θ, exception)
    handler.latest_err = sprint(showerror, exception)
    handler.latest_err_pos = collect(θ)
    return nothing
end

# The exact change in joint log density across one leapfrog step on a standard normal
# target started from the origin.
leapfrog_solution(ϵ, inv_M, ρ) = -ϵ^4 * inv_M^3 * ρ^2 / 8

# Degenerate values that a validated setting must reject, grouped by the bound in force.
inf_nan() = (NaN, Inf)
inf_nan_neg() = (inf_nan()..., -0.3)
inf_nan_neg_zero() = (inf_nan_neg()..., 0.0)
inf_nan_neg_zero_geq_one() = (inf_nan_neg_zero()..., 1.0, 1.5)
inf_nan_neg_zero_leq_one() = (inf_nan_neg_zero()..., 1.0, 0.99)

# The geometric mean of the step sizes `adapt_step_sizes` settles on across `num_seeds` runs
# in `dims` dimensions under a constant diagonal mass matrix.
function geometric_mean_adapted_step(dims, mass, base_seed, num_seeds)
    log_sum = 0.0
    for i in 0:(num_seeds-1)
        cfg = InitConfig(; num_chains=1, dims, mass=fill(float(mass), dims))
        log_sum += log(
            only(
                adapt_step_sizes(StableRNG(base_seed + i), good_logp_grad!, cfg).step_sizes,
            ),
        )
    end
    return exp(log_sum / num_seeds)
end

@testset verbose = true "Walnuts" begin

    #####
    ##### Random number generation
    #####

    @testset "random" begin
        @testset "generating advances the caller's generator" begin
            consumed = StableRNG(RNG_TEST_SEED)
            untouched = StableRNG(RNG_TEST_SEED)
            rand(consumed)
            @test rand(consumed) != rand(untouched)
        end

        @testset "uniform draws lie in [0, 1)" begin
            rng = StableRNG(RNG_TEST_SEED)
            @test all(1:TEST_SIZE) do _
                u = rand(rng)
                0.0 <= u < 1.0
            end
        end

        @testset "uniform draws have the right mean and variance" begin
            rng = StableRNG(RNG_TEST_SEED)
            total = 0.0
            total_sq = 0.0
            for _ in 1:TEST_SIZE
                u = rand(rng)
                total += u
                total_sq += u * u
            end
            mean = total / TEST_SIZE
            var = total_sq / TEST_SIZE - mean * mean
            @test mean ≈ 0.5 atol = 0.01        # ten standard errors
            @test var ≈ 1 / 12 atol = 0.01      # eight standard errors
        end

        @testset "binary draws are true half the time" begin
            rng = StableRNG(RNG_TEST_SEED)
            true_count = count(_ -> rand(rng, Bool), 1:TEST_SIZE)
            @test true_count / TEST_SIZE ≈ 0.5 atol = 0.02   # ten standard errors
        end

        @testset "returned normal draws have the right mean and variance" begin
            rng = StableRNG(RNG_TEST_SEED)
            v = randn(rng, TEST_SIZE)
            mean = sum(v) / TEST_SIZE
            var = sum(x -> abs2(x - mean), v) / (TEST_SIZE - 1)
            @test mean ≈ 0.0 atol = 0.05
            @test var ≈ 1.0 atol = 0.05
        end

        @testset "written normal draws have the right mean and variance" begin
            rng = StableRNG(RNG_TEST_SEED)
            v = Vector{Float64}(undef, TEST_SIZE)
            randn!(rng, v)
            mean = sum(v) / TEST_SIZE
            var = sum(x -> abs2(x - mean), v) / (TEST_SIZE - 1)
            @test mean ≈ 0.0 atol = 0.05
            @test var ≈ 1.0 atol = 0.05
        end
    end

    #####
    ##### logp_momentum
    #####

    @testset "logp_momentum" begin
        @testset "single element" begin
            @test Walnutpie.logp_momentum([2.0], [1.0]) ≈ -2.0 rtol = ULP4
        end

        @testset "zero momentum" begin
            @test Walnutpie.logp_momentum(zeros(4), fill(2.5, 4)) ≈ 0.0 atol = 0.0
        end

        @testset "unit mass, three dimensions" begin
            @test Walnutpie.logp_momentum([1.0, 2.0, 3.0], ones(3)) ≈
                  -0.5 * (1.0 + 4.0 + 9.0) rtol = ULP4
        end

        @testset "known value, and how it transforms" begin
            ρ = [2.0, 3.0]
            inv_M = [0.5, 2.0]
            @test Walnutpie.logp_momentum(ρ, inv_M) ≈ -10.0 rtol = ULP4
            @test Walnutpie.logp_momentum(ρ, inv_M) ≈ Walnutpie.logp_momentum(-ρ, inv_M) rtol =
                ULP4
            @test Walnutpie.logp_momentum(ρ, 2.5 * inv_M) ≈
                  2.5 * Walnutpie.logp_momentum(ρ, inv_M) rtol = ULP4
        end
    end

    #####
    ##### NoExceptLogpGrad
    #####

    @testset "NoExceptLogpGrad" begin
        @testset "a target that returns is passed through" begin
            handler = TattleTaleHandler()
            wrapped = Walnutpie.NoExceptLogpGrad(good_logp_grad!, handler)
            x = [1.0, 2.0, 3.0]
            ∇ = similar(x)
            logp = wrapped(∇, x)
            @test logp ≈ -sum(abs2, x) / 2 rtol = ULP4
            @test ∇ ≈ -x atol = 1e-10
        end

        @testset "a target that throws gives an impossible point" begin
            handler = TattleTaleHandler()
            wrapped = Walnutpie.NoExceptLogpGrad(throwing_logp_grad!, handler)
            x = [1.0, 2.0, 3.0]
            ∇ = ones(3)
            logp = wrapped(∇, x)
            @test logp == -Inf
            @test ∇ ≈ zeros(3) atol = 1e-10
            @test handler.latest_err == THROWING_ERROR_MSG
        end
    end

    #####
    ##### grad
    #####

    @testset "grad" begin
        @testset "returns the gradient" begin
            θ = [1.0, 2.0, 3.0]
            @test Walnutpie.grad(good_logp_grad!, θ) ≈ -θ atol = 1e-10
        end

        @testset "is zero at the origin" begin
            θ = zeros(4)
            @test Walnutpie.grad(good_logp_grad!, θ) ≈ zeros(4) atol = 1e-10
        end
    end

    #####
    ##### l2_rel_diff
    #####

    @testset "l2_rel_diff" begin
        @testset "single element" begin
            @test Walnutpie.l2_rel_diff([2.0], [1.0]) ≈ 1.0 rtol = ULP4
        end

        @testset "zero difference" begin
            a = [1.0, 2.0, 3.0]
            @test Walnutpie.l2_rel_diff(a, a) ≈ 0.0 atol = 0.0
        end

        @testset "known value" begin
            @test Walnutpie.l2_rel_diff([3.0, 5.0], [1.0, 4.0]) ≈ sqrt(2.0^2 + 0.25^2) rtol =
                ULP4
        end

        @testset "invariant to a common scaling" begin
            a = [2.0, 3.0, 4.0]
            b = [1.0, 2.0, 3.0]
            @test Walnutpie.l2_rel_diff(3.0 * a, 3.0 * b) ≈ Walnutpie.l2_rel_diff(a, b) rtol =
                ULP4
        end
    end

    #####
    ##### leapfrog_error
    #####

    @testset "leapfrog_error" begin
        @testset "a zero state loses nothing" begin
            @test Walnutpie.leapfrog_error(
                good_logp_grad!,
                zeros(3),
                zeros(3),
                ones(3),
                1.0,
            ) ≈ 0.0 atol = 0.0
        end

        @testset "from the origin, one dimension" begin
            ρ, inv_M, ϵ = 2.5, 0.3, 0.75
            @test Walnutpie.leapfrog_error(good_logp_grad!, [0.0], [ρ], [inv_M], ϵ) ≈
                  leapfrog_solution(ϵ, inv_M, ρ) atol = 1e-12
        end

        @testset "dimensions contribute additively" begin
            @test Walnutpie.leapfrog_error(
                good_logp_grad!,
                zeros(2),
                ones(2),
                ones(2),
                1.0,
            ) ≈ 2 * leapfrog_solution(1.0, 1.0, 1.0) atol = 1e-12
        end

        @testset "from the origin, non-unit inverse mass" begin
            ρ, inv_M, ϵ = 1.0, 0.25, 1.0
            @test Walnutpie.leapfrog_error(good_logp_grad!, [0.0], [ρ], [inv_M], ϵ) ≈
                  leapfrog_solution(ϵ, inv_M, ρ) atol = 1e-12
        end

        @testset "halving the step divides the error by sixteen" begin
            @test Walnutpie.leapfrog_error(good_logp_grad!, [0.0], [1.0], [1.0], 1.0) ≈
                  leapfrog_solution(1.0, 1.0, 1.0) atol = 1e-12
            @test Walnutpie.leapfrog_error(good_logp_grad!, [0.0], [1.0], [1.0], 0.5) ≈
                  leapfrog_solution(1.0, 1.0, 1.0) / 16 atol = 1e-12
        end

        @testset "one dimension, worked by hand" begin
            @test Walnutpie.leapfrog_error(good_logp_grad!, [1.0], [1.0], [1.0], 1.0) ≈
                  -5.0 / 32.0 atol = 1e-12
        end

        @testset "zero momentum, worked by hand" begin
            @test Walnutpie.leapfrog_error(good_logp_grad!, [1.0], [0.0], [1.0], 1.0) ≈
                  3.0 / 32.0 atol = 1e-12
        end

        @testset "a tiny step loses almost nothing" begin
            @test Walnutpie.leapfrog_error(
                good_logp_grad!,
                [1.0, -2.0],
                [0.5, 1.0],
                ones(2),
                1e-4,
            ) ≈ 0.0 atol = 1e-12
        end
    end

    #####
    ##### WelfordAccumulator
    #####

    @testset "WelfordAccumulator" begin
        @testset "starts with nothing observed" begin
            acc = Walnutpie.WelfordAccumulator()
            @test acc.n == 0
            @test acc.mean == 0.0
            @test isnan(Walnutpie.sample_variance(acc))
        end

        @testset "one observation leaves the variance undefined" begin
            acc = Walnutpie.WelfordAccumulator()
            Walnutpie.observe!(acc, 3.0)
            @test acc.n == 1
            @test acc.mean ≈ 3.0 rtol = ULP4
            @test isnan(Walnutpie.sample_variance(acc))
        end

        @testset "reports the count, the mean and the sample variance" begin
            xs = [2.0, 4.0, 4.0, 8.0]
            acc = Walnutpie.WelfordAccumulator()
            foreach(x -> Walnutpie.observe!(acc, x), xs)
            @test acc.n == length(xs)
            @test acc.mean ≈ sum(xs) / length(xs) rtol = ULP4
            @test Walnutpie.sample_variance(acc) ≈ var(xs) rtol = ULP4
        end

        @testset "stays accurate for values far larger than their spread" begin
            xs = [1.0, 2.0, 3.0, 4.0]
            acc = Walnutpie.WelfordAccumulator()
            foreach(x -> Walnutpie.observe!(acc, x), 1e9 .+ xs)
            @test Walnutpie.sample_variance(acc) ≈ var(xs) rtol = 1e-8
        end

        @testset "resetting returns it to its initial state" begin
            acc = Walnutpie.WelfordAccumulator()
            foreach(x -> Walnutpie.observe!(acc, x), [1.0, 2.0, 3.0])
            Walnutpie.reset!(acc)
            @test acc.n == 0
            @test acc.mean == 0.0
            @test acc.sum_sq_dev == 0.0
            @test isnan(Walnutpie.sample_variance(acc))
        end
    end

    #####
    ##### OnlineMoments
    #####

    @testset "OnlineMoments" begin
        @testset "reports a variance of one before any observation" begin
            m = Walnutpie.OnlineMoments(3)
            @test m.weight == 0.0
            @test m.mean == zeros(3)
            @test Walnutpie.variance(m) == ones(3)
        end

        @testset "an initial mean and variance carry the weight they are given" begin
            m = Walnutpie.OnlineMoments(;
                init_weight=4.0,
                init_mean=[1.0, 2.0],
                init_variance=[3.0, 4.0],
            )
            @test m.weight ≈ 4.0 rtol = ULP4
            @test m.mean ≈ [1.0, 2.0] rtol = ULP4
            @test Walnutpie.variance(m) ≈ [3.0, 4.0] rtol = ULP4
        end

        @testset "the initial weight must be positive and finite" begin
            for x in inf_nan_neg_zero()
                @test_throws "init_weight must be in (0, inf)." Walnutpie.OnlineMoments(;
                    init_weight=x,
                    init_mean=[1.0],
                    init_variance=[1.0],
                )
            end
        end

        @testset "the initial mean and variance must be the same size" begin
            @test_throws "init_mean and init_variance must be the same size: 2 and 1" Walnutpie.OnlineMoments(;
                init_weight=1.0,
                init_mean=[1.0, 2.0],
                init_variance=[1.0],
            )
        end

        @testset "the discount factor must lie in [0, 1]" begin
            for x in inf_nan_neg()
                m = Walnutpie.OnlineMoments(2)
                @test_throws "discount_factor must be in [0, 1]" Walnutpie.observe!(
                    m,
                    x,
                    [1.0, 2.0],
                )
            end
            m = Walnutpie.OnlineMoments(2)
            @test_throws "discount_factor must be in [0, 1]" Walnutpie.observe!(
                m,
                1.5,
                [1.0, 2.0],
            )
        end

        @testset "an observation from k updates ago is weighted by the discount to the k" begin
            discount = 0.7
            ys = [[1.0], [2.0], [5.0], [-3.0]]
            m = Walnutpie.OnlineMoments(1)
            foreach(y -> Walnutpie.observe!(m, discount, y), ys)
            weights = [discount^(length(ys) - n) for n in eachindex(ys)]
            values = only.(ys)
            expected_mean = sum(weights .* values) / sum(weights)
            expected_variance =
                sum(weights .* (values .- expected_mean) .^ 2) / sum(weights)
            @test m.weight ≈ sum(weights) rtol = ULP4
            @test only(m.mean) ≈ expected_mean rtol = 1e-12
            @test only(Walnutpie.variance(m)) ≈ expected_variance rtol = 1e-12
        end

        @testset "discounting nothing gives the plain running mean and variance" begin
            rng = StableRNG(RNG_TEST_SEED)
            ys = [randn(rng, 3) for _ in 1:20]
            m = Walnutpie.OnlineMoments(3)
            accs = [Walnutpie.WelfordAccumulator() for _ in 1:3]
            for y in ys
                Walnutpie.observe!(m, 1.0, y)
                for i in eachindex(accs, y)
                    Walnutpie.observe!(accs[i], y[i])
                end
            end
            @test m.weight ≈ length(ys) rtol = ULP4
            @test m.mean ≈ [acc.mean for acc in accs] rtol = 1e-12
            @test m.sum_sq_dev ≈ [acc.sum_sq_dev for acc in accs] rtol = 1e-12
        end

        @testset "forgetting everything leaves only the latest observation" begin
            m = Walnutpie.OnlineMoments(2)
            Walnutpie.observe!(m, 0.0, [1.0, 2.0])
            Walnutpie.observe!(m, 0.0, [7.0, -4.0])
            @test m.weight ≈ 1.0 rtol = ULP4
            @test m.mean ≈ [7.0, -4.0] rtol = ULP4
            @test Walnutpie.variance(m) ≈ zeros(2) atol = 1e-12
        end
    end

    #####
    ##### Adam
    #####

    @testset "Adam" begin
        # The tuning parameters `WarmupConfig` defaults to.
        adam(step_size) = Walnutpie.Adam(;
            step_size,
            accept_rate_target=0.8,
            learning_rate=0.05,
            gradient_decay=0.8,
            sq_gradient_decay=0.9,
            stabilization=1e-4,
            learn_rate_decay=0.5,
        )

        @testset "starts at the step size it was given" begin
            @test step_size(adam(0.5)) ≈ 0.5 rtol = ULP4
        end

        @testset "accepting more often than the target raises the step size" begin
            a = adam(0.5)
            Walnutpie.observe!(a, 0.99)
            @test step_size(a) > 0.5
        end

        @testset "accepting less often than the target lowers the step size" begin
            a = adam(0.5)
            Walnutpie.observe!(a, 0.01)
            @test step_size(a) < 0.5
        end

        @testset "observing the target rate leaves the step size alone" begin
            a = adam(0.5)
            Walnutpie.observe!(a, 0.8)
            @test step_size(a) ≈ 0.5 rtol = ULP4
        end

        @testset "reproduces the step sizes of the C++ optimiser" begin
            # Step sizes the `walnutpie` optimiser returns for these acceptance rates under
            # the tuning parameters above, started from a step size of 0.5.
            expected = [
                0.47561810902517943,
                0.46792762436003871,
                0.45996143488390506,
                0.45675795404344804,
                0.45445656435693116,
                0.44948741722573327,
                0.44514543309541593,
                0.44206629938437181,
                0.43772906172931214,
                0.43348682910243352,
            ]
            rates = [0.10, 0.95, 0.50, 0.99, 0.80, 0.20, 0.70, 0.85, 0.30, 0.60]
            a = adam(0.5)
            for (rate, want) in zip(rates, expected)
                Walnutpie.observe!(a, rate)
                @test step_size(a) ≈ want rtol = ULP4
            end
        end

        @testset "the decaying learning rate makes later moves smaller" begin
            a = adam(0.5)
            before_first = log(step_size(a))
            Walnutpie.observe!(a, 0.2)
            first_move = abs(log(step_size(a)) - before_first)
            for _ in 1:98
                Walnutpie.observe!(a, 0.2)
            end
            before_last = log(step_size(a))
            Walnutpie.observe!(a, 0.2)
            @test abs(log(step_size(a)) - before_last) < first_move
        end

        @testset "the warmup configuration supplies the tuning parameters" begin
            warmup = WarmupConfig(; step_accept_rate_target=0.65)
            a = Walnutpie.Adam(warmup, 0.25)
            @test step_size(a) ≈ 0.25 rtol = ULP4
            @test a.accept_rate_target ≈ 0.65 rtol = ULP4
            @test a.learning_rate ≈ warmup.step_learning_rate rtol = ULP4
            @test a.gradient_decay ≈ warmup.step_gradient_decay rtol = ULP4
            @test a.sq_gradient_decay ≈ warmup.step_sq_gradient_decay rtol = ULP4
            @test a.stabilization ≈ warmup.step_stabilization rtol = ULP4
            @test a.learn_rate_decay ≈ warmup.step_learn_rate_decay rtol = ULP4
        end
    end

    #####
    ##### InitChainConfig
    #####

    @testset "InitChainConfig" begin
        chain = InitChainConfig(; step_size=0.1, position=[1.0, 2.0], mass=[0.5, 1.5])

        @testset "stores the step size" begin
            @test chain.step_size ≈ 0.1 rtol = ULP4
        end

        @testset "stores the position" begin
            @test length(chain.position) == 2
            @test chain.position ≈ [1.0, 2.0] rtol = ULP4
        end

        @testset "stores the mass matrix" begin
            @test length(chain.mass) == 2
            @test chain.mass ≈ [0.5, 1.5] rtol = ULP4
        end
    end

    #####
    ##### InitConfig
    #####

    @testset "InitConfig" begin
        @testset "defaults to a tenth, the origin, and unit mass" begin
            cfg = InitConfig(; num_chains=3, dims=2)
            @test num_chains(cfg) == 3
            @test dims(cfg) == 2
            for c in 1:3
                @test cfg.step_sizes[c] ≈ 0.1 rtol = ULP4
                @test cfg.positions[c] == zeros(2)
                @test cfg.masses[c] == ones(2)
            end
        end

        @testset "no chains means no dimensions" begin
            cfg = InitConfig(; num_chains=0, dims=0)
            @test num_chains(cfg) == 0
            @test dims(cfg) == 0
            @test isempty(cfg.step_sizes)
            @test isempty(cfg.positions)
            @test isempty(cfg.masses)
        end

        @testset "one step size is shared by every chain" begin
            cfg = InitConfig(; num_chains=3, dims=2, step_size=0.5)
            @test length(cfg.step_sizes) == 3
            @test cfg.step_sizes ≈ fill(0.5, 3) rtol = ULP4
        end

        @testset "a shared step size must be finite and positive" begin
            for x in inf_nan_neg_zero()
                @test_throws "step size must be finite and > 0" InitConfig(;
                    num_chains=3,
                    dims=2,
                    step_size=x,
                )
            end
        end

        @testset "a step size per chain is taken in order" begin
            sizes = [0.1, 0.2, 0.3]
            cfg = InitConfig(; num_chains=3, dims=2, step_size=sizes)
            @test cfg.step_sizes ≈ sizes rtol = ULP4
        end

        @testset "there must be one step size per chain" begin
            @test_throws "step_sizes size must match num_chains" InitConfig(;
                num_chains=3,
                dims=2,
                step_size=[0.1, 0.2],
            )
        end

        @testset "every step size in a collection must be finite and positive" begin
            for x in inf_nan_neg_zero()
                @test_throws "step_size must be finite and > 0" InitConfig(;
                    num_chains=3,
                    dims=2,
                    step_size=[0.1, x, 0.3],
                )
                @test_throws "step_size must be finite and > 0" InitConfig(;
                    num_chains=3,
                    dims=2,
                    step_size=[x, 0.1, 0.3],
                )
            end
        end

        @testset "one position is shared by every chain" begin
            position = [3.0, 4.0]
            cfg = InitConfig(; num_chains=3, dims=2, position)
            for c in 1:3
                @test cfg.positions[c] ≈ position atol = 1e-10
            end
        end

        @testset "a shared position must have one coordinate per dimension" begin
            @test_throws "position size must match dims" InitConfig(;
                num_chains=3,
                dims=2,
                position=[1.0, 2.0, 3.0],
            )
        end

        @testset "a shared position must be finite" begin
            for x in inf_nan()
                @test_throws "position must be finite" InitConfig(;
                    num_chains=3,
                    dims=2,
                    position=[1.0, x],
                )
            end
        end

        @testset "a position per chain is taken in order" begin
            positions = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            cfg = InitConfig(; num_chains=3, dims=2, position=positions)
            for c in 1:3
                @test cfg.positions[c] ≈ positions[c] atol = 1e-10
            end
        end

        @testset "two chains take a position each" begin
            positions = [[1.0, 2.0], [3.0, 4.0]]
            cfg = InitConfig(; num_chains=2, dims=2, position=positions)
            for c in 1:2
                @test cfg.positions[c] ≈ positions[c] atol = 1e-10
            end
        end

        @testset "there must be one position per chain" begin
            @test_throws "positions size must match num_chains" InitConfig(;
                num_chains=3,
                dims=2,
                position=[zeros(2) for _ in 1:2],
            )
        end

        @testset "every position in a collection must match the dimensions" begin
            @test_throws "position size must match dims" InitConfig(;
                num_chains=3,
                dims=2,
                position=[zeros(3) for _ in 1:3],
            )
        end

        @testset "every position in a collection must be finite" begin
            for x in inf_nan()
                middle = [zeros(2) for _ in 1:3]
                middle[2][1] = x
                @test_throws "positions must be finite" InitConfig(;
                    num_chains=3,
                    dims=2,
                    position=middle,
                )
                first_chain = [zeros(2) for _ in 1:3]
                first_chain[1][2] = x
                @test_throws "positions must be finite" InitConfig(;
                    num_chains=3,
                    dims=2,
                    position=first_chain,
                )
            end
        end

        @testset "one mass matrix is shared by every chain" begin
            mass = [2.0, 3.0]
            cfg = InitConfig(; num_chains=3, dims=2, mass)
            for c in 1:3
                @test cfg.masses[c] ≈ mass atol = 1e-10
            end
        end

        @testset "a shared mass matrix must have one entry per dimension" begin
            @test_throws "masses size must match dims" InitConfig(;
                num_chains=3,
                dims=2,
                mass=[1.0, 2.0, 3.0],
            )
        end

        @testset "a shared mass matrix must be finite and positive" begin
            for x in inf_nan_neg_zero()
                @test_throws "masses must be finite and > 0" InitConfig(;
                    num_chains=3,
                    dims=2,
                    mass=[1.0, x],
                )
                @test_throws "masses must be finite and > 0" InitConfig(;
                    num_chains=3,
                    dims=2,
                    mass=[x, 2.9],
                )
            end
        end

        @testset "a mass matrix per chain is taken in order" begin
            masses = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            cfg = InitConfig(; num_chains=3, dims=2, mass=masses)
            for c in 1:3
                @test cfg.masses[c] ≈ masses[c] atol = 1e-10
            end
        end

        @testset "two chains take a mass matrix each" begin
            masses = [[1.0, 2.0], [3.0, 4.0]]
            cfg = InitConfig(; num_chains=2, dims=2, mass=masses)
            for c in 1:2
                @test cfg.masses[c] ≈ masses[c] atol = 1e-10
            end
        end

        @testset "there must be one mass matrix per chain" begin
            @test_throws "masses size must match num_chains" InitConfig(;
                num_chains=3,
                dims=2,
                mass=[ones(2) for _ in 1:2],
            )
        end

        @testset "every mass matrix in a collection must match the dimensions" begin
            @test_throws "all masses size must match dims" InitConfig(;
                num_chains=3,
                dims=2,
                mass=[ones(3) for _ in 1:3],
            )
        end

        @testset "every mass matrix in a collection must be finite and positive" begin
            for x in inf_nan_neg_zero()
                middle = [ones(2) for _ in 1:3]
                middle[2][1] = x
                @test_throws "masses must be finite and > 0" InitConfig(;
                    num_chains=3,
                    dims=2,
                    mass=middle,
                )
                last_chain = [ones(2) for _ in 1:3]
                last_chain[3][2] = x
                @test_throws "masses must be finite and > 0" InitConfig(;
                    num_chains=3,
                    dims=2,
                    mass=last_chain,
                )
            end
        end

        @testset "the dimension count may not be negative" begin
            @test_throws "dims must be in {0, 1, ... }" InitConfig(; num_chains=1, dims=-1)
            @test_throws "num_chains must be in {0, 1, ... }" InitConfig(;
                num_chains=-1,
                dims=2,
            )
        end

        @testset "a chain's own settings can be read back" begin
            cfg = InitConfig(;
                num_chains=3,
                dims=2,
                step_size=0.25,
                position=[1.0, 2.0],
                mass=[3.0, 4.0],
            )
            chain = init_chain_config(cfg, 1)
            @test chain.step_size ≈ 0.25 rtol = ULP4
            @test chain.position ≈ [1.0, 2.0] atol = 1e-10
            @test chain.mass ≈ [3.0, 4.0] atol = 1e-10
        end
    end

    #####
    ##### random_positions
    #####

    @testset "random_positions" begin
        @testset "draws one position per chain over every dimension" begin
            positions = random_positions(StableRNG(RNG_TEST_SEED), 3, 4)
            @test length(positions) == 3
            @test all(p -> length(p) == 4, positions)
        end

        @testset "the scale multiplies the draw" begin
            unit = random_positions(StableRNG(RNG_TEST_SEED), 2, 3; scale=1.0)
            doubled = random_positions(StableRNG(RNG_TEST_SEED), 2, 3; scale=2.0)
            for c in 1:2
                @test doubled[c] ≈ 2.0 .* unit[c] atol = 1e-10
            end
        end

        @testset "the scale must be finite and positive" begin
            for x in inf_nan_neg_zero()
                @test_throws "init_scale must be finite and > 0" random_positions(
                    StableRNG(RNG_TEST_SEED),
                    3,
                    2;
                    scale=x,
                )
            end
        end
    end

    #####
    ##### estimate_masses
    #####

    @testset "estimate_masses" begin
        @testset "gives one mass matrix per position, over every dimension" begin
            positions = fill([1.0, 2.0], 3)
            masses = estimate_masses(good_logp_grad!, positions; smoothing=0.5)
            @test length(masses) == 3
            @test all(m -> length(m) == 2, masses)
        end

        @testset "matches a hand calculation" begin
            # At [1, 2] the standard normal has gradient [-1, -2], so with half the weight on
            # the absolute gradient and half on one the mass is [1, 1.5].
            smoothing = 0.5
            masses = estimate_masses(good_logp_grad!, [[1.0, 2.0]]; smoothing)
            expected = (1 - smoothing) .* [1.0, 2.0] .+ smoothing
            @test only(masses) ≈ expected atol = 1e-10
        end

        @testset "the smoothing must lie strictly between zero and one" begin
            positions = fill([100.0, -50.0], 2)
            @test_throws "mass_smoothing must be in (0, 1)" estimate_masses(
                good_logp_grad!,
                positions;
                smoothing=1.0,
            )
            for x in inf_nan_neg()
                @test_throws "mass_smoothing must be in (0, 1)" estimate_masses(
                    good_logp_grad!,
                    positions;
                    smoothing=x,
                )
            end
        end

        @testset "an empty run has nothing to average" begin
            @test isempty(
                estimate_masses(
                    good_logp_grad!,
                    Vector{Float64}[];
                    smoothing=0.5,
                    average=true,
                ),
            )
        end

        @testset "averaging gives every chain the geometric mean" begin
            for size in (1, 2, 9, 32)
                positions = random_positions(StableRNG(139872), size, size)
                per_chain = estimate_masses(good_logp_grad!, positions; smoothing=0.01)
                averaged = estimate_masses(
                    good_logp_grad!,
                    positions;
                    smoothing=0.01,
                    average=true,
                )
                geometric_mean = exp.(sum(m -> log.(m), per_chain) ./ size)
                @test length(averaged) == size
                for mass in averaged
                    @test mass ≈ geometric_mean atol = 1e-10
                end
            end
        end
    end

    #####
    ##### adapt_step_sizes
    #####

    @testset "adapt_step_sizes" begin
        @testset "a tiny and a huge starting step meet within a factor of two" begin
            from_tiny = adapt_step_sizes(
                StableRNG(287456),
                good_logp_grad!,
                InitConfig(; num_chains=1, dims=3, step_size=1e-4),
            )
            from_huge = adapt_step_sizes(
                StableRNG(287456),
                good_logp_grad!,
                InitConfig(; num_chains=1, dims=3, step_size=100.0),
            )
            @test log2(only(from_tiny.step_sizes)) ≈ log2(only(from_huge.step_sizes)) atol =
                1.01
        end

        @testset "the step shrinks as the dimension grows" begin
            # The best leapfrog step falls off as the fourth root of the dimension.
            h1 = geometric_mean_adapted_step(1, 1.0, 185737, 256)
            h100 = geometric_mean_adapted_step(100, 1.0, 185737, 64)
            h10000 = geometric_mean_adapted_step(10000, 1.0, 185737, 16)
            @test h1 > h100 > h10000
            @test log(h100 / h10000) ≈ 0.25 * log(100.0) atol = 0.5
            @test h1 / h10000 > 4.0
        end

        @testset "the step grows with the square root of the mass" begin
            unit = geometric_mean_adapted_step(10, 1.0, 285222, 1024)
            heavy = geometric_mean_adapted_step(10, 100.0, 285222, 1024)
            light = geometric_mean_adapted_step(10, 0.01, 285222, 1024)
            @test heavy / unit ≈ 10 atol = 1
            @test unit / light ≈ 10 atol = 1
        end
    end

    #####
    ##### WarmupConfig
    #####

    @testset "WarmupConfig" begin
        @testset "the defaults" begin
            cfg = WarmupConfig()
            @test cfg.min_iter == 50
            @test cfg.max_iter == 1000
            @test cfg.step_size_converge_tol ≈ 0.1 rtol = ULP4
            @test cfg.mass_converge_tol ≈ 1.0 rtol = ULP4
            @test cfg.mass_init_count ≈ 4.0 rtol = ULP4
            @test cfg.mass_additive_smoothing ≈ 1e-5 rtol = ULP4
            @test cfg.max_macro_steps_target ≈ 15.0 rtol = ULP4
            @test cfg.step_accept_rate_target ≈ 0.8 rtol = ULP4
            @test cfg.step_learning_rate ≈ 0.05 rtol = ULP4
            @test cfg.step_gradient_decay ≈ 0.8 rtol = ULP4
            @test cfg.step_sq_gradient_decay ≈ 0.9 rtol = ULP4
            @test cfg.step_stabilization ≈ 1e-4 rtol = ULP4
            @test cfg.step_learn_rate_decay ≈ 0.5 rtol = ULP4
            @test cfg.publish_stride == 5
            @test cfg.yield_period == 32
        end

        @testset "the iteration bounds are taken as given" begin
            cfg = WarmupConfig(; min_iter=10, max_iter=500)
            @test cfg.min_iter == 10
            @test cfg.max_iter == 500
        end

        @testset "the iteration bounds may be equal" begin
            cfg = WarmupConfig(; min_iter=100, max_iter=100)
            @test cfg.min_iter == 100
            @test cfg.max_iter == 100
        end

        @testset "the minimum may not exceed the maximum" begin
            @test_throws "min_iter must be <= max_iter" WarmupConfig(;
                min_iter=500,
                max_iter=10,
            )
        end

        @testset "step_size_converge_tol is taken as given" begin
            @test WarmupConfig(; step_size_converge_tol=0.05).step_size_converge_tol ≈ 0.05 rtol =
                ULP4
        end

        @testset "step_size_converge_tol must be finite and positive" begin
            for x in inf_nan_neg_zero()
                @test_throws "step_size_converge_tol must be finite and > 0" WarmupConfig(;
                    step_size_converge_tol=x,
                )
            end
        end

        @testset "mass_converge_tol is taken as given" begin
            @test WarmupConfig(; mass_converge_tol=0.5).mass_converge_tol ≈ 0.5 rtol = ULP4
        end

        @testset "mass_converge_tol must be finite and positive" begin
            for x in inf_nan_neg_zero()
                @test_throws "mass_converge_tol must be finite and > 0" WarmupConfig(;
                    mass_converge_tol=x,
                )
            end
        end

        @testset "mass_init_count is taken as given" begin
            @test WarmupConfig(; mass_init_count=10.0).mass_init_count ≈ 10.0 rtol = ULP4
        end

        @testset "mass_init_count must be finite and positive" begin
            for x in inf_nan_neg_zero()
                @test_throws "mass_init_count must be finite and > 0" WarmupConfig(;
                    mass_init_count=x,
                )
            end
        end

        @testset "mass_additive_smoothing is taken as given" begin
            @test WarmupConfig(; mass_additive_smoothing=0.01).mass_additive_smoothing ≈
                  0.01 rtol = ULP4
        end

        @testset "mass_additive_smoothing must be finite and positive" begin
            for x in inf_nan_neg_zero()
                @test_throws "mass_additive_smoothing must be finite and > 0" WarmupConfig(;
                    mass_additive_smoothing=x,
                )
            end
        end

        @testset "max_macro_steps_target is taken as given" begin
            @test WarmupConfig(; max_macro_steps_target=20.0).max_macro_steps_target ≈ 20.0 rtol =
                ULP4
        end

        @testset "max_macro_steps_target must be finite and positive" begin
            for x in inf_nan_neg_zero()
                @test_throws "max_macro_steps_target must be finite and > 0" WarmupConfig(;
                    max_macro_steps_target=x,
                )
            end
        end

        @testset "step_accept_rate_target is taken as given" begin
            @test WarmupConfig(; step_accept_rate_target=0.65).step_accept_rate_target ≈
                  0.65 rtol = ULP4
        end

        @testset "step_accept_rate_target must be a probability" begin
            for x in inf_nan_neg_zero_geq_one()
                @test_throws "step_accept_rate_target must be in (0, 1)" WarmupConfig(;
                    step_accept_rate_target=x,
                )
            end
        end

        @testset "step_learning_rate is taken as given" begin
            @test WarmupConfig(; step_learning_rate=0.1).step_learning_rate ≈ 0.1 rtol =
                ULP4
        end

        @testset "step_learning_rate must be finite and positive" begin
            for x in inf_nan_neg_zero()
                @test_throws "step_learning_rate must be finite and > 0" WarmupConfig(;
                    step_learning_rate=x,
                )
            end
        end

        @testset "step_gradient_decay is taken as given" begin
            @test WarmupConfig(; step_gradient_decay=0.9).step_gradient_decay ≈ 0.9 rtol =
                ULP4
        end

        @testset "step_gradient_decay must be a probability" begin
            for x in inf_nan_neg_zero_geq_one()
                @test_throws "step_gradient_decay must be in (0, 1)" WarmupConfig(;
                    step_gradient_decay=x,
                )
            end
        end

        @testset "step_sq_gradient_decay is taken as given" begin
            @test WarmupConfig(; step_sq_gradient_decay=0.95).step_sq_gradient_decay ≈ 0.95 rtol =
                ULP4
        end

        @testset "step_sq_gradient_decay must be a probability" begin
            for x in inf_nan_neg_zero_geq_one()
                @test_throws "step_sq_gradient_decay must be in (0, 1)" WarmupConfig(;
                    step_sq_gradient_decay=x,
                )
            end
        end

        @testset "step_stabilization is taken as given" begin
            @test WarmupConfig(; step_stabilization=1e-3).step_stabilization ≈ 1e-3 rtol =
                ULP4
        end

        @testset "step_stabilization must be finite and positive" begin
            for x in inf_nan_neg_zero()
                @test_throws "step_stabilization must be finite and > 0" WarmupConfig(;
                    step_stabilization=x,
                )
            end
        end

        @testset "step_learn_rate_decay is taken as given" begin
            @test WarmupConfig(; step_learn_rate_decay=0.75).step_learn_rate_decay ≈ 0.75 rtol =
                ULP4
        end

        @testset "step_learn_rate_decay must be a probability" begin
            for x in inf_nan_neg_zero_geq_one()
                @test_throws "step_learn_rate_decay must be in (0, 1)" WarmupConfig(;
                    step_learn_rate_decay=x,
                )
            end
        end

        @testset "publish_stride is taken as given" begin
            @test WarmupConfig(; publish_stride=10).publish_stride == 10
        end

        @testset "publish_stride must be a positive count" begin
            @test_throws "publish_stride must be in {1, 2, ... }" WarmupConfig(;
                publish_stride=0,
            )
        end

        @testset "yield_period is taken as given" begin
            @test WarmupConfig(; yield_period=64).yield_period == 64
        end

        @testset "yield_period must be a positive count" begin
            @test_throws "yield_period must be in {1, 2, ... }" WarmupConfig(;
                yield_period=0,
            )
        end

        @testset "every setting can be given at once" begin
            cfg = WarmupConfig(;
                min_iter=25,
                max_iter=200,
                step_size_converge_tol=0.05,
                mass_converge_tol=0.5,
                mass_init_count=2.0,
                mass_additive_smoothing=1e-4,
                max_macro_steps_target=10.0,
                step_accept_rate_target=0.75,
                step_learning_rate=0.1,
                step_gradient_decay=0.85,
                step_sq_gradient_decay=0.95,
                step_stabilization=1e-3,
                step_learn_rate_decay=0.6,
                publish_stride=2,
                yield_period=16,
            )
            @test cfg.min_iter == 25
            @test cfg.max_iter == 200
            @test cfg.step_size_converge_tol ≈ 0.05 rtol = ULP4
            @test cfg.mass_converge_tol ≈ 0.5 rtol = ULP4
            @test cfg.mass_init_count ≈ 2.0 rtol = ULP4
            @test cfg.mass_additive_smoothing ≈ 1e-4 rtol = ULP4
            @test cfg.max_macro_steps_target ≈ 10.0 rtol = ULP4
            @test cfg.step_accept_rate_target ≈ 0.75 rtol = ULP4
            @test cfg.step_learning_rate ≈ 0.1 rtol = ULP4
            @test cfg.step_gradient_decay ≈ 0.85 rtol = ULP4
            @test cfg.step_sq_gradient_decay ≈ 0.95 rtol = ULP4
            @test cfg.step_stabilization ≈ 1e-3 rtol = ULP4
            @test cfg.step_learn_rate_decay ≈ 0.6 rtol = ULP4
            @test cfg.publish_stride == 2
            @test cfg.yield_period == 16
        end

        @testset "an iteration count may not be negative" begin
            @test_throws "min_iter must be in {0, 1, ... }" WarmupConfig(; min_iter=-1)
            @test_throws "max_iter must be in {0, 1, ... }" WarmupConfig(;
                min_iter=0,
                max_iter=-1,
            )
        end
    end

    #####
    ##### SamplingConfig
    #####

    @testset "SamplingConfig" begin
        @testset "the defaults" begin
            cfg = SamplingConfig()
            @test cfg.min_iter == 50
            @test cfg.max_iter == 1000
            @test cfg.max_trajectory_doublings == 5
            @test cfg.max_step_halvings == 5
            @test cfg.max_hamiltonian_error ≈ 0.5 rtol = ULP4
            @test cfg.min_micro_steps == 1
            @test cfg.rhat_converge_tol ≈ 1.01 rtol = ULP4
        end

        @testset "the iteration bounds are taken as given" begin
            cfg = SamplingConfig(; min_iter=10, max_iter=500)
            @test cfg.min_iter == 10
            @test cfg.max_iter == 500
        end

        @testset "the iteration bounds may be equal" begin
            cfg = SamplingConfig(; min_iter=100, max_iter=100)
            @test cfg.min_iter == 100
            @test cfg.max_iter == 100
        end

        @testset "the minimum may not exceed the maximum" begin
            @test_throws "min_iter must be <= max_iter" SamplingConfig(;
                min_iter=500,
                max_iter=10,
            )
        end

        @testset "max_trajectory_doublings is taken as given" begin
            @test SamplingConfig(; max_trajectory_doublings=10).max_trajectory_doublings ==
                  10
        end

        @testset "max_trajectory_doublings may be zero" begin
            @test SamplingConfig(; max_trajectory_doublings=0).max_trajectory_doublings == 0
        end

        @testset "max_step_halvings is taken as given" begin
            @test SamplingConfig(; max_step_halvings=8).max_step_halvings == 8
        end

        @testset "max_step_halvings may be zero" begin
            @test SamplingConfig(; max_step_halvings=0).max_step_halvings == 0
        end

        @testset "max_hamiltonian_error is taken as given" begin
            @test SamplingConfig(; max_hamiltonian_error=1.0).max_hamiltonian_error ≈ 1.0 rtol =
                ULP4
        end

        @testset "max_hamiltonian_error must be finite and positive" begin
            for x in inf_nan_neg_zero()
                @test_throws "max_hamiltonian_error must be finite and > 0" SamplingConfig(;
                    max_hamiltonian_error=x,
                )
            end
        end

        @testset "min_micro_steps is taken as given" begin
            @test SamplingConfig(; min_micro_steps=4).min_micro_steps == 4
        end

        @testset "min_micro_steps must be a positive count" begin
            @test_throws "min_micro_steps must be in {1, 2, ... }" SamplingConfig(;
                min_micro_steps=0,
            )
        end

        @testset "rhat_converge_tol is taken as given" begin
            @test SamplingConfig(; rhat_converge_tol=1.05).rhat_converge_tol ≈ 1.05 rtol =
                ULP4
        end

        @testset "rhat_converge_tol must be finite and greater than one" begin
            for x in inf_nan_neg_zero_leq_one()
                @test_throws "rhat_converge_tol must be finite and > 1" SamplingConfig(;
                    rhat_converge_tol=x,
                )
            end
        end

        @testset "every setting can be given at once" begin
            cfg = SamplingConfig(;
                min_iter=25,
                max_iter=200,
                max_trajectory_doublings=8,
                max_step_halvings=3,
                max_hamiltonian_error=1.0,
                min_micro_steps=2,
                rhat_converge_tol=1.05,
            )
            @test cfg.min_iter == 25
            @test cfg.max_iter == 200
            @test cfg.max_trajectory_doublings == 8
            @test cfg.max_step_halvings == 3
            @test cfg.max_hamiltonian_error ≈ 1.0 rtol = ULP4
            @test cfg.min_micro_steps == 2
            @test cfg.rhat_converge_tol ≈ 1.05 rtol = ULP4
        end

        @testset "a count of doublings or halvings may not be negative" begin
            @test_throws "max_trajectory_doublings must be in {0, 1, ... }" SamplingConfig(;
                max_trajectory_doublings=-1,
            )
            @test_throws "max_step_halvings must be in {0, 1, ... }" SamplingConfig(;
                max_step_halvings=-1,
            )
        end
    end

    #####
    ##### WalnutsConfig
    #####

    @testset "WalnutsConfig" begin
        @testset "keeps the three configurations apart" begin
            cfg = WalnutsConfig(;
                init=InitConfig(; num_chains=2, dims=3, step_size=0.25),
                warmup=WarmupConfig(; min_iter=10, max_iter=200),
                sampling=SamplingConfig(; min_iter=5, max_iter=100),
            )
            @test cfg.warmup.min_iter == 10
            @test cfg.warmup.max_iter == 200
            @test cfg.sampling.min_iter == 5
            @test cfg.sampling.max_iter == 100
            @test num_chains(cfg.init) == 2
            @test cfg.init.step_sizes ≈ fill(0.25, 2) rtol = ULP4
        end

        @testset "warmup and sampling default to their own defaults" begin
            cfg = WalnutsConfig(; init=InitConfig(; num_chains=1, dims=1))
            @test cfg.warmup == WarmupConfig()
            @test cfg.sampling == SamplingConfig()
        end
    end
end
