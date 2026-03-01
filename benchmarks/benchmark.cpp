#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "bootstrap_runner.hpp"
#include "ntt.h"
#include "ntt_plan.hpp"

#define REPETITIONS 3

namespace {

template <typename Transform>
CoeffPoly<Transform> DeterministicInput() {
    std::vector<uint64_t> data(Transform::N, 0);
    for (size_t i = 0; i < Transform::N; ++i) {
        data[i] = ((i * 17ULL) + 13ULL) % Transform::p;
    }
    return CoeffPoly<Transform>::FromUnsigned(data);
}

template <typename Transform, Backend B>
static void BM_PlanForward(benchmark::State &state) {
    using Plan = CanonicalNttPlan<Transform, B>;
    Plan plan;
    typename Plan::Workspace workspace;
    const auto coeff = DeterministicInput<Transform>();

    for (auto _ : state) {
        const auto eval = plan.forward(coeff, workspace);
        benchmark::DoNotOptimize(eval.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * Transform::N));
}

bdf17::ExperimentConfig BenchmarkConfig() {
    bdf17::ExperimentConfig config{};
    config.seed = 0xBDF1700BULL;
    config.num_trials = 1;
    config.expcrt_variant = bdf17::ExpCrtVariant::TensorTrick;
    config.profile_name = "toy-equivalence";
    config.lut_name = "lowbit";
    config.enable_lwe_dim_reduction_override = false;
    config.has_enable_lwe_dim_reduction_override = false;
    config.emit_json = false;
    return config;
}

template <typename Params>
std::pair<std::vector<int64_t>, int64_t> BuildAccumulatorInput(bdf17::BootstrapRunner<Params> &runner) {
    uint64_t packed_plain = 0;
    std::string bits_le;
    bdf17::StageStats encrypt_stats;
    bdf17::StageStats pack_stats;
    bdf17::StageStats keyswitch_stats;
    bdf17::StageStats modswitch_stats;

    auto bit_ciphertexts = runner.EncryptInputBits(packed_plain, bits_le, encrypt_stats);
    auto packed_frontend = runner.PackBits(bit_ciphertexts, pack_stats);
    auto packed_for_accumulator = runner.FrontendKeySwitchIfNeeded(packed_frontend, keyswitch_stats);
    auto packed_internal = runner.FrontendModSwitchIn(packed_for_accumulator, modswitch_stats);

    std::vector<int64_t> a(Params::kLweAccumulatorDimension, 0);
    for (size_t i = 0; i < Params::kLweAccumulatorDimension; ++i) {
        a[i] = static_cast<int64_t>(packed_internal.a[i]);
    }
    return {std::move(a), static_cast<int64_t>(packed_internal.b)};
}

using Primitive769 = NTT<1152921504602791681ULL, 11ULL, 769, 11>;
using Primitive12289 = NTT<1152921504107839489ULL, 19ULL, 12289, 11>;

} // namespace

static void BM_PlanForwardPrimitive769Auto(benchmark::State &state) {
    BM_PlanForward<Primitive769, Backend::Auto>(state);
}

BENCHMARK(BM_PlanForwardPrimitive769Auto)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_PlanForwardPrimitive12289Auto(benchmark::State &state) {
    BM_PlanForward<Primitive12289, Backend::Auto>(state);
}

BENCHMARK(BM_PlanForwardPrimitive12289Auto)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_PlanForwardPrimitive12289Scalar(benchmark::State &state) {
    BM_PlanForward<Primitive12289, Backend::Scalar>(state);
}

BENCHMARK(BM_PlanForwardPrimitive12289Scalar)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_BootstrapStageAccumulatorPToy(benchmark::State &state) {
    using Params = bdf17::ToyEquivalenceProfile;
    static bdf17::BootstrapRunner<Params> runner(BenchmarkConfig());
    static const auto input = BuildAccumulatorInput<Params>(runner);

    for (auto _ : state) {
        bdf17::StageStats stats;
        auto ct_p = runner.RunAccumulatorP(input.first, input.second, stats);
        benchmark::DoNotOptimize(ct_p);
        benchmark::DoNotOptimize(stats);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_BootstrapStageAccumulatorPToy)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_BootstrapTrialToy(benchmark::State &state) {
    using Params = bdf17::ToyEquivalenceProfile;
    static bdf17::BootstrapRunner<Params> runner(BenchmarkConfig());
    size_t trial_index = 0;

    for (auto _ : state) {
        bdf17::BootstrapMetrics metrics;
        auto result = runner.RunTrial({trial_index}, &metrics);
        ++trial_index;
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(metrics);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_BootstrapTrialToy)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);
