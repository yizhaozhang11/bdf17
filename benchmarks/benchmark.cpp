#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "bootstrap_runner.hpp"
#include "ntt.h"
#include "ntt_backend.hpp"
#include "ntt_plan.hpp"

#define REPETITIONS 3

namespace {

constexpr uint64_t kBenchmarkSeedToy = 0xBDF1700BULL;

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

bdf17::ExperimentConfig BenchmarkConfigToy() {
    bdf17::ExperimentConfig config{};
    config.seed = kBenchmarkSeedToy;
    config.num_trials = 1;
    config.expcrt_variant = bdf17::ExpCrtVariant::TensorTrick;
    config.profile_name = "toy-equivalence";
    config.lut_name = "lowbit";
    config.enable_lwe_dim_reduction_override = false;
    config.has_enable_lwe_dim_reduction_override = false;
    config.emit_json = false;
    return config;
}

const char *BackendName(const Backend backend) {
    switch (backend) {
        case Backend::Auto:
            return "auto";
        case Backend::Scalar:
            return "scalar";
        case Backend::Avx2:
            return "avx2";
        case Backend::Avx512:
            return "avx512";
    }
    return "unknown";
}

double BackendCode(const Backend backend) {
    switch (backend) {
        case Backend::Scalar:
            return 0.0;
        case Backend::Avx2:
            return 1.0;
        case Backend::Avx512:
            return 2.0;
        case Backend::Auto:
            return 3.0;
    }
    return -1.0;
}

template <typename Params>
void SetPipelineMetadata(
    benchmark::State &state,
    const bdf17::ExperimentConfig &config,
    const char *setup_mode,
    const char *benchmark_kind,
    const char *stage_name) {
    const Backend resolved_backend = ResolveAutoBackend();
    state.counters["seed"] = static_cast<double>(config.seed);
    state.counters["backend_id"] = BackendCode(resolved_backend);
    state.counters["p_degree"] = static_cast<double>(Params::EvalP::N);
    state.counters["q_degree"] = static_cast<double>(Params::EvalQ::N);
    state.counters["tensor_dimension"] = static_cast<double>(Params::kTensorDimension);
    state.counters["plain_modulus"] = static_cast<double>(Params::kPlainModulus);

    std::string label;
    label.reserve(192);
    label += "profile=";
    label += Params::kProfileName;
    label += ",seed=";
    label += std::to_string(config.seed);
    label += ",backend=";
    label += BackendName(resolved_backend);
    label += ",lut=";
    label += config.lut_name;
    label += ",setup=";
    label += setup_mode;
    label += ",kind=";
    label += benchmark_kind;
    label += ",stage=";
    label += stage_name;
    state.SetLabel(label);
}

void AddStageStatsCounters(benchmark::State &state, const bdf17::StageStats &stats, const size_t iterations) {
    const double denom = iterations == 0 ? 1.0 : static_cast<double>(iterations);
    state.counters["stage_ms_avg"] = stats.ms / denom;
    state.counters["stage_forward_ntt_avg"] = static_cast<double>(stats.forward_ntt) / denom;
    state.counters["stage_inverse_ntt_avg"] = static_cast<double>(stats.inverse_ntt) / denom;
    state.counters["stage_galois_avg"] = static_cast<double>(stats.galois) / denom;
    state.counters["stage_keyswitch_avg"] = static_cast<double>(stats.keyswitch) / denom;
    state.counters["stage_extmult_avg"] = static_cast<double>(stats.extmult) / denom;
    state.counters["stage_trace_calls_avg"] = static_cast<double>(stats.trace_calls) / denom;
    state.counters["stage_approx_bytes"] = static_cast<double>(stats.approx_bytes) / denom;
}

void AddTrialMetricsCounters(benchmark::State &state, const bdf17::BootstrapMetrics &metrics, const size_t iterations) {
    const double denom = iterations == 0 ? 1.0 : static_cast<double>(iterations);

    state.counters["total_ms_avg"] = metrics.total.ms / denom;
    state.counters["encrypt_ms_avg"] = metrics.encrypt_bits.ms / denom;
    state.counters["pack_ms_avg"] = metrics.pack_bits.ms / denom;
    state.counters["keyswitch_ms_avg"] = metrics.lwe_keyswitch.ms / denom;
    state.counters["modswitch_in_ms_avg"] = metrics.modswitch_in.ms / denom;
    state.counters["accum_p_ms_avg"] = metrics.accum_p.ms / denom;
    state.counters["accum_q_ms_avg"] = metrics.accum_q.ms / denom;
    state.counters["expcrt_ms_avg"] = metrics.expcrt.ms / denom;
    state.counters["fun_extract_ms_avg"] = metrics.fun_extract.ms / denom;
    state.counters["modswitch_out_ms_avg"] = metrics.modswitch_out.ms / denom;
    state.counters["decode_ms_avg"] = metrics.decode_and_check.ms / denom;

    state.counters["approx_lwe_ksk_bytes"] = static_cast<double>(metrics.lwe_keyswitch.approx_bytes) / denom;
    state.counters["approx_accum_p_key_bytes"] = static_cast<double>(metrics.accum_p.approx_bytes) / denom;
    state.counters["approx_accum_q_key_bytes"] = static_cast<double>(metrics.accum_q.approx_bytes) / denom;
    state.counters["approx_expcrt_key_bytes"] = static_cast<double>(metrics.expcrt.approx_bytes) / denom;
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

template <typename Params>
struct PipelineFixture {
    explicit PipelineFixture(const bdf17::ExperimentConfig &config) : runner(config) {
        const auto input = BuildAccumulatorInput<Params>(runner);
        a = input.first;
        b = input.second;
        ct_p = runner.RunAccumulatorP(a, b, accum_p_stats);
        ct_q = runner.RunAccumulatorQ(a, b, accum_q_stats);
        tensor_ct = runner.RunExpCRT(ct_p, ct_q, expcrt_stats);
    }

    bdf17::BootstrapRunner<Params> runner;
    std::vector<int64_t> a;
    int64_t b = 0;

    typename Params::SchemePt::RLWECiphertext ct_p;
    typename Params::SchemeQt::RLWECiphertext ct_q;
    typename Params::SchemePQ::RLWECiphertext tensor_ct;

    bdf17::StageStats accum_p_stats;
    bdf17::StageStats accum_q_stats;
    bdf17::StageStats expcrt_stats;
};

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

static void BM_BootstrapStageAccumulatorPToyWarm(benchmark::State &state) {
    using Params = bdf17::ToyEquivalenceProfile;
    static const bdf17::ExperimentConfig config = BenchmarkConfigToy();
    static PipelineFixture<Params> fixture(config);

    bdf17::StageStats aggregate_stats;
    for (auto _ : state) {
        bdf17::StageStats stats;
        auto ct_p = fixture.runner.RunAccumulatorP(fixture.a, fixture.b, stats);
        bdf17::MergeStageStats(aggregate_stats, stats);
        benchmark::DoNotOptimize(ct_p);
        benchmark::ClobberMemory();
    }

    SetPipelineMetadata<Params>(state, config, "warm", "stage", "accum_p");
    AddStageStatsCounters(state, aggregate_stats, static_cast<size_t>(state.iterations()));
}

BENCHMARK(BM_BootstrapStageAccumulatorPToyWarm)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_BootstrapStageAccumulatorQToyWarm(benchmark::State &state) {
    using Params = bdf17::ToyEquivalenceProfile;
    static const bdf17::ExperimentConfig config = BenchmarkConfigToy();
    static PipelineFixture<Params> fixture(config);

    bdf17::StageStats aggregate_stats;
    for (auto _ : state) {
        bdf17::StageStats stats;
        auto ct_q = fixture.runner.RunAccumulatorQ(fixture.a, fixture.b, stats);
        bdf17::MergeStageStats(aggregate_stats, stats);
        benchmark::DoNotOptimize(ct_q);
        benchmark::ClobberMemory();
    }

    SetPipelineMetadata<Params>(state, config, "warm", "stage", "accum_q");
    AddStageStatsCounters(state, aggregate_stats, static_cast<size_t>(state.iterations()));
}

BENCHMARK(BM_BootstrapStageAccumulatorQToyWarm)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_BootstrapStageExpCRTToyWarm(benchmark::State &state) {
    using Params = bdf17::ToyEquivalenceProfile;
    static const bdf17::ExperimentConfig config = BenchmarkConfigToy();
    static PipelineFixture<Params> fixture(config);

    bdf17::StageStats aggregate_stats;
    for (auto _ : state) {
        bdf17::StageStats stats;
        auto tensor_ct = fixture.runner.RunExpCRT(fixture.ct_p, fixture.ct_q, stats);
        bdf17::MergeStageStats(aggregate_stats, stats);
        benchmark::DoNotOptimize(tensor_ct);
        benchmark::ClobberMemory();
    }

    SetPipelineMetadata<Params>(state, config, "warm", "stage", "expcrt");
    AddStageStatsCounters(state, aggregate_stats, static_cast<size_t>(state.iterations()));
}

BENCHMARK(BM_BootstrapStageExpCRTToyWarm)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_BootstrapStageFunExtractToyWarm(benchmark::State &state) {
    using Params = bdf17::ToyEquivalenceProfile;
    static const bdf17::ExperimentConfig config = BenchmarkConfigToy();
    static PipelineFixture<Params> fixture(config);

    bdf17::StageStats aggregate_stats;
    for (auto _ : state) {
        bdf17::StageStats stats;
        auto extracted = fixture.runner.RunFunExtract(fixture.tensor_ct, stats);
        bdf17::MergeStageStats(aggregate_stats, stats);
        benchmark::DoNotOptimize(extracted);
        benchmark::ClobberMemory();
    }

    SetPipelineMetadata<Params>(state, config, "warm", "stage", "fun_extract");
    AddStageStatsCounters(state, aggregate_stats, static_cast<size_t>(state.iterations()));
}

BENCHMARK(BM_BootstrapStageFunExtractToyWarm)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_BootstrapTrialToyWarm(benchmark::State &state) {
    using Params = bdf17::ToyEquivalenceProfile;
    static const bdf17::ExperimentConfig config = BenchmarkConfigToy();
    static bdf17::BootstrapRunner<Params> runner(config);

    bdf17::BootstrapMetrics aggregate_metrics;
    size_t trial_index = 0;
    for (auto _ : state) {
        bdf17::BootstrapMetrics metrics;
        auto result = runner.RunTrial({trial_index}, &metrics);
        ++trial_index;
        bdf17::MergeBootstrapMetrics(aggregate_metrics, metrics);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    SetPipelineMetadata<Params>(state, config, "warm", "trial", "end_to_end");
    AddTrialMetricsCounters(state, aggregate_metrics, static_cast<size_t>(state.iterations()));
}

BENCHMARK(BM_BootstrapTrialToyWarm)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_BootstrapTrialToyCold(benchmark::State &state) {
    using Params = bdf17::ToyEquivalenceProfile;
    static const bdf17::ExperimentConfig config = BenchmarkConfigToy();

    bdf17::BootstrapMetrics aggregate_metrics;
    size_t trial_index = 0;
    for (auto _ : state) {
        bdf17::BootstrapRunner<Params> runner(config);
        bdf17::BootstrapMetrics metrics;
        auto result = runner.RunTrial({trial_index}, &metrics);
        ++trial_index;
        bdf17::MergeBootstrapMetrics(aggregate_metrics, metrics);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    SetPipelineMetadata<Params>(state, config, "cold", "trial", "end_to_end");
    AddTrialMetricsCounters(state, aggregate_metrics, static_cast<size_t>(state.iterations()));
}

BENCHMARK(BM_BootstrapTrialToyCold)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);
