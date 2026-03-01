#include <gtest/gtest.h>

#include "bootstrap_runner.hpp"

namespace {

bdf17::ExperimentConfig SmoothPipelineConfig(uint64_t seed) {
    bdf17::ExperimentConfig config{};
    config.seed = seed;
    config.num_trials = 1;
    config.expcrt_variant = bdf17::ExpCrtVariant::TensorTrick;
    config.profile_name = "smooth-ntt-1153x1297";
    config.lut_name = "lowbit";
    config.enable_lwe_dim_reduction_override = false;
    config.has_enable_lwe_dim_reduction_override = false;
    config.emit_json = false;
    return config;
}

} // namespace

TEST(BootstrapPipeline, SmoothProfileSeededTrialIsDeterministicAcrossRuns) {
    const bdf17::ExperimentConfig config = SmoothPipelineConfig(0x1234ULL);

    bdf17::BootstrapRunner<bdf17::SmoothNtt1153x1297Profile> run_a(config);
    bdf17::BootstrapRunner<bdf17::SmoothNtt1153x1297Profile> run_b(config);

    bdf17::BootstrapMetrics metrics_a;
    bdf17::BootstrapMetrics metrics_b;
    const auto trial_a = run_a.RunTrial({0}, &metrics_a);
    const auto trial_b = run_b.RunTrial({0}, &metrics_b);

    EXPECT_TRUE(trial_a.ok);
    EXPECT_TRUE(trial_b.ok);
    EXPECT_EQ(trial_a.expected, trial_a.got);
    EXPECT_EQ(trial_b.expected, trial_b.got);

    EXPECT_EQ(trial_a.trial_index, trial_b.trial_index);
    EXPECT_EQ(trial_a.bits_le, trial_b.bits_le);
    EXPECT_EQ(trial_a.packed_plain, trial_b.packed_plain);
    EXPECT_EQ(trial_a.phase, trial_b.phase);
    EXPECT_EQ(trial_a.expected, trial_b.expected);
    EXPECT_EQ(trial_a.got, trial_b.got);
    EXPECT_EQ(trial_a.ok, trial_b.ok);

    EXPECT_GT(metrics_a.total.ms, 0.0);
    EXPECT_GT(metrics_a.total.forward_ntt, 0u);
    EXPECT_GT(metrics_a.total.inverse_ntt, 0u);
    EXPECT_GT(metrics_a.total.extmult, 0u);

    EXPECT_EQ(metrics_a.total.forward_ntt, metrics_b.total.forward_ntt);
    EXPECT_EQ(metrics_a.total.inverse_ntt, metrics_b.total.inverse_ntt);
    EXPECT_EQ(metrics_a.total.galois, metrics_b.total.galois);
    EXPECT_EQ(metrics_a.total.keyswitch, metrics_b.total.keyswitch);
    EXPECT_EQ(metrics_a.total.extmult, metrics_b.total.extmult);
    EXPECT_EQ(metrics_a.total.trace_calls, metrics_b.total.trace_calls);
}
