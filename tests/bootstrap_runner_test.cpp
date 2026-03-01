#include <gtest/gtest.h>

#include "bootstrap_runner.hpp"

TEST(BootstrapRunner, SingleTrialProducesResultAndMetrics) {
    bdf17::ExperimentConfig config{};
    config.seed = 0xC0FFEEULL;
    config.num_trials = 1;
    config.expcrt_variant = bdf17::ExpCrtVariant::TensorTrick;
    config.profile_name = "toy-equivalence";
    config.lut_name = "lowbit";
    config.enable_lwe_dim_reduction_override = false;
    config.has_enable_lwe_dim_reduction_override = false;
    config.emit_json = false;

    bdf17::BootstrapRunner<bdf17::ToyEquivalenceProfile> runner(config);
    bdf17::BootstrapMetrics metrics;
    const auto trial = runner.RunTrial({0}, &metrics);

    EXPECT_EQ(trial.trial_index, 0u);
    EXPECT_EQ(trial.bits_le.size(), runner.packing_width());
    EXPECT_TRUE(trial.ok);

    EXPECT_GT(metrics.total.ms, 0.0);
    EXPECT_GT(metrics.accum_p.extmult + metrics.accum_q.extmult, 0u);
    EXPECT_GE(metrics.fun_extract.trace_calls, 1u);
    EXPECT_GE(metrics.total.keyswitch, metrics.lwe_keyswitch.keyswitch);
}

TEST(BootstrapRunner, RejectsUnsupportedPaperVariantBeforeTrial) {
    bdf17::ExperimentConfig config{};
    config.seed = 0xC0FFEEULL;
    config.num_trials = 1;
    config.expcrt_variant = bdf17::ExpCrtVariant::Paper;
    config.profile_name = "toy-equivalence";
    config.lut_name = "lowbit";
    config.enable_lwe_dim_reduction_override = false;
    config.has_enable_lwe_dim_reduction_override = false;
    config.emit_json = false;

    EXPECT_THROW((void)bdf17::BootstrapRunner<bdf17::ToyEquivalenceProfile>(config), std::runtime_error);
}
