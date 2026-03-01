#include <bit>
#include <vector>

#include <gtest/gtest.h>

#include "fun_extract.hpp"

TEST(LutSemantics, LowBitAndHammingParityBuildersMatchDefinitions) {
    using Params = bdf17::SmoothNtt1153x1297Profile;

    const auto lowbit = bdf17::BuildLowBitLut<Params>();
    const auto parity = bdf17::BuildHammingParityLut<Params>();

    ASSERT_EQ(lowbit.size(), Params::kPlainModulus);
    ASSERT_EQ(parity.size(), Params::kPlainModulus);

    bool found_difference = false;
    for (size_t i = 0; i < Params::kPlainModulus; ++i) {
        EXPECT_EQ(lowbit[i], i & 1ULL) << "i=" << i;
        EXPECT_EQ(parity[i], static_cast<size_t>(std::popcount(static_cast<unsigned long long>(i)) & 1U)) << "i=" << i;
        if (lowbit[i] != parity[i]) {
            found_difference = true;
        }
    }

    EXPECT_TRUE(found_difference);
}

TEST(LutSemantics, ThresholdBuilderMatchesExpectedStepFunction) {
    using Params = bdf17::ToyEquivalenceProfile;

    const auto threshold = bdf17::BuildThresholdLut<Params>(4);
    const std::vector<size_t> expected{0, 0, 0, 0, 1, 1, 1, 1};

    EXPECT_EQ(threshold, expected);
    EXPECT_THROW((void)bdf17::BuildThresholdLut<Params>(Params::kPlainModulus), std::runtime_error);
}

TEST(LutSemantics, TruthTableBuilderValidatesShapeAndRange) {
    using Params = bdf17::ToyEquivalenceProfile;

    const std::vector<size_t> valid{0, 1, 1, 0, 1, 0, 0, 1};
    EXPECT_EQ(bdf17::BuildTruthTableLut<Params>(valid), valid);

    const std::vector<size_t> wrong_size{0, 1, 1};
    EXPECT_THROW((void)bdf17::BuildTruthTableLut<Params>(wrong_size), std::runtime_error);

    std::vector<size_t> out_of_range = valid;
    out_of_range[2] = Params::kPlainModulus;
    EXPECT_THROW((void)bdf17::BuildTruthTableLut<Params>(out_of_range), std::runtime_error);
}

TEST(LutSemantics, LutKindParserAcceptsAliases) {
    EXPECT_EQ(bdf17::ParseLutKindOrThrow("lowbit"), bdf17::LutKind::LowBit);
    EXPECT_EQ(bdf17::ParseLutKindOrThrow("parity"), bdf17::LutKind::HammingParity);
    EXPECT_EQ(bdf17::ParseLutKindOrThrow("hamming_parity"), bdf17::LutKind::HammingParity);
    EXPECT_EQ(bdf17::ParseLutKindOrThrow("threshold"), bdf17::LutKind::Threshold);
    EXPECT_EQ(bdf17::ParseLutKindOrThrow("truth-table"), bdf17::LutKind::TruthTable);
    EXPECT_THROW((void)bdf17::ParseLutKindOrThrow("foo"), std::runtime_error);
}
