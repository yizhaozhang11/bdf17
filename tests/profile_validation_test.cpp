#include <stdexcept>

#include <gtest/gtest.h>

#include "params.hpp"

namespace {

using InvalidDimReductionProfile = bdf17::ProfileBase<
    CircNTT<1093ULL, 5ULL, 7, 3>,
    CircNTT<1093ULL, 5ULL, 13, 2>,
    CircNTT<1093ULL, 5ULL, 7, 3>,
    CircNTT<1093ULL, 5ULL, 13, 2>,
    5,
    4,
    8,
    16ULL,
    16ULL,
    false,
    0.0,
    0.0,
    0.33,
    0.3,
    false>;

} // namespace

TEST(ProfileValidation, DefaultProfileValidForTensorTrick) {
    EXPECT_NO_THROW(
        (bdf17::ValidateProfileOrThrow<bdf17::SmoothNtt1153x1297Profile>(false, bdf17::ExpCrtVariant::TensorTrick)));
}

TEST(ProfileValidation, RejectsDimMismatchWhenDimReductionDisabled) {
    EXPECT_THROW(
        (bdf17::ValidateProfileOrThrow<InvalidDimReductionProfile>(false, bdf17::ExpCrtVariant::TensorTrick)),
        std::runtime_error);
}

TEST(ProfileValidation, RejectsUnsupportedPaperVariantEarly) {
    EXPECT_THROW(
        (bdf17::ValidateProfileOrThrow<bdf17::SmoothNtt1153x1297Profile>(false, bdf17::ExpCrtVariant::Paper)),
        std::runtime_error);
}
