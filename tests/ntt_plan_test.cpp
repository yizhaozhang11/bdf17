#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "ntt.h"
#include "ntt_plan.hpp"
#include "poly.h"
#include "typed_poly.hpp"

namespace {

using CircToyP = CircNTT<1093ULL, 5ULL, 7, 3>;
using CircToyQ = CircNTT<1093ULL, 5ULL, 13, 2>;
using TensorToy = TensorNTTImpl<CircToyP, CircToyQ>;

template <typename Transform>
std::vector<uint64_t> RandomCoeffVector(uint64_t seed) {
    std::mt19937_64 engine(seed);
    std::uniform_int_distribution<uint64_t> dist(0, Transform::p - 1);

    std::vector<uint64_t> out(Transform::N, 0);
    for (size_t i = 0; i < Transform::N; ++i) {
        out[i] = dist(engine);
    }
    return out;
}

template <typename Transform, Backend B>
void ExpectPlanMatchesLegacyForwardInverse(uint64_t seed) {
    using Plan = CanonicalNttPlan<Transform, B>;
    using LegacyPoly = Poly<Transform>;
    using Coeff = CoeffPoly<Transform>;

    const auto input = RandomCoeffVector<Transform>(seed);

    Plan plan;
    const auto coeff = Coeff::FromUnsigned(input);
    const auto eval = plan.forward(coeff);
    const auto roundtrip = plan.inverse(eval);

    auto legacy = LegacyPoly::FromCoeff(input);
    auto legacy_eval = legacy;
    legacy_eval.ToNTT();
    auto legacy_roundtrip = legacy_eval;
    legacy_roundtrip.ToCoeff();

    for (size_t i = 0; i < Transform::N; ++i) {
        EXPECT_EQ(eval[i], legacy_eval.a[i]);
        EXPECT_EQ(roundtrip[i], legacy_roundtrip.a[i]);
        EXPECT_EQ(roundtrip[i], input[i]);
    }
}

template <typename Transform>
void ExpectPlanBackendsMatchLegacy(uint64_t seed) {
    ExpectPlanMatchesLegacyForwardInverse<Transform, Backend::Auto>(seed + 0);
    ExpectPlanMatchesLegacyForwardInverse<Transform, Backend::Scalar>(seed + 1);
    ExpectPlanMatchesLegacyForwardInverse<Transform, Backend::Avx2>(seed + 2);
    ExpectPlanMatchesLegacyForwardInverse<Transform, Backend::Avx512>(seed + 3);
}

} // namespace

TEST(NttPlan, CircPBackendsMatchLegacySemantics) {
    ExpectPlanBackendsMatchLegacy<CircToyP>(101);
}

TEST(NttPlan, CircQBackendsMatchLegacySemantics) {
    ExpectPlanBackendsMatchLegacy<CircToyQ>(201);
}

TEST(NttPlan, TensorBackendsMatchLegacySemantics) {
    ExpectPlanBackendsMatchLegacy<TensorToy>(301);
}
