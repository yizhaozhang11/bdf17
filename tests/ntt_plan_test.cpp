#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "ntt.h"
#include "ntt_plan.hpp"
#include "params.hpp"
#include "typed_poly.hpp"

namespace {

using CircToyP = CircNTT<1093ULL, 5ULL, 7, 3>;
using CircToyQ = CircNTT<1093ULL, 5ULL, 13, 2>;
using TensorToy = TensorNTTImpl<CircToyP, CircToyQ>;
using CurrentCircP = bdf17::DefaultParams::NTTp;
using CurrentCircQ = bdf17::DefaultParams::NTTq;
using EdgeCircUV10 = CircNTT<72057421557668737ULL, 5ULL, 3, 2>;
using EdgeCircUV1m = CircNTT<2053ULL, 2ULL, 19, 2>;

template <typename Transform>
void ForwardRaw(uint64_t *a) {
    if constexpr (requires(uint64_t *ptr) { Transform::GetInstance().ForwardNTT(ptr); }) {
        Transform::GetInstance().ForwardNTT(a);
    } else {
        Transform::ForwardNTT(a);
    }
}

template <typename Transform>
void InverseRaw(uint64_t *a) {
    if constexpr (requires(uint64_t *ptr) { Transform::GetInstance().InverseNTT(ptr); }) {
        Transform::GetInstance().InverseNTT(a);
    } else {
        Transform::InverseNTT(a);
    }
}

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
void ExpectPlanMatchesTransformForwardInverse(uint64_t seed) {
    using Plan = CanonicalNttPlan<Transform, B>;
    using Coeff = CoeffPoly<Transform>;

    const auto input = RandomCoeffVector<Transform>(seed);

    Plan plan;
    const auto coeff = Coeff::FromUnsigned(input);
    const auto eval = plan.forward(coeff);
    const auto roundtrip = plan.inverse(eval);

    auto expected_eval = input;
    ForwardRaw<Transform>(expected_eval.data());

    for (size_t i = 0; i < Transform::N; ++i) {
        EXPECT_EQ(eval[i], expected_eval[i]);
    }

    auto expected_roundtrip = expected_eval;
    InverseRaw<Transform>(expected_roundtrip.data());
    for (size_t i = 0; i < Transform::N; ++i) {
        EXPECT_EQ(roundtrip[i], expected_roundtrip[i]);
        EXPECT_EQ(roundtrip[i], input[i]);
    }
}

template <typename Transform>
void ExpectAutoMatchesDirectTransform(uint64_t seed) {
    ExpectPlanMatchesTransformForwardInverse<Transform, Backend::Auto>(seed);
}

template <typename Transform, Backend B>
void ExpectBackendMatchesScalar(uint64_t seed) {
    using ScalarPlan = CanonicalNttPlan<Transform, Backend::Scalar>;
    using OtherPlan = CanonicalNttPlan<Transform, B>;
    using Coeff = CoeffPoly<Transform>;

    const auto input = RandomCoeffVector<Transform>(seed);
    const auto coeff = Coeff::FromUnsigned(input);

    ScalarPlan scalar_plan;
    OtherPlan other_plan;
    typename ScalarPlan::Workspace scalar_workspace;
    typename OtherPlan::Workspace other_workspace;

    const auto eval_scalar = scalar_plan.forward(coeff, scalar_workspace);
    const auto eval_other = other_plan.forward(coeff, other_workspace);
    for (size_t i = 0; i < Transform::N; ++i) {
        EXPECT_EQ(eval_other[i], eval_scalar[i]);
    }

    const auto round_scalar = scalar_plan.inverse(eval_scalar, scalar_workspace);
    const auto round_other = other_plan.inverse(eval_other, other_workspace);
    for (size_t i = 0; i < Transform::N; ++i) {
        EXPECT_EQ(round_other[i], round_scalar[i]);
        EXPECT_EQ(round_other[i], input[i]);
    }
}

template <typename Transform>
void ExpectBackendsMatchScalar(uint64_t seed) {
    ExpectBackendMatchesScalar<Transform, Backend::Auto>(seed + 0);
    if constexpr (BackendCompiled<Backend::Avx2>()) {
        ExpectBackendMatchesScalar<Transform, Backend::Avx2>(seed + 1);
    }
    if constexpr (BackendCompiled<Backend::Avx512>()) {
        ExpectBackendMatchesScalar<Transform, Backend::Avx512>(seed + 2);
    }
}

template <typename Transform>
void ExpectWorkspaceReuse(uint64_t seed) {
    using Plan = CanonicalNttPlan<Transform, Backend::Auto>;
    using Coeff = CoeffPoly<Transform>;

    Plan plan;
    typename Plan::Workspace workspace;
    const auto coeff = Coeff::FromUnsigned(RandomCoeffVector<Transform>(seed));

    const auto eval = plan.forward(coeff, workspace);
    ASSERT_GE(workspace.scratch.size(), Transform::N);
    uint64_t *const scratch_ptr = workspace.scratch.data();

    const auto roundtrip = plan.inverse(eval, workspace);
    EXPECT_EQ(workspace.scratch.data(), scratch_ptr);
    for (size_t i = 0; i < Transform::N; ++i) {
        EXPECT_EQ(roundtrip[i], coeff[i]);
    }
}

} // namespace

TEST(NttPlan, CircPAutoMatchesTransformSemantics) {
    ExpectAutoMatchesDirectTransform<CircToyP>(101);
}

TEST(NttPlan, CircQAutoMatchesTransformSemantics) {
    ExpectAutoMatchesDirectTransform<CircToyQ>(201);
}

TEST(NttPlan, TensorAutoMatchesTransformSemantics) {
    ExpectAutoMatchesDirectTransform<TensorToy>(301);
}

TEST(NttPlan, ScalarVsAvxBackendsMatchOnDefaultRings) {
    ExpectBackendsMatchScalar<CurrentCircP>(401);
    ExpectBackendsMatchScalar<CurrentCircQ>(501);
}

TEST(NttPlan, ScalarVsAvxBackendsMatchOnEdgeRings) {
    ExpectBackendsMatchScalar<EdgeCircUV10>(601);
    ExpectBackendsMatchScalar<EdgeCircUV1m>(701);
}

TEST(NttPlan, ScalarVsAvxBackendsMatchOnTensorRing) {
    ExpectBackendsMatchScalar<TensorToy>(801);
}

TEST(NttPlan, TensorWorkspaceIsReusedAcrossCalls) {
    ExpectWorkspaceReuse<TensorToy>(901);
}
