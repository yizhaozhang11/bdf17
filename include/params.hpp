#ifndef BDF17_PARAMS_HPP
#define BDF17_PARAMS_HPP

#include <cstddef>
#include <cstdint>

#include "ntt.h"
#include "ntt_plan.hpp"
#include "rlwe.h"
#include "typed_poly.hpp"

namespace bdf17 {

struct DefaultParams {
    // n, t, B
    static constexpr size_t kLweFrontendDimension = 600;
    static constexpr size_t kLweAccumulatorDimension = 600;
    static constexpr uint64_t kPlainModulus = 64;
    static constexpr uint64_t kKeySwitchBase = 1ULL << 8;
    static constexpr uint64_t kLweKeySwitchBase = kKeySwitchBase;
    static constexpr bool kEnableLweDimReduction = false;
    static constexpr double kLweNoiseVar = 4.0;
    static constexpr double kRlweNoiseVar = 4.0;
    static constexpr double kLweSecretDensity = 0.33;
    static constexpr double kAccumulatorSecretDensity = 0.3;

    static_assert(kPlainModulus > 0 && (kPlainModulus & (kPlainModulus - 1)) == 0, "kPlainModulus must be a power of two");

    // Accumulator rings
    using NTTp = CircNTT<72057421557668737LL, 5LL, 1153, 5>;
    using NTTq = CircNTT<72057421557668737LL, 5LL, 1297, 10>;

    // Mod-switched rings used before ExpCRT
    using NTTpt = CircNTT<108533126017LL, 10LL, 1153, 5>;
    using NTTqt = CircNTT<108533126017LL, 10LL, 1297, 10>;

    using NTTpq = TensorNTTImpl<NTTpt, NTTqt>;

    // Typed-domain aliases used by migrated components.
    using EvalP = EvalPoly<NTTp>;
    using EvalQ = EvalPoly<NTTq>;
    using EvalPt = EvalPoly<NTTpt>;
    using EvalQt = EvalPoly<NTTqt>;
    using EvalPQ = EvalPoly<NTTpq>;

    using CoeffP = CoeffPoly<NTTp>;
    using CoeffQ = CoeffPoly<NTTq>;
    using CoeffPt = CoeffPoly<NTTpt>;
    using CoeffQt = CoeffPoly<NTTqt>;
    using CoeffPQ = CoeffPoly<NTTpq>;

    using PlanP = CanonicalNttPlan<NTTp>;
    using PlanQ = CanonicalNttPlan<NTTq>;
    using PlanPt = CanonicalNttPlan<NTTpt>;
    using PlanQt = CanonicalNttPlan<NTTqt>;
    using PlanPQ = CanonicalNttPlan<NTTpq>;

    using Z = NTTpq::Z;

    using SchemeP = SchemeImpl<NTTp, kKeySwitchBase, PlanP>;
    using SchemeQ = SchemeImpl<NTTq, kKeySwitchBase, PlanQ>;
    using SchemePQ = SchemeImpl<NTTpq, kKeySwitchBase, PlanPQ>;
    using SchemePt = SchemeImpl<NTTpt, kKeySwitchBase, PlanPt>;
    using SchemeQt = SchemeImpl<NTTqt, kKeySwitchBase, PlanQt>;

    static constexpr size_t kTensorDimension = EvalP::N * EvalQ::N;
    static constexpr uint64_t kAccumulatorInputModulus = kTensorDimension;
    static constexpr uint64_t kExtractModulus = Z::p;
    static constexpr uint64_t kFrontendModulus = kExtractModulus;

    static_assert(kLweFrontendDimension <= EvalP::N, "kLweFrontendDimension must not exceed EvalP::N");
    static_assert(kLweAccumulatorDimension > 0, "kLweAccumulatorDimension must be positive");
};

} // namespace bdf17

#endif // BDF17_PARAMS_HPP
