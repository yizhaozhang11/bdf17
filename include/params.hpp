#ifndef BDF17_PARAMS_HPP
#define BDF17_PARAMS_HPP

#include <cstddef>
#include <cstdint>

#include "ntt.h"
#include "poly.h"
#include "rlwe.h"

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

    static constexpr size_t kNumTrials = 8;
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

    using PolyP = Poly<NTTp>;
    using PolyQ = Poly<NTTq>;
    using PolyPt = Poly<NTTpt>;
    using PolyQt = Poly<NTTqt>;
    using PolyPQ = Poly<NTTpq>;

    using Z = NTTpq::Z;

    using SchemeP = SchemeImpl<PolyP, kKeySwitchBase>;
    using SchemeQ = SchemeImpl<PolyQ, kKeySwitchBase>;
    using SchemePQ = SchemeImpl<PolyPQ, kKeySwitchBase>;
    using SchemePt = SchemeImpl<PolyPt, kKeySwitchBase>;
    using SchemeQt = SchemeImpl<PolyQt, kKeySwitchBase>;

    static constexpr size_t kTensorDimension = PolyP::N * PolyQ::N;
    static constexpr uint64_t kAccumulatorInputModulus = kTensorDimension;
    static constexpr uint64_t kExtractModulus = Z::p;
    static constexpr uint64_t kFrontendModulus = kExtractModulus;

    static_assert(kLweFrontendDimension <= PolyP::N, "kLweFrontendDimension must not exceed PolyP::N");
    static_assert(kLweAccumulatorDimension > 0, "kLweAccumulatorDimension must be positive");
};

} // namespace bdf17

#endif // BDF17_PARAMS_HPP
