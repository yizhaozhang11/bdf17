#ifndef BDF17_PARAMS_HPP
#define BDF17_PARAMS_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "expcrt_variant.hpp"
#include "ntt.h"
#include "ntt_plan.hpp"
#include "rlwe.h"
#include "typed_poly.hpp"

namespace bdf17 {

enum class ProfileIntent : uint8_t {
    Experimental,
    Toy,
    PaperComparison,
};

inline const char *ProfileIntentName(const ProfileIntent intent) {
    switch (intent) {
        case ProfileIntent::Experimental:
            return "experimental";
        case ProfileIntent::Toy:
            return "toy";
        case ProfileIntent::PaperComparison:
            return "paper-comparison";
    }
    return "unknown";
}

template <
    typename NTTp_,
    typename NTTq_,
    typename NTTpt_,
    typename NTTqt_,
    size_t LweFrontendDimension_,
    size_t LweAccumulatorDimension_,
    uint64_t PlainModulus_,
    uint64_t KeySwitchBase_,
    uint64_t LweKeySwitchBase_,
    bool EnableLweDimReduction_,
    double LweNoiseVar_,
    double RlweNoiseVar_,
    double LweSecretDensity_,
    double AccumulatorSecretDensity_,
    bool SupportsPaperExpCrt_>
struct ProfileBase {
    static constexpr size_t kLweFrontendDimension = LweFrontendDimension_;
    static constexpr size_t kLweAccumulatorDimension = LweAccumulatorDimension_;
    static constexpr uint64_t kPlainModulus = PlainModulus_;
    static constexpr uint64_t kKeySwitchBase = KeySwitchBase_;
    static constexpr uint64_t kLweKeySwitchBase = LweKeySwitchBase_;
    static constexpr bool kEnableLweDimReduction = EnableLweDimReduction_;
    static constexpr double kLweNoiseVar = LweNoiseVar_;
    static constexpr double kRlweNoiseVar = RlweNoiseVar_;
    static constexpr double kLweSecretDensity = LweSecretDensity_;
    static constexpr double kAccumulatorSecretDensity = AccumulatorSecretDensity_;
    static constexpr bool kSupportsPaperExpCrt = SupportsPaperExpCrt_;

    using NTTp = NTTp_;
    using NTTq = NTTq_;
    using NTTpt = NTTpt_;
    using NTTqt = NTTqt_;
    using NTTpq = TensorNTTImpl<NTTpt, NTTqt>;

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
};

struct SmoothNtt1153x1297Profile
    : public ProfileBase<
          CircNTT<72057421557668737ULL, 5ULL, 1153, 5>,
          CircNTT<72057421557668737ULL, 5ULL, 1297, 10>,
          CircNTT<108533126017ULL, 10ULL, 1153, 5>,
          CircNTT<108533126017ULL, 10ULL, 1297, 10>,
          600,
          600,
          64,
          1ULL << 8,
          1ULL << 8,
          false,
          4.0,
          4.0,
          0.33,
          0.3,
          false> {
    static constexpr const char *kProfileName = "smooth-ntt-1153x1297";
    static constexpr ProfileIntent kProfileIntent = ProfileIntent::Experimental;
    static constexpr bool kPaperFaithfulTarget = false;
};

struct ToyEquivalenceProfile
    : public ProfileBase<
          CircNTT<1093ULL, 5ULL, 7, 3>,
          CircNTT<1093ULL, 5ULL, 13, 2>,
          CircNTT<1093ULL, 5ULL, 7, 3>,
          CircNTT<1093ULL, 5ULL, 13, 2>,
          4,
          4,
          8,
          16ULL,
          16ULL,
          false,
          0.0,
          0.0,
          0.33,
          0.3,
          false> {
    static constexpr const char *kProfileName = "toy-equivalence";
    static constexpr ProfileIntent kProfileIntent = ProfileIntent::Toy;
    static constexpr bool kPaperFaithfulTarget = false;
};

using DefaultParams = SmoothNtt1153x1297Profile;

template <typename Params>
void ValidateProfileOrThrow(bool enable_lwe_dim_reduction, ExpCrtVariant expcrt_variant) {
    if (Params::kPlainModulus == 0 || (Params::kPlainModulus & (Params::kPlainModulus - 1)) != 0) {
        throw std::runtime_error("profile validation failed: plaintext modulus must be a power of two");
    }
    if (Params::kLweFrontendDimension == 0) {
        throw std::runtime_error("profile validation failed: frontend LWE dimension must be non-zero");
    }
    if (Params::kLweAccumulatorDimension == 0) {
        throw std::runtime_error("profile validation failed: accumulator LWE dimension must be non-zero");
    }
    if (!enable_lwe_dim_reduction && Params::kLweFrontendDimension != Params::kLweAccumulatorDimension) {
        throw std::runtime_error("profile validation failed: dim reduction disabled but frontend and accumulator dimensions differ");
    }
    if (Params::kLweFrontendDimension > Params::EvalP::N) {
        throw std::runtime_error("profile validation failed: frontend LWE dimension exceeds p-ring degree");
    }
    if (Params::kAccumulatorInputModulus == 0) {
        throw std::runtime_error("profile validation failed: accumulator input modulus must be non-zero");
    }
    if (expcrt_variant == ExpCrtVariant::TensorTrick && Params::kAccumulatorInputModulus != Params::kTensorDimension) {
        throw std::runtime_error("profile validation failed: tensor-trick path requires accumulator input modulus == tensor dimension");
    }
    if (Params::kExtractModulus == 0 || Params::kFrontendModulus == 0) {
        throw std::runtime_error("profile validation failed: extract/frontend moduli must be explicit non-zero values");
    }
    if (expcrt_variant == ExpCrtVariant::Paper && !Params::kSupportsPaperExpCrt) {
        throw std::runtime_error("profile validation failed: selected ExpCRT variant is not supported by this profile/build");
    }
}

} // namespace bdf17

#endif // BDF17_PARAMS_HPP
