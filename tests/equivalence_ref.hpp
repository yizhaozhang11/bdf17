#ifndef BDF17_TESTS_EQUIVALENCE_REF_HPP
#define BDF17_TESTS_EQUIVALENCE_REF_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "ntt.h"
#include "rlwe.h"

namespace bdf17::equiv_ref {

using CircToyP = CircNTT<1093ULL, 5ULL, 7, 3>;
using CircToyQ = CircNTT<1093ULL, 5ULL, 13, 2>;
using TensorToy = TensorNTTImpl<CircToyP, CircToyQ>;

struct ToyEqParams {
    static constexpr size_t kPlainModulus = 8;
    static constexpr size_t kLweFrontendDimension = 4;
    static constexpr size_t kLweAccumulatorDimension = 4;
    static constexpr uint64_t kKeySwitchBase = 16;
    static constexpr uint64_t kLweKeySwitchBase = 16;
    static constexpr bool kEnableLweDimReduction = false;

    using NTTp = CircToyP;
    using NTTq = CircToyQ;
    using NTTpt = CircToyP;
    using NTTqt = CircToyQ;
    using NTTpq = TensorToy;

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

    using SchemeP = SchemeImpl<NTTp, kKeySwitchBase>;
    using SchemeQ = SchemeImpl<NTTq, kKeySwitchBase>;
    using SchemePt = SchemeImpl<NTTpt, kKeySwitchBase>;
    using SchemeQt = SchemeImpl<NTTqt, kKeySwitchBase>;
    using SchemePQ = SchemeImpl<NTTpq, kKeySwitchBase>;

    static constexpr size_t kTensorDimension = EvalP::N * EvalQ::N;
    static constexpr uint64_t kAccumulatorInputModulus = kTensorDimension;
    static constexpr uint64_t kExtractModulus = Z::p;
    static constexpr uint64_t kFrontendModulus = kExtractModulus;
};

inline uint64_t AddMod(uint64_t a, uint64_t b, uint64_t mod) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(a % mod) + (b % mod)) % mod);
}

inline uint64_t SubMod(uint64_t a, uint64_t b, uint64_t mod) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(a % mod) + mod - (b % mod)) % mod);
}

inline uint64_t MulMod(uint64_t a, uint64_t b, uint64_t mod) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(a % mod) * (b % mod)) % mod);
}

inline uint64_t PowMod(uint64_t a, uint64_t e, uint64_t mod) {
    uint64_t base = a % mod;
    uint64_t out = 1 % mod;
    while (e > 0) {
        if ((e & 1ULL) != 0ULL) {
            out = MulMod(out, base, mod);
        }
        base = MulMod(base, base, mod);
        e >>= 1ULL;
    }
    return out;
}

inline uint64_t InvMod(uint64_t a, uint64_t mod) {
    const uint64_t reduced = a % mod;
    if (reduced == 0) {
        throw std::runtime_error("InvMod undefined for zero");
    }
    return PowMod(reduced, mod - 2, mod);
}

inline constexpr size_t kP = ToyEqParams::EvalP::N;
inline constexpr size_t kQ = ToyEqParams::EvalQ::N;
inline constexpr size_t kPQ = ToyEqParams::EvalPQ::N;
inline constexpr uint64_t kMod = ToyEqParams::Z::p;

inline const uint64_t kAlpha = InvMod(kQ % kP, kP);
inline const uint64_t kBeta = InvMod(kP % kQ, kQ);
inline const uint64_t kAlphaInv = InvMod(kAlpha, kP);
inline const uint64_t kBetaInv = InvMod(kBeta, kQ);

inline size_t RoundNearestIndexRef(size_t plain_modulus, size_t m, size_t exponent_modulus) {
    const __uint128_t scaled = static_cast<__uint128_t>(2) * plain_modulus * m + exponent_modulus;
    const __uint128_t denom = static_cast<__uint128_t>(2) * exponent_modulus;
    return static_cast<size_t>((scaled / denom) % plain_modulus);
}

inline std::vector<uint64_t> BuildPaperBootstrapFunctionRef(
    const std::vector<uint64_t> &plain_lut,
    size_t plain_modulus,
    size_t exponent_modulus) {
    if (plain_lut.size() != plain_modulus) {
        throw std::runtime_error("BuildPaperBootstrapFunctionRef plain_lut size mismatch");
    }
    if (plain_modulus == 0 || exponent_modulus == 0) {
        throw std::runtime_error("BuildPaperBootstrapFunctionRef modulus must be nonzero");
    }

    std::vector<uint64_t> out(exponent_modulus, 0);
    for (size_t m = 0; m < exponent_modulus; ++m) {
        const size_t idx = RoundNearestIndexRef(plain_modulus, m, exponent_modulus);
        out[m] = plain_lut[idx] % plain_modulus;
    }
    return out;
}

inline std::vector<uint64_t> PaperLutPolyRef(const std::vector<uint64_t> &f_paper) {
    const size_t n = f_paper.size();
    if (n == 0) {
        throw std::runtime_error("PaperLutPolyRef input size must be nonzero");
    }

    std::vector<uint64_t> out(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const size_t idx = (n - i) % n;
        out[idx] = AddMod(out[idx], f_paper[i], kMod);
    }
    return out;
}

inline size_t PaperFoldIndex(size_t a, size_t b) {
    return (a * kQ + b * kP) % kPQ;
}

inline size_t CrtFoldIndex(size_t u, size_t v) {
    return (((kAlpha * u) % kP) * kQ + ((kBeta * v) % kQ) * kP) % kPQ;
}

inline std::vector<uint64_t> MakeMonomialP(size_t i) {
    std::vector<uint64_t> out(kP, 0);
    out.at(i % kP) = 1;
    return out;
}

inline std::vector<uint64_t> MakeMonomialQ(size_t j) {
    std::vector<uint64_t> out(kQ, 0);
    out.at(j % kQ) = 1;
    return out;
}

inline std::vector<uint64_t> MakePairMonomial(size_t i, size_t j) {
    std::vector<uint64_t> out(kPQ, 0);
    out.at((i % kP) * kQ + (j % kQ)) = 1;
    return out;
}

inline std::vector<uint64_t> MakeFoldedMonomial(size_t k) {
    std::vector<uint64_t> out(kPQ, 0);
    out.at(k % kPQ) = 1;
    return out;
}

inline std::vector<uint64_t> CoeffTensorRef(const std::vector<uint64_t> &p_coeff, const std::vector<uint64_t> &q_coeff) {
    if (p_coeff.size() != kP || q_coeff.size() != kQ) {
        throw std::runtime_error("CoeffTensorRef input size mismatch");
    }

    std::vector<uint64_t> out(kPQ, 0);
    for (size_t i = 0; i < kP; ++i) {
        for (size_t j = 0; j < kQ; ++j) {
            out[i * kQ + j] = MulMod(p_coeff[i], q_coeff[j], kMod);
        }
    }
    return out;
}

inline std::vector<uint64_t> GaloisPRef(const std::vector<uint64_t> &coeffs, uint64_t gamma) {
    if (coeffs.size() != kP) {
        throw std::runtime_error("GaloisPRef input size mismatch");
    }

    std::vector<uint64_t> out(kP, 0);
    for (size_t i = 0; i < kP; ++i) {
        const size_t idx = (static_cast<uint64_t>(i) * gamma) % kP;
        out[idx] = AddMod(out[idx], coeffs[i], kMod);
    }
    return out;
}

inline std::vector<uint64_t> GaloisQRef(const std::vector<uint64_t> &coeffs, uint64_t gamma) {
    if (coeffs.size() != kQ) {
        throw std::runtime_error("GaloisQRef input size mismatch");
    }

    std::vector<uint64_t> out(kQ, 0);
    for (size_t j = 0; j < kQ; ++j) {
        const size_t idx = (static_cast<uint64_t>(j) * gamma) % kQ;
        out[idx] = AddMod(out[idx], coeffs[j], kMod);
    }
    return out;
}

inline std::vector<uint64_t> GaloisPairRef(const std::vector<uint64_t> &pair_coeffs, uint64_t gamma_p, uint64_t gamma_q) {
    if (pair_coeffs.size() != kPQ) {
        throw std::runtime_error("GaloisPairRef input size mismatch");
    }

    std::vector<uint64_t> out(kPQ, 0);
    for (size_t i = 0; i < kP; ++i) {
        for (size_t j = 0; j < kQ; ++j) {
            const size_t src = i * kQ + j;
            const size_t dst_i = (static_cast<uint64_t>(i) * gamma_p) % kP;
            const size_t dst_j = (static_cast<uint64_t>(j) * gamma_q) % kQ;
            const size_t dst = dst_i * kQ + dst_j;
            out[dst] = AddMod(out[dst], pair_coeffs[src], kMod);
        }
    }
    return out;
}

inline std::vector<uint64_t> FoldPaperRef(const std::vector<uint64_t> &pair_coeffs) {
    if (pair_coeffs.size() != kPQ) {
        throw std::runtime_error("FoldPaperRef input size mismatch");
    }

    std::vector<uint64_t> out(kPQ, 0);
    for (size_t a = 0; a < kP; ++a) {
        for (size_t b = 0; b < kQ; ++b) {
            const size_t src = a * kQ + b;
            const size_t dst = PaperFoldIndex(a, b);
            out[dst] = AddMod(out[dst], pair_coeffs[src], kMod);
        }
    }
    return out;
}

inline std::vector<uint64_t> FoldCrtRef(const std::vector<uint64_t> &pair_coeffs) {
    if (pair_coeffs.size() != kPQ) {
        throw std::runtime_error("FoldCrtRef input size mismatch");
    }

    std::vector<uint64_t> out(kPQ, 0);
    for (size_t u = 0; u < kP; ++u) {
        for (size_t v = 0; v < kQ; ++v) {
            const size_t src = u * kQ + v;
            const size_t dst = CrtFoldIndex(u, v);
            out[dst] = AddMod(out[dst], pair_coeffs[src], kMod);
        }
    }
    return out;
}

inline std::vector<uint64_t> UnfoldPaperRef(const std::vector<uint64_t> &folded) {
    if (folded.size() != kPQ) {
        throw std::runtime_error("UnfoldPaperRef input size mismatch");
    }

    std::vector<uint64_t> out(kPQ, 0);
    for (size_t a = 0; a < kP; ++a) {
        for (size_t b = 0; b < kQ; ++b) {
            out[a * kQ + b] = folded[PaperFoldIndex(a, b)] % kMod;
        }
    }
    return out;
}

inline std::vector<uint64_t> TransportPaperFoldedLutToCurrentPairBasisRef(const std::vector<uint64_t> &folded_paper) {
    const auto pair_paper = UnfoldPaperRef(folded_paper);
    return GaloisPairRef(pair_paper, kAlphaInv, kBetaInv);
}

inline std::vector<uint64_t> PaperTraceCoeffRef(const std::vector<uint64_t> &folded) {
    if (folded.size() != kPQ) {
        throw std::runtime_error("PaperTraceCoeffRef input size mismatch");
    }

    std::vector<uint64_t> out(kP, 0);
    for (size_t a = 0; a < kP; ++a) {
        out[a] = folded[a * kQ] % kMod;
    }
    return out;
}

inline std::vector<uint64_t> CrtTraceTransportedRef(const std::vector<uint64_t> &pair_coeffs) {
    const auto twisted_pair = GaloisPairRef(pair_coeffs, kAlpha, kBeta);
    const auto folded = FoldPaperRef(twisted_pair);
    const auto traced_paper = PaperTraceCoeffRef(folded);
    return GaloisPRef(traced_paper, kAlphaInv);
}

inline std::vector<uint64_t> ConvolutionCyclicRef(const std::vector<uint64_t> &lhs, const std::vector<uint64_t> &rhs, size_t n) {
    if (lhs.size() != n || rhs.size() != n) {
        throw std::runtime_error("ConvolutionCyclicRef input size mismatch");
    }

    std::vector<uint64_t> out(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const uint64_t l = lhs[i];
        if (l == 0) {
            continue;
        }
        for (size_t j = 0; j < n; ++j) {
            const uint64_t r = rhs[j];
            if (r == 0) {
                continue;
            }
            const size_t idx = (i + j) % n;
            out[idx] = AddMod(out[idx], MulMod(l, r, kMod), kMod);
        }
    }
    return out;
}

inline std::vector<uint64_t> Convolution2DRef(const std::vector<uint64_t> &lhs, const std::vector<uint64_t> &rhs) {
    if (lhs.size() != kPQ || rhs.size() != kPQ) {
        throw std::runtime_error("Convolution2DRef input size mismatch");
    }

    std::vector<uint64_t> out(kPQ, 0);
    for (size_t i0 = 0; i0 < kP; ++i0) {
        for (size_t j0 = 0; j0 < kQ; ++j0) {
            const uint64_t l = lhs[i0 * kQ + j0];
            if (l == 0) {
                continue;
            }
            for (size_t i1 = 0; i1 < kP; ++i1) {
                for (size_t j1 = 0; j1 < kQ; ++j1) {
                    const uint64_t r = rhs[i1 * kQ + j1];
                    if (r == 0) {
                        continue;
                    }
                    const size_t out_i = (i0 + i1) % kP;
                    const size_t out_j = (j0 + j1) % kQ;
                    const size_t out_idx = out_i * kQ + out_j;
                    out[out_idx] = AddMod(out[out_idx], MulMod(l, r, kMod), kMod);
                }
            }
        }
    }
    return out;
}

inline std::vector<uint64_t> Convolution1DRef(const std::vector<uint64_t> &lhs, const std::vector<uint64_t> &rhs) {
    return ConvolutionCyclicRef(lhs, rhs, kPQ);
}

inline uint64_t TracePtoZRef(const std::vector<uint64_t> &coeffs) {
    if (coeffs.size() != kP) {
        throw std::runtime_error("TracePtoZRef input size mismatch");
    }
    return coeffs[0] % kMod;
}

inline std::vector<uint64_t> ScaleVectorModRef(const std::vector<uint64_t> &coeffs, uint64_t scale) {
    std::vector<uint64_t> out(coeffs.size(), 0);
    for (size_t i = 0; i < coeffs.size(); ++i) {
        out[i] = MulMod(coeffs[i], scale, kMod);
    }
    return out;
}

inline std::vector<uint64_t> UnitMonomialRef(size_t n) {
    std::vector<uint64_t> out(n, 0);
    out[0] = 1;
    return out;
}

inline std::vector<uint64_t> Galois1DRef(const std::vector<uint64_t> &coeffs, uint64_t gamma) {
    const size_t n = coeffs.size();
    if (n == 0) {
        throw std::runtime_error("Galois1DRef input size must be nonzero");
    }

    std::vector<uint64_t> out(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const size_t idx = (static_cast<uint64_t>(i) * gamma) % n;
        out[idx] = AddMod(out[idx], coeffs[i], kMod);
    }
    return out;
}

struct RlweCiphertextRef {
    std::vector<uint64_t> a;
    std::vector<uint64_t> b;
};

inline RlweCiphertextRef MakeNoiselessRlweCtRef(
    const std::vector<uint64_t> &s,
    const std::vector<uint64_t> &message_monomial,
    uint64_t delta,
    uint64_t seed) {
    if (s.size() != message_monomial.size() || s.empty()) {
        throw std::runtime_error("MakeNoiselessRlweCtRef input size mismatch");
    }

    std::mt19937_64 engine(seed);
    std::uniform_int_distribution<uint64_t> dist(0, kMod - 1);

    RlweCiphertextRef ct;
    ct.a.resize(s.size(), 0);
    ct.b.resize(s.size(), 0);

    bool nonzero = false;
    for (size_t i = 0; i < ct.a.size(); ++i) {
        ct.a[i] = dist(engine);
        nonzero = nonzero || (ct.a[i] != 0);
    }
    if (!nonzero) {
        ct.a[0] = 1;
    }

    ct.b = ConvolutionCyclicRef(ct.a, s, s.size());
    const auto encoded_m = ScaleVectorModRef(message_monomial, delta);
    for (size_t i = 0; i < ct.b.size(); ++i) {
        ct.b[i] = AddMod(ct.b[i], encoded_m[i], kMod);
    }
    return ct;
}

inline RlweCiphertextRef GaloisCtRef(const RlweCiphertextRef &ct, uint64_t gamma) {
    return {Galois1DRef(ct.a, gamma), Galois1DRef(ct.b, gamma)};
}

inline std::vector<uint64_t> TensorCoeffPaperRef(const std::vector<uint64_t> &p_coeff, const std::vector<uint64_t> &q_coeff) {
    return FoldPaperRef(CoeffTensorRef(p_coeff, q_coeff));
}

struct PaperExpCrtOutputRef {
    std::vector<uint64_t> a0;
    std::vector<uint64_t> a1;
    std::vector<uint64_t> a2;
    std::vector<uint64_t> b;
};

struct PaperExpCrtSecretRef {
    std::vector<uint64_t> s0;
    std::vector<uint64_t> s1;
    std::vector<uint64_t> s2;
};

struct PaperExpCrtStateRef {
    PaperExpCrtOutputRef ct;
    PaperExpCrtSecretRef secret;
};

inline PaperExpCrtStateRef PaperExpCrtRef(
    const RlweCiphertextRef &cp,
    const RlweCiphertextRef &cq,
    const std::vector<uint64_t> &sp,
    const std::vector<uint64_t> &sq,
    uint64_t expcrt_scale) {
    if (cp.a.size() != kP || cp.b.size() != kP || sp.size() != kP) {
        throw std::runtime_error("PaperExpCrtRef p-side size mismatch");
    }
    if (cq.a.size() != kQ || cq.b.size() != kQ || sq.size() != kQ) {
        throw std::runtime_error("PaperExpCrtRef q-side size mismatch");
    }

    const auto cp_tw = GaloisCtRef(cp, kAlpha);
    const auto cq_tw = GaloisCtRef(cq, kBeta);

    PaperExpCrtStateRef out;
    out.ct.a0 = ScaleVectorModRef(TensorCoeffPaperRef(cp_tw.a, cq_tw.a), expcrt_scale);
    out.ct.a1 = ScaleVectorModRef(TensorCoeffPaperRef(cp_tw.a, cq_tw.b), expcrt_scale);
    out.ct.a2 = ScaleVectorModRef(TensorCoeffPaperRef(cp_tw.b, cq_tw.a), expcrt_scale);
    out.ct.b = ScaleVectorModRef(TensorCoeffPaperRef(cp_tw.b, cq_tw.b), expcrt_scale);

    const auto sp_tw = GaloisPRef(sp, kAlpha);
    const auto sq_tw = GaloisQRef(sq, kBeta);
    const auto one_p = UnitMonomialRef(kP);
    const auto one_q = UnitMonomialRef(kQ);

    out.secret.s0 = TensorCoeffPaperRef(sp_tw, sq_tw);
    for (size_t i = 0; i < out.secret.s0.size(); ++i) {
        out.secret.s0[i] = SubMod(0, out.secret.s0[i], kMod);
    }
    out.secret.s1 = TensorCoeffPaperRef(sp_tw, one_q);
    out.secret.s2 = TensorCoeffPaperRef(one_p, sq_tw);
    return out;
}

inline std::vector<uint64_t> PhaseRef(const PaperExpCrtOutputRef &ct, const PaperExpCrtSecretRef &s_ref) {
    if (ct.a0.size() != kPQ || ct.a1.size() != kPQ || ct.a2.size() != kPQ || ct.b.size() != kPQ) {
        throw std::runtime_error("PhaseRef ciphertext size mismatch");
    }
    if (s_ref.s0.size() != kPQ || s_ref.s1.size() != kPQ || s_ref.s2.size() != kPQ) {
        throw std::runtime_error("PhaseRef secret size mismatch");
    }

    std::vector<uint64_t> phase = ct.b;
    const auto t0 = Convolution1DRef(ct.a0, s_ref.s0);
    const auto t1 = Convolution1DRef(ct.a1, s_ref.s1);
    const auto t2 = Convolution1DRef(ct.a2, s_ref.s2);
    for (size_t i = 0; i < kPQ; ++i) {
        phase[i] = SubMod(phase[i], t0[i], kMod);
        phase[i] = SubMod(phase[i], t1[i], kMod);
        phase[i] = SubMod(phase[i], t2[i], kMod);
    }
    return phase;
}

inline size_t DecodeMonomialPhaseRef(const std::vector<uint64_t> &phase, uint64_t expected_scale) {
    if (phase.empty()) {
        throw std::runtime_error("DecodeMonomialPhaseRef empty phase");
    }

    bool found = false;
    size_t found_index = 0;
    uint64_t found_coeff = 0;
    for (size_t i = 0; i < phase.size(); ++i) {
        if (phase[i] == 0) {
            continue;
        }
        if (found) {
            throw std::runtime_error("DecodeMonomialPhaseRef expected monomial phase");
        }
        found = true;
        found_index = i;
        found_coeff = phase[i];
    }
    if (!found) {
        throw std::runtime_error("DecodeMonomialPhaseRef found zero phase");
    }

    if (expected_scale != 0 && found_coeff != (expected_scale % kMod)) {
        throw std::runtime_error("DecodeMonomialPhaseRef scale mismatch");
    }
    return found_index;
}

struct Mismatch {
    size_t index;
    uint64_t lhs;
    uint64_t rhs;
};

inline std::optional<Mismatch> FirstMismatch(const std::vector<uint64_t> &lhs, const std::vector<uint64_t> &rhs) {
    const size_t min_size = lhs.size() < rhs.size() ? lhs.size() : rhs.size();
    for (size_t i = 0; i < min_size; ++i) {
        if (lhs[i] != rhs[i]) {
            return Mismatch{i, lhs[i], rhs[i]};
        }
    }
    if (lhs.size() == rhs.size()) {
        return std::nullopt;
    }
    const uint64_t missing = std::numeric_limits<uint64_t>::max();
    return Mismatch{
        min_size,
        min_size < lhs.size() ? lhs[min_size] : missing,
        min_size < rhs.size() ? rhs[min_size] : missing,
    };
}

} // namespace bdf17::equiv_ref

#endif // BDF17_TESTS_EQUIVALENCE_REF_HPP
