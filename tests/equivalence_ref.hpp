#ifndef BDF17_TESTS_EQUIVALENCE_REF_HPP
#define BDF17_TESTS_EQUIVALENCE_REF_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "ntt.h"
#include "poly.h"
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

    using PolyP = Poly<NTTp>;
    using PolyQ = Poly<NTTq>;
    using PolyPt = Poly<NTTpt>;
    using PolyQt = Poly<NTTqt>;
    using PolyPQ = Poly<NTTpq>;
    using Z = NTTpq::Z;

    using SchemeP = SchemeImpl<PolyP, kKeySwitchBase>;
    using SchemeQ = SchemeImpl<PolyQ, kKeySwitchBase>;
    using SchemePt = SchemeImpl<PolyPt, kKeySwitchBase>;
    using SchemeQt = SchemeImpl<PolyQt, kKeySwitchBase>;
    using SchemePQ = SchemeImpl<PolyPQ, kKeySwitchBase>;

    static constexpr size_t kTensorDimension = PolyP::N * PolyQ::N;
    static constexpr uint64_t kAccumulatorInputModulus = kTensorDimension;
    static constexpr uint64_t kExtractModulus = Z::p;
    static constexpr uint64_t kFrontendModulus = kExtractModulus;
};

inline uint64_t AddMod(uint64_t a, uint64_t b, uint64_t mod) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(a % mod) + (b % mod)) % mod);
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

inline constexpr size_t kP = ToyEqParams::PolyP::N;
inline constexpr size_t kQ = ToyEqParams::PolyQ::N;
inline constexpr size_t kPQ = ToyEqParams::PolyPQ::N;
inline constexpr uint64_t kMod = ToyEqParams::Z::p;

inline const uint64_t kAlpha = InvMod(kQ % kP, kP);
inline const uint64_t kBeta = InvMod(kP % kQ, kQ);
inline const uint64_t kAlphaInv = InvMod(kAlpha, kP);

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
    if (lhs.size() != kPQ || rhs.size() != kPQ) {
        throw std::runtime_error("Convolution1DRef input size mismatch");
    }

    std::vector<uint64_t> out(kPQ, 0);
    for (size_t i = 0; i < kPQ; ++i) {
        const uint64_t l = lhs[i];
        if (l == 0) {
            continue;
        }
        for (size_t j = 0; j < kPQ; ++j) {
            const uint64_t r = rhs[j];
            if (r == 0) {
                continue;
            }
            const size_t idx = (i + j) % kPQ;
            out[idx] = AddMod(out[idx], MulMod(l, r, kMod), kMod);
        }
    }
    return out;
}

inline uint64_t TracePtoZRef(const std::vector<uint64_t> &coeffs) {
    if (coeffs.size() != kP) {
        throw std::runtime_error("TracePtoZRef input size mismatch");
    }
    return coeffs[0] % kMod;
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
