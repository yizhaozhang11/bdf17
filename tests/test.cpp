#include <array>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ntt.h"
#include "ntt_plan.hpp"
#include "params.hpp"
#include "rlwe.h"
#include "typed_poly.hpp"

namespace {

using PrimitiveToy = NTT<1093ULL, 5ULL, 13, 2>;
using CircToyP = CircNTT<1093ULL, 5ULL, 7, 3>;
using CircToyQ = CircNTT<1093ULL, 5ULL, 13, 2>;
using TensorToy = TensorNTTImpl<CircToyP, CircToyQ>;
using CurrentCircP = bdf17::DefaultParams::NTTp;
using CurrentCircQ = bdf17::DefaultParams::NTTq;

// Edge/corner cases across mixed-radix exponents in N = 2^u * 3^v (where N = O - 1).
using PrimitiveUV00 = NTT<72057421557668737ULL, 5ULL, 2, 1>;         // (u, v) = (0, 0), metadata-only
using CircUV10 = CircNTT<72057421557668737ULL, 5ULL, 3, 2>;          // (u, v) = (1, 0)
using CircUV11 = CircNTT<72057421557668737ULL, 5ULL, 7, 3>;          // (u, v) = (1, 1)
using CircUV1mP = CircNTT<2053ULL, 2ULL, 19, 2>;                      // (u, v) = (1, 2)
using CircUVn0Q = CircNTT<1361ULL, 3ULL, 17, 3>;                      // (u, v) = (4, 0)
using CircUV1mQ = CircNTT<26407ULL, 5ULL, 163, 2>;                    // (u, v) = (1, 4)
using CircUVnmP = CircNTT<3984769ULL, 17ULL, 1153, 5>;                // (u, v) = (7, 2)
using CircUVnmQ = CircNTT<10085473ULL, 5ULL, 1297, 10>;               // (u, v) = (4, 4)

template <typename Transform>
void ExpectRoundTrip(uint64_t seed);

template <typename Transform>
void ExpectLinearity(uint64_t seed_lhs, uint64_t seed_rhs);

bool ExtendedNttSuiteEnabled() {
    const char *flag = std::getenv("BDF17_ENABLE_EXTENDED_NTT_TESTS");
    if (flag == nullptr) {
        return false;
    }
    std::string_view value(flag);
    return value == "1" || value == "true" || value == "on";
}

template <typename CircTransform, size_t ExpectedU, size_t ExpectedV>
void ExpectUvCase(uint64_t seed_base) {
    using Primitive = typename CircTransform::PrimitiveNTT;
    static_assert(Primitive::u == ExpectedU);
    static_assert(Primitive::v == ExpectedV);

    EXPECT_EQ(Primitive::u, ExpectedU);
    EXPECT_EQ(Primitive::v, ExpectedV);

    ExpectRoundTrip<Primitive>(seed_base + 0);
    ExpectRoundTrip<CircTransform>(seed_base + 1);
    ExpectLinearity<Primitive>(seed_base + 2, seed_base + 3);
    ExpectLinearity<CircTransform>(seed_base + 4, seed_base + 5);
}

uint64_t MulMod(uint64_t lhs, uint64_t rhs, uint64_t mod) {
    return (uint64_t)((__uint128_t)lhs * rhs % mod);
}

uint64_t PowMod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1 % mod;
    base %= mod;
    while (exp > 0) {
        if (exp & 1ULL) {
            result = MulMod(result, base, mod);
        }
        base = MulMod(base, base, mod);
        exp >>= 1ULL;
    }
    return result;
}

template <typename Transform>
void Forward(uint64_t *a) {
    if constexpr (requires { Transform::GetInstance(); }) {
        Transform::GetInstance().ForwardNTT(a);
    } else {
        Transform::ForwardNTT(a);
    }
}

template <typename Transform>
void Inverse(uint64_t *a) {
    if constexpr (requires { Transform::GetInstance(); }) {
        Transform::GetInstance().InverseNTT(a);
    } else {
        Transform::InverseNTT(a);
    }
}

template <typename Transform>
std::vector<uint64_t> RandomVector(uint64_t seed) {
    std::mt19937_64 engine(seed);
    std::uniform_int_distribution<uint64_t> dist(0, Transform::p - 1);

    std::vector<uint64_t> out(Transform::N);
    for (size_t i = 0; i < Transform::N; ++i) {
        out[i] = dist(engine);
    }
    return out;
}

template <typename Transform>
std::vector<uint64_t> AddMod(const std::vector<uint64_t> &lhs, const std::vector<uint64_t> &rhs) {
    std::vector<uint64_t> out(Transform::N);
    for (size_t i = 0; i < Transform::N; ++i) {
        out[i] = (lhs[i] + rhs[i]) % Transform::p;
    }
    return out;
}

template <typename Transform>
std::vector<std::vector<uint64_t>> BoundaryVectors() {
    std::vector<std::vector<uint64_t>> out;

    out.emplace_back(Transform::N, 0);
    out.emplace_back(Transform::N, 1 % Transform::p);
    out.emplace_back(Transform::N, Transform::p - 1);

    std::vector<uint64_t> monomial_0(Transform::N, 0);
    monomial_0[0] = 1;
    out.push_back(monomial_0);

    std::vector<uint64_t> monomial_mid(Transform::N, 0);
    monomial_mid[Transform::N / 2] = 1;
    out.push_back(monomial_mid);

    std::vector<uint64_t> monomial_last(Transform::N, 0);
    monomial_last[Transform::N - 1] = 1;
    out.push_back(monomial_last);

    std::vector<uint64_t> alternating(Transform::N, 0);
    for (size_t i = 0; i < Transform::N; ++i) {
        alternating[i] = (i & 1ULL) ? (Transform::p - 1) : 0;
    }
    out.push_back(alternating);

    std::vector<uint64_t> near_reduction(Transform::N, 0);
    for (size_t i = 0; i < Transform::N; ++i) {
        switch (i % 6) {
            case 0:
                near_reduction[i] = Transform::p - 2;
                break;
            case 1:
                near_reduction[i] = Transform::p - 1;
                break;
            case 2:
                near_reduction[i] = 0;
                break;
            case 3:
                near_reduction[i] = 1;
                break;
            case 4:
                near_reduction[i] = 2;
                break;
            default:
                near_reduction[i] = Transform::p / 2;
                break;
        }
    }
    out.push_back(near_reduction);

    return out;
}

inline uint64_t SignedMod(int64_t value, uint64_t mod) {
    const int64_t mod_i64 = static_cast<int64_t>(mod);
    int64_t residue = value % mod_i64;
    if (residue < 0) {
        residue += mod_i64;
    }
    return static_cast<uint64_t>(residue);
}

template <typename Transform>
void ExpectRoundTrip(uint64_t seed) {
    auto input = RandomVector<Transform>(seed);
    auto transformed = input;
    Forward<Transform>(transformed.data());
    Inverse<Transform>(transformed.data());
    EXPECT_EQ(transformed, input);
}

template <typename Transform>
void ExpectLinearity(uint64_t seed_lhs, uint64_t seed_rhs) {
    auto lhs = RandomVector<Transform>(seed_lhs);
    auto rhs = RandomVector<Transform>(seed_rhs);
    auto lhs_plus_rhs = AddMod<Transform>(lhs, rhs);

    auto t_lhs = lhs;
    auto t_rhs = rhs;
    auto t_lhs_plus_rhs = lhs_plus_rhs;

    Forward<Transform>(t_lhs.data());
    Forward<Transform>(t_rhs.data());
    Forward<Transform>(t_lhs_plus_rhs.data());

    for (size_t i = 0; i < Transform::N; ++i) {
        const uint64_t expected = (t_lhs[i] + t_rhs[i]) % Transform::p;
        EXPECT_EQ(t_lhs_plus_rhs[i], expected);
    }
}

template <typename CircTransform>
std::vector<uint64_t> NaiveCircularConvolution(const std::vector<uint64_t> &lhs, const std::vector<uint64_t> &rhs) {
    std::vector<uint64_t> out(CircTransform::N, 0);
    for (size_t i = 0; i < CircTransform::N; ++i) {
        for (size_t j = 0; j < CircTransform::N; ++j) {
            const size_t k = (i + j) % CircTransform::N;
            out[k] = (out[k] + MulMod(lhs[i], rhs[j], CircTransform::p)) % CircTransform::p;
        }
    }
    return out;
}

template <typename CircTransform>
std::vector<uint64_t> NaiveForwardCirc(const std::vector<uint64_t> &coeffs) {
    using Primitive = typename CircTransform::PrimitiveNTT;
    const uint64_t p = CircTransform::p;
    const uint64_t omega = Primitive::omega_O;

    std::vector<uint64_t> out(CircTransform::N, 0);
    for (size_t k = 0; k < CircTransform::N; ++k) {
        const uint64_t wk = PowMod(omega, k, p);
        uint64_t power = 1;
        uint64_t acc = 0;
        for (size_t j = 0; j < CircTransform::N; ++j) {
            acc = (acc + MulMod(coeffs[j], power, p)) % p;
            power = MulMod(power, wk, p);
        }
        out[k] = acc;
    }
    return out;
}

template <typename CircTransform>
std::vector<uint64_t> NaiveInverseCirc(const std::vector<uint64_t> &evals) {
    using Primitive = typename CircTransform::PrimitiveNTT;
    const uint64_t p = CircTransform::p;
    const uint64_t omega_inv = PowMod(Primitive::omega_O, p - 2, p);
    const uint64_t n_inv = PowMod(CircTransform::N, p - 2, p);

    std::vector<uint64_t> out(CircTransform::N, 0);
    for (size_t j = 0; j < CircTransform::N; ++j) {
        uint64_t acc = 0;
        for (size_t k = 0; k < CircTransform::N; ++k) {
            const uint64_t twiddle = PowMod(omega_inv, j * k, p);
            acc = (acc + MulMod(evals[k], twiddle, p)) % p;
        }
        out[j] = MulMod(acc, n_inv, p);
    }
    return out;
}

template <typename TensorTransform>
std::vector<uint64_t> NaiveForwardTensor(const std::vector<uint64_t> &coeffs) {
    using NTTp = typename TensorTransform::NTTp;
    using NTTq = typename TensorTransform::NTTq;

    std::vector<uint64_t> out = coeffs;
    std::vector<uint64_t> tmp(std::max(NTTp::N, NTTq::N), 0);

    for (size_t i = 0; i < NTTp::N; ++i) {
        for (size_t j = 0; j < NTTq::N; ++j) {
            tmp[j] = out[NTTq::N * i + j];
        }
        auto row = NaiveForwardCirc<NTTq>(std::vector<uint64_t>(tmp.begin(), tmp.begin() + NTTq::N));
        for (size_t j = 0; j < NTTq::N; ++j) {
            out[NTTq::N * i + j] = row[j];
        }
    }

    for (size_t j = 0; j < NTTq::N; ++j) {
        for (size_t i = 0; i < NTTp::N; ++i) {
            tmp[i] = out[NTTq::N * i + j];
        }
        auto col = NaiveForwardCirc<NTTp>(std::vector<uint64_t>(tmp.begin(), tmp.begin() + NTTp::N));
        for (size_t i = 0; i < NTTp::N; ++i) {
            out[NTTq::N * i + j] = col[i];
        }
    }

    return out;
}

template <typename CircTransform>
void ExpectCircularConvolutionViaNTT(uint64_t seed_lhs, uint64_t seed_rhs) {
    using Coeff = CoeffPoly<CircTransform>;
    using Eval = EvalPoly<CircTransform>;
    CanonicalNttPlan<CircTransform> plan;

    auto lhs = RandomVector<CircTransform>(seed_lhs);
    auto rhs = RandomVector<CircTransform>(seed_rhs);
    auto expected = NaiveCircularConvolution<CircTransform>(lhs, rhs);

    Coeff a_coeff = Coeff::FromUnsigned(lhs);
    Coeff b_coeff = Coeff::FromUnsigned(rhs);
    Eval a_eval = plan.forward(a_coeff);
    Eval b_eval = plan.forward(b_coeff);
    Eval c_eval = a_eval * b_eval;
    Coeff c_coeff = plan.inverse(c_eval);

    for (size_t i = 0; i < CircTransform::N; ++i) {
        EXPECT_EQ(c_coeff[i], expected[i]);
    }
}

template <typename Transform>
void ExpectBoundaryRoundTrip() {
    const auto vectors = BoundaryVectors<Transform>();
    for (const auto &input : vectors) {
        auto transformed = input;
        Forward<Transform>(transformed.data());
        Inverse<Transform>(transformed.data());
        EXPECT_EQ(transformed, input);
    }
}

template <typename CircTransform>
void ExpectBoundaryCircularConvolutionViaNTT() {
    using Coeff = CoeffPoly<CircTransform>;
    using Eval = EvalPoly<CircTransform>;
    CanonicalNttPlan<CircTransform> plan;
    const auto vectors = BoundaryVectors<CircTransform>();

    const std::array<std::pair<size_t, size_t>, 3> index_pairs{{
        {1, 6}, // all ones x alternating
        {5, 7}, // monomial_last x near_reduction
        {4, 7}, // monomial_mid x near_reduction
    }};

    for (const auto &[lhs_idx, rhs_idx] : index_pairs) {
        const auto &lhs = vectors[lhs_idx];
        const auto &rhs = vectors[rhs_idx];
        const auto expected = NaiveCircularConvolution<CircTransform>(lhs, rhs);

        Coeff a_coeff = Coeff::FromUnsigned(lhs);
        Coeff b_coeff = Coeff::FromUnsigned(rhs);
        Eval a_eval = plan.forward(a_coeff);
        Eval b_eval = plan.forward(b_coeff);
        Eval c_eval = a_eval * b_eval;
        Coeff c_coeff = plan.inverse(c_eval);

        for (size_t i = 0; i < CircTransform::N; ++i) {
            EXPECT_EQ(c_coeff[i], expected[i]);
        }
    }
}

template <typename TensorTransform>
void ExpectBoundaryTensorForward() {
    const auto vectors = BoundaryVectors<TensorTransform>();
    for (const auto &input : vectors) {
        auto transformed = input;
        Forward<TensorTransform>(transformed.data());
        const auto expected = NaiveForwardTensor<TensorTransform>(input);
        EXPECT_EQ(transformed, expected);
    }
}

template <typename CircTransform>
void ExpectCoeffGaloisActionMatchesPermutation(size_t automorphism) {
    using Coeff = CoeffPoly<CircTransform>;

    Coeff coeff;
    for (size_t i = 0; i < Coeff::N; ++i) {
        coeff[i] = (37 + 17 * i) % Coeff::p;
    }

    const auto actual = GaloisApply(coeff, automorphism);
    Coeff expected;
    expected[0] = coeff[0];
    for (size_t i = 1; i < Coeff::N; ++i) {
        expected[i * automorphism % Coeff::O] = coeff[i];
    }

    for (size_t i = 0; i < Coeff::N; ++i) {
        EXPECT_EQ(actual[i], expected[i]);
    }
}

template <typename CircTransform>
void ExpectEvalGaloisActionMatchesCoeffThenForward(size_t automorphism, uint64_t seed) {
    using Coeff = CoeffPoly<CircTransform>;
    using Eval = EvalPoly<CircTransform>;
    CanonicalNttPlan<CircTransform> plan;

    auto coeff_input = Coeff::FromUnsigned(RandomVector<CircTransform>(seed));
    auto coeff_galois = GaloisApply(coeff_input, automorphism);
    auto coeff_then_forward = plan.forward(coeff_galois);

    Eval eval_input = plan.forward(coeff_input);
    auto eval_galois = GaloisApply(eval_input, automorphism);

    for (size_t i = 0; i < Eval::N; ++i) {
        EXPECT_EQ(eval_galois[i], coeff_then_forward[i]);
    }

    Coeff eval_back = plan.inverse(eval_galois);
    for (size_t i = 0; i < Coeff::N; ++i) {
        EXPECT_EQ(eval_back[i], coeff_galois[i]);
    }
}

} // namespace

TEST(NTTPrimitive, RoundTrip) {
    ExpectRoundTrip<PrimitiveToy>(1);
    ExpectRoundTrip<PrimitiveToy>(2);
}

TEST(NTTPrimitive, Linearity) {
    ExpectLinearity<PrimitiveToy>(3, 4);
}

TEST(CircNTT, RoundTrip) {
    ExpectRoundTrip<CircToyP>(5);
    ExpectRoundTrip<CircToyQ>(6);
}

TEST(CircNTT, BoundaryVectorsRoundTrip) {
    ExpectBoundaryRoundTrip<CircToyP>();
    ExpectBoundaryRoundTrip<CircToyQ>();
}

TEST(CircNTT, Linearity) {
    ExpectLinearity<CircToyP>(7, 8);
    ExpectLinearity<CircToyQ>(9, 10);
}

TEST(CircNTT, CircularConvolutionViaNTT) {
    ExpectCircularConvolutionViaNTT<CircToyP>(11, 12);
    ExpectCircularConvolutionViaNTT<CircToyQ>(13, 14);
}

TEST(CircNTT, BoundaryVectorsCircularConvolutionViaNTT) {
    ExpectBoundaryCircularConvolutionViaNTT<CircToyP>();
    ExpectBoundaryCircularConvolutionViaNTT<CircToyQ>();
}

TEST(CircNTT, MatchesNaiveForwardAndInverse) {
    auto input_p = RandomVector<CircToyP>(21);
    auto input_q = RandomVector<CircToyQ>(22);

    auto opt_p = input_p;
    auto opt_q = input_q;
    Forward<CircToyP>(opt_p.data());
    Forward<CircToyQ>(opt_q.data());

    auto naive_p = NaiveForwardCirc<CircToyP>(input_p);
    auto naive_q = NaiveForwardCirc<CircToyQ>(input_q);
    EXPECT_EQ(opt_p, naive_p);
    EXPECT_EQ(opt_q, naive_q);

    auto back_p = opt_p;
    auto back_q = opt_q;
    Inverse<CircToyP>(back_p.data());
    Inverse<CircToyQ>(back_q.data());

    auto naive_back_p = NaiveInverseCirc<CircToyP>(opt_p);
    auto naive_back_q = NaiveInverseCirc<CircToyQ>(opt_q);
    EXPECT_EQ(back_p, naive_back_p);
    EXPECT_EQ(back_q, naive_back_q);
    EXPECT_EQ(back_p, input_p);
    EXPECT_EQ(back_q, input_q);
}

TEST(TensorNTT, RoundTrip) {
    ExpectRoundTrip<TensorToy>(15);
    ExpectRoundTrip<TensorToy>(16);
}

TEST(TensorNTT, Linearity) {
    ExpectLinearity<TensorToy>(17, 18);
}

TEST(TensorNTT, MatchesNaiveSeparableForward) {
    auto input = RandomVector<TensorToy>(23);
    auto opt = input;
    Forward<TensorToy>(opt.data());
    auto naive = NaiveForwardTensor<TensorToy>(input);
    EXPECT_EQ(opt, naive);
}

TEST(TensorNTT, BoundaryVectorsMatchNaiveSeparableForward) {
    ExpectBoundaryTensorForward<TensorToy>();
}

TEST(CurrentRings, BoundaryVectorsRoundTrip) {
    ExpectBoundaryRoundTrip<CurrentCircP>();
    ExpectBoundaryRoundTrip<CurrentCircQ>();
}

TEST(CurrentRings, BoundaryVectorsCircularConvolutionViaNTT) {
    ExpectBoundaryCircularConvolutionViaNTT<CurrentCircP>();
    ExpectBoundaryCircularConvolutionViaNTT<CurrentCircQ>();
}

TEST(NTTMatrix, EdgeExponentDegenerateMetadataOnly) {
    // N = 1 (u=0, v=0) is representable in metadata, but the current kernel fast path
    // is implemented for N >= 2 and is therefore not executed here.
    EXPECT_EQ(PrimitiveUV00::u, 0);
    EXPECT_EQ(PrimitiveUV00::v, 0);
    EXPECT_EQ(PrimitiveUV00::N, 1);
}

TEST(NTTMatrix, EdgeExponentFastCases) {
    // Covers realizable pairs from:
    // - n = 7, m = 2: (0,0 metadata-only), (1,0), (1,1), (1,2)
    // - n = 4, m = 4: (4,0), (1,4)
    // Note: the current mixed-radix kernel requires N >= 2 for execution.
    ExpectUvCase<CircUV10, 1, 0>(200);
    ExpectUvCase<CircUV11, 1, 1>(300);
    ExpectUvCase<CircUV1mP, 1, 2>(400);
    ExpectUvCase<CircUVn0Q, 4, 0>(500);
    ExpectUvCase<CircUV1mQ, 1, 4>(600);

    ExpectCircularConvolutionViaNTT<CircUV10>(700, 701);
    ExpectCircularConvolutionViaNTT<CircUV11>(702, 703);
    ExpectCircularConvolutionViaNTT<CircUV1mP>(704, 705);
    ExpectCircularConvolutionViaNTT<CircUVn0Q>(706, 707);
    ExpectCircularConvolutionViaNTT<CircUV1mQ>(708, 709);
}

TEST(NTTMatrix, EdgeExponentLargeCasesOptional) {
    if (!ExtendedNttSuiteEnabled()) {
        GTEST_SKIP() << "Set BDF17_ENABLE_EXTENDED_NTT_TESTS=1 to run large matrix edge cases.";
    }

    // Largest edge pairs:
    // - n = 7, m = 2: (7,2)
    // - n = 4, m = 4: (4,4)
    ExpectUvCase<CircUVnmP, 7, 2>(800);
    ExpectUvCase<CircUVnmQ, 4, 4>(900);

    ExpectCircularConvolutionViaNTT<CircUVnmP>(1000, 1001);
    ExpectCircularConvolutionViaNTT<CircUVnmQ>(1002, 1003);
}

TEST(PolyOps, CoeffDomainGaloisConjugateMatchesPermutation) {
    ExpectCoeffGaloisActionMatchesPermutation<CircToyP>(3);
    ExpectCoeffGaloisActionMatchesPermutation<CircToyQ>(5);
}

TEST(PolyOps, EvalDomainGaloisConjugateMatchesCoeffThenForward) {
    ExpectEvalGaloisActionMatchesCoeffThenForward<CircToyP>(3, 44);
    ExpectEvalGaloisActionMatchesCoeffThenForward<CircToyQ>(5, 55);
}

TEST(PolyOps, SignedFromCoeffNormalizesNegativeMultiplesOfModulus) {
    using CoeffToy = CoeffPoly<CircToyP>;
    const int64_t p = static_cast<int64_t>(CoeffToy::p);
    const std::vector<int64_t> input{
        0,
        1,
        -1,
        p - 1,
        p,
        p + 1,
        -p,
        -p - 1,
        -2 * p,
        -2 * p - 1,
        2 * p,
        2 * p + 1,
    };
    auto coeff = CoeffToy::FromSigned(input);

    for (size_t i = 0; i < input.size() && i < CoeffToy::N; ++i) {
        EXPECT_EQ(coeff[i], SignedMod(input[i], CoeffToy::p));
    }
}

TEST(RLWESampling, EncryptSamplesZeroSumA) {
    using SchemeToy = SchemeImpl<CircToyP, 16>;

    std::vector<int64_t> sk(CircToyP::N, 0);
    sk[1] = 1;
    SchemeToy scheme(sk);

    SchemeToy::Coeff m_coeff;
    SchemeToy::Plan plan;
    auto m = plan.forward(m_coeff);

    for (size_t iter = 0; iter < 10; ++iter) {
        auto ct = scheme.RLWEEncrypt(m, scheme.sk, 8);
        auto a_coeff = plan.inverse(ct[0]);

        uint64_t sum = 0;
        for (size_t i = 0; i < CircToyP::N; ++i) {
            sum = (sum + a_coeff[i]) % CircToyP::p;
        }
        EXPECT_EQ(sum, 0ULL);
    }
}
