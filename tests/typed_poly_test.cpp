#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "ntt.h"
#include "poly.h"
#include "typed_poly.hpp"

namespace {

using CircToy = CircNTT<1093ULL, 5ULL, 13, 2>;
using LegacyPolyToy = Poly<CircToy>;
using CoeffToy = CoeffPoly<CircToy>;
using EvalToy = EvalPoly<CircToy>;

static_assert(requires(const CoeffToy &a, const CoeffToy &b) { a + b; });
static_assert(requires(const CoeffToy &a, const CoeffToy &b) { a - b; });
static_assert(requires(const EvalToy &a, const EvalToy &b) { a * b; });
static_assert(requires(const CoeffToy &a) { a * 7ULL; });
static_assert(requires(const EvalToy &a) { 7ULL * a; });

uint64_t SignedMod(int64_t value, uint64_t mod) {
    const int64_t mod_i64 = static_cast<int64_t>(mod);
    int64_t residue = value % mod_i64;
    if (residue < 0) {
        residue += mod_i64;
    }
    return static_cast<uint64_t>(residue);
}

EvalToy LegacyForwardToEval(const std::vector<uint64_t> &coeff) {
    auto legacy = LegacyPolyToy::FromCoeff(coeff);
    legacy.ToNTT();
    EvalToy out;
    for (size_t i = 0; i < CircToy::N; ++i) {
        out[i] = legacy.a[i];
    }
    return out;
}

} // namespace

TEST(TypedPoly, CoeffConstructorsAndMonomial) {
    const std::vector<int64_t> signed_input{
        0,
        1,
        -1,
        static_cast<int64_t>(CircToy::p),
        static_cast<int64_t>(CircToy::p) + 1,
        -static_cast<int64_t>(CircToy::p),
        -2 * static_cast<int64_t>(CircToy::p),
        -2 * static_cast<int64_t>(CircToy::p) - 1,
    };
    auto from_signed = CoeffToy::FromSigned(signed_input);
    for (size_t i = 0; i < signed_input.size(); ++i) {
        EXPECT_EQ(from_signed[i], SignedMod(signed_input[i], CircToy::p));
    }

    const std::vector<uint64_t> unsigned_input{
        0,
        1,
        CircToy::p - 1,
        CircToy::p,
        CircToy::p + 1,
        2 * CircToy::p + 5,
    };
    auto from_unsigned = CoeffToy::FromUnsigned(unsigned_input);
    for (size_t i = 0; i < unsigned_input.size(); ++i) {
        EXPECT_EQ(from_unsigned[i], unsigned_input[i] % CircToy::p);
    }

    auto monomial = CoeffToy::Monomial(CircToy::N + 2, CircToy::p + 7);
    for (size_t i = 0; i < CircToy::N; ++i) {
        const uint64_t expected = (i == 2) ? 7ULL : 0ULL;
        EXPECT_EQ(monomial[i], expected);
    }
}

TEST(TypedPoly, CoeffArithmetic) {
    auto a = CoeffToy::FromUnsigned(std::vector<uint64_t>{1, 2, 3, 4, 5, 6});
    auto b = CoeffToy::FromUnsigned(std::vector<uint64_t>{10, 20, 30, 40, 50, 60});

    auto sum = a + b;
    auto back = sum - b;
    EXPECT_EQ(back, a);

    auto scaled = a * 11ULL;
    for (size_t i = 0; i < CircToy::N; ++i) {
        const uint64_t expected = Zp<CircToy::p>::Mul(a[i], 11ULL);
        EXPECT_EQ(scaled[i], expected);
    }
}

TEST(TypedPoly, EvalPointwiseMultiplyMatchesLegacyPoly) {
    const std::vector<uint64_t> lhs_coeff{1, 8, 5, 3, 2, 7, 6, 4, 9, 10, 11, 12, 13};
    const std::vector<uint64_t> rhs_coeff{4, 3, 2, 1, 9, 8, 7, 6, 5, 11, 10, 12, 13};

    auto lhs_eval = LegacyForwardToEval(lhs_coeff);
    auto rhs_eval = LegacyForwardToEval(rhs_coeff);
    auto typed_product = lhs_eval * rhs_eval;

    auto lhs_legacy = LegacyPolyToy::FromCoeff(lhs_coeff);
    auto rhs_legacy = LegacyPolyToy::FromCoeff(rhs_coeff);
    lhs_legacy.ToNTT();
    rhs_legacy.ToNTT();
    auto legacy_product = lhs_legacy * rhs_legacy;

    for (size_t i = 0; i < CircToy::N; ++i) {
        EXPECT_EQ(typed_product[i], legacy_product.a[i]);
    }
}

TEST(TypedPoly, CoeffGaloisMatchesLegacyPoly) {
    const std::vector<uint64_t> coeff{7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

    auto typed_coeff = CoeffToy::FromUnsigned(coeff);
    auto typed_galois = GaloisApply(typed_coeff, 5);

    auto legacy_coeff = LegacyPolyToy::FromCoeff(coeff);
    auto legacy_galois = LegacyPolyToy::GaloisConjugate(legacy_coeff, 5);

    for (size_t i = 0; i < CircToy::N; ++i) {
        EXPECT_EQ(typed_galois[i], legacy_galois.a[i]);
    }
}

TEST(TypedPoly, EvalGaloisMatchesLegacyPoly) {
    const std::vector<uint64_t> coeff{5, 4, 3, 2, 1, 7, 8, 9, 10, 11, 12, 13, 6};

    auto legacy_eval = LegacyPolyToy::FromCoeff(coeff);
    legacy_eval.ToNTT();

    EvalToy typed_eval;
    for (size_t i = 0; i < CircToy::N; ++i) {
        typed_eval[i] = legacy_eval.a[i];
    }

    auto typed_galois = GaloisApply(typed_eval, 5);
    auto legacy_galois = LegacyPolyToy::GaloisConjugate(legacy_eval, 5);

    for (size_t i = 0; i < CircToy::N; ++i) {
        EXPECT_EQ(typed_galois[i], legacy_galois.a[i]);
    }
}
