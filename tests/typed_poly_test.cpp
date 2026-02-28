#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "ntt.h"
#include "ntt_plan.hpp"
#include "typed_poly.hpp"

namespace {

using CircToy = CircNTT<1093ULL, 5ULL, 13, 2>;
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

uint64_t MulMod(uint64_t lhs, uint64_t rhs, uint64_t mod) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(lhs) * rhs) % mod);
}

std::vector<uint64_t> NaiveCircularConvolution(const std::vector<uint64_t> &lhs, const std::vector<uint64_t> &rhs) {
    std::vector<uint64_t> out(CircToy::N, 0);
    for (size_t i = 0; i < CircToy::N; ++i) {
        for (size_t j = 0; j < CircToy::N; ++j) {
            const size_t k = (i + j) % CircToy::N;
            out[k] = (out[k] + MulMod(lhs[i], rhs[j], CircToy::p)) % CircToy::p;
        }
    }
    return out;
}

EvalToy ForwardToEval(const std::vector<uint64_t> &coeff) {
    CanonicalNttPlan<CircToy> plan;
    return plan.forward(CoeffToy::FromUnsigned(coeff));
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

TEST(TypedPoly, EvalPointwiseMultiplyMatchesCircularConvolution) {
    CanonicalNttPlan<CircToy> plan;
    const std::vector<uint64_t> lhs_coeff{1, 8, 5, 3, 2, 7, 6, 4, 9, 10, 11, 12, 13};
    const std::vector<uint64_t> rhs_coeff{4, 3, 2, 1, 9, 8, 7, 6, 5, 11, 10, 12, 13};

    auto lhs_eval = ForwardToEval(lhs_coeff);
    auto rhs_eval = ForwardToEval(rhs_coeff);
    auto typed_product = lhs_eval * rhs_eval;
    auto typed_product_coeff = plan.inverse(typed_product);

    const auto expected = NaiveCircularConvolution(lhs_coeff, rhs_coeff);
    for (size_t i = 0; i < CircToy::N; ++i) {
        EXPECT_EQ(typed_product_coeff[i], expected[i]);
    }
}

TEST(TypedPoly, CoeffGaloisMatchesPermutation) {
    const std::vector<uint64_t> coeff{7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

    auto typed_coeff = CoeffToy::FromUnsigned(coeff);
    auto typed_galois = GaloisApply(typed_coeff, 5);

    CoeffToy expected;
    expected[0] = typed_coeff[0];
    for (size_t i = 1; i < CircToy::N; ++i) {
        expected[i * 5 % CircToy::O] = typed_coeff[i];
    }
    for (size_t i = 0; i < CircToy::N; ++i) {
        EXPECT_EQ(typed_galois[i], expected[i]);
    }
}

TEST(TypedPoly, EvalGaloisMatchesCoeffThenForward) {
    CanonicalNttPlan<CircToy> plan;
    const std::vector<uint64_t> coeff{5, 4, 3, 2, 1, 7, 8, 9, 10, 11, 12, 13, 6};

    auto typed_coeff = CoeffToy::FromUnsigned(coeff);
    auto typed_eval = plan.forward(typed_coeff);
    auto typed_eval_galois = GaloisApply(typed_eval, 5);

    auto coeff_galois = GaloisApply(typed_coeff, 5);
    auto coeff_then_forward = plan.forward(coeff_galois);
    for (size_t i = 0; i < CircToy::N; ++i) {
        EXPECT_EQ(typed_eval_galois[i], coeff_then_forward[i]);
    }

    auto eval_then_coeff = plan.inverse(typed_eval_galois);
    for (size_t i = 0; i < CircToy::N; ++i) {
        EXPECT_EQ(eval_then_coeff[i], coeff_galois[i]);
    }
}
