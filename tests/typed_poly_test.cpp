#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "ntt.h"
#include "ntt_plan.hpp"
#include "typed_poly.hpp"

namespace {

using CircToy = CircNTT<1093ULL, 5ULL, 13, 2>;
using CoeffToy = CoeffPoly<CircToy>;
using EvalToy = EvalPoly<CircToy>;
using CoeffToyView = CoeffPolyView<CircToy>;
using ConstCoeffToyView = ConstCoeffPolyView<CircToy>;
using CoeffToyBuffer = CoeffPolyBuffer<CircToy>;

static_assert(requires(const CoeffToy &a, const CoeffToy &b) { a + b; });
static_assert(requires(const CoeffToy &a, const CoeffToy &b) { a - b; });
static_assert(requires(CoeffToy &a, const CoeffToy &b) { a += b; });
static_assert(requires(CoeffToy &a, const CoeffToy &b) { a -= b; });
static_assert(requires(CoeffToy &a) { a *= 7ULL; });
static_assert(requires(const EvalToy &a, const EvalToy &b) { a * b; });
static_assert(requires(EvalToy &a, const EvalToy &b) { a *= b; });
static_assert(requires(const CoeffToy &a) { a * 7ULL; });
static_assert(requires(const EvalToy &a) { 7ULL * a; });
static_assert(requires { CoeffToy::FromUnsigned(std::span<const uint64_t>{}); });
static_assert(requires { CoeffToy::FromSigned(std::span<const int64_t>{}); });
static_assert(requires(CoeffToy &a) { a.view(); });
static_assert(requires(const CoeffToy &a) { a.view(); });
static_assert(requires(CoeffToyView v) { v.span(); v.data(); });
static_assert(requires(ConstCoeffToyView v) { v.span(); v.data(); });
static_assert(requires(CoeffToyBuffer b) { b.raw_span(); b.poly_count(); b[0]; });

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
    const std::array<int64_t, 8> signed_input{
        0,
        1,
        -1,
        static_cast<int64_t>(CircToy::p),
        static_cast<int64_t>(CircToy::p) + 1,
        -static_cast<int64_t>(CircToy::p),
        -2 * static_cast<int64_t>(CircToy::p),
        -2 * static_cast<int64_t>(CircToy::p) - 1,
    };
    auto from_signed = CoeffToy::FromSigned(std::span<const int64_t>(signed_input));
    for (size_t i = 0; i < signed_input.size(); ++i) {
        EXPECT_EQ(from_signed[i], SignedMod(signed_input[i], CircToy::p));
    }

    const std::array<uint64_t, 6> unsigned_input{
        0,
        1,
        CircToy::p - 1,
        CircToy::p,
        CircToy::p + 1,
        2 * CircToy::p + 5,
    };
    auto from_unsigned = CoeffToy::FromUnsigned(std::span<const uint64_t>(unsigned_input));
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
    const std::array<uint64_t, 6> a_input{1, 2, 3, 4, 5, 6};
    const std::array<uint64_t, 6> b_input{10, 20, 30, 40, 50, 60};
    auto a = CoeffToy::FromUnsigned(std::span<const uint64_t>(a_input));
    auto b = CoeffToy::FromUnsigned(std::span<const uint64_t>(b_input));

    auto sum = a + b;
    auto back = sum - b;
    EXPECT_EQ(back, a);

    auto scaled = a * 11ULL;
    for (size_t i = 0; i < CircToy::N; ++i) {
        const uint64_t expected = Zp<CircToy::p>::Mul(a[i], 11ULL);
        EXPECT_EQ(scaled[i], expected);
    }
}

TEST(TypedPoly, InPlaceOpsAndZeroHelpers) {
    const std::array<uint64_t, 6> a_input{2, 4, 6, 8, 10, 12};
    const std::array<uint64_t, 6> b_input{1, 3, 5, 7, 9, 11};
    auto a = CoeffToy::FromUnsigned(std::span<const uint64_t>(a_input));
    auto b = CoeffToy::FromUnsigned(std::span<const uint64_t>(b_input));

    auto accum = a;
    accum += b;
    accum -= b;
    EXPECT_EQ(accum, a);

    auto scaled = a;
    scaled *= 17ULL;
    for (size_t i = 0; i < CircToy::N; ++i) {
        EXPECT_EQ(scaled[i], Zp<CircToy::p>::Mul(a[i], 17ULL));
    }

    auto zero = CoeffToy::Zero();
    for (size_t i = 0; i < CircToy::N; ++i) {
        EXPECT_EQ(zero[i], 0ULL);
    }

    zero.Fill(CircToy::p + 9);
    for (size_t i = 0; i < CircToy::N; ++i) {
        EXPECT_EQ(zero[i], 9ULL);
    }
    zero.SetZero();
    for (size_t i = 0; i < CircToy::N; ++i) {
        EXPECT_EQ(zero[i], 0ULL);
    }

    auto lhs_eval = ForwardToEval(std::vector<uint64_t>{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25});
    auto rhs_eval = ForwardToEval(std::vector<uint64_t>{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26});
    auto expected_eval = lhs_eval * rhs_eval;

    lhs_eval *= rhs_eval;
    EXPECT_EQ(lhs_eval, expected_eval);
}

TEST(TypedPoly, PolyViewSharesUnderlyingStorage) {
    const std::array<uint64_t, 6> coeff_input{1, 2, 3, 4, 5, 6};
    auto coeff = CoeffToy::FromUnsigned(std::span<const uint64_t>(coeff_input));
    CoeffToyView coeff_view = coeff.view();
    coeff_view[3] = 777;
    EXPECT_EQ(coeff[3], 777ULL);

    const CoeffToy &const_coeff = coeff;
    ConstCoeffToyView const_view = const_coeff.view();
    EXPECT_EQ(const_view[3], coeff[3]);

    CoeffToy copied_from_view(const_view);
    EXPECT_EQ(copied_from_view, coeff);
}

TEST(TypedPoly, PolyBufferProvidesContiguousViews) {
    CoeffToyBuffer buffer(3);
    EXPECT_EQ(buffer.poly_count(), 3U);
    ASSERT_EQ(buffer.raw_span().size(), 3 * CircToy::N);

    for (size_t poly_idx = 0; poly_idx < buffer.poly_count(); ++poly_idx) {
        auto view = buffer[poly_idx];
        for (size_t i = 0; i < CircToy::N; ++i) {
            view[i] = static_cast<uint64_t>(100 * poly_idx + i);
        }
    }

    const auto raw = buffer.raw_span();
    for (size_t poly_idx = 0; poly_idx < buffer.poly_count(); ++poly_idx) {
        for (size_t i = 0; i < CircToy::N; ++i) {
            const size_t flat_idx = poly_idx * CircToy::N + i;
            EXPECT_EQ(raw[flat_idx], static_cast<uint64_t>(100 * poly_idx + i));
        }
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
