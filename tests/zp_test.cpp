#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "detail/simd_mod_arith.hpp"
#include "ntt.h"
#include "zp.h"

namespace {

using SmallZ = Zp<1093ULL>;
using LargeZ = Zp<72057421557668737ULL>;
using PrimitiveDefaultP = NTT<72057421557668737ULL, 5ULL, 1153, 5>;
using PrimitiveToy = NTT<1093ULL, 5ULL, 13, 2>;

template <typename Z>
uint64_t RefAdd(uint64_t a, uint64_t b) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(a) + b) % Z::p);
}

template <typename Z>
uint64_t RefSub(uint64_t a, uint64_t b) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(a) + Z::p - b) % Z::p);
}

template <typename Z>
uint64_t RefMul(uint64_t a, uint64_t b) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(a) * b) % Z::p);
}

template <typename Z>
void ExpectArithmeticAgreement(uint64_t seed) {
    std::mt19937_64 prng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, Z::p - 1);

    const std::vector<uint64_t> edge_values{
        0,
        1,
        2,
        Z::p / 2,
        Z::p > 2 ? Z::p - 2 : 0,
        Z::p - 1,
    };

    for (const uint64_t a : edge_values) {
        for (const uint64_t b : edge_values) {
            EXPECT_EQ(Z::Add(a, b), RefAdd<Z>(a, b));
            EXPECT_EQ(Z::Sub(a, b), RefSub<Z>(a, b));
            EXPECT_EQ(Z::Mul(a, b), RefMul<Z>(a, b));

            const auto mul_const = Z::MakeConstMultiplier(b);
            EXPECT_EQ(Z::MulConst(a, mul_const), RefMul<Z>(a, b));
            const uint64_t expected_shoup =
                static_cast<uint64_t>((static_cast<__uint128_t>(b % Z::p) << 64) / Z::p);
            EXPECT_EQ(mul_const.shoup, expected_shoup);
        }
    }

    for (size_t i = 0; i < 5000; ++i) {
        const uint64_t a = dist(prng);
        const uint64_t b = dist(prng);
        EXPECT_EQ(Z::Add(a, b), RefAdd<Z>(a, b));
        EXPECT_EQ(Z::Sub(a, b), RefSub<Z>(a, b));
        EXPECT_EQ(Z::Mul(a, b), RefMul<Z>(a, b));

        const auto mul_const = Z::MakeConstMultiplier(b);
        EXPECT_EQ(Z::MulConst(a, mul_const), RefMul<Z>(a, b));
    }
}

template <typename Z>
void ExpectTwiddleChain(uint64_t twiddle) {
    const auto twiddle_const = Z::MakeConstMultiplier(twiddle);
    uint64_t value = 1;
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(value, Z::Pow(twiddle, i));
        value = Z::MulConst(value, twiddle_const);
    }
}

TEST(Zp, ScalarContractFlags) {
    static_assert(SmallZ::kScalarFastPathSupported);
    static_assert(SmallZ::kSimdFastPathSupported);
    static_assert(LargeZ::kScalarFastPathSupported);
    static_assert(LargeZ::kSimdFastPathSupported);
    EXPECT_TRUE(SmallZ::kScalarFastPathSupported);
    EXPECT_TRUE(SmallZ::kSimdFastPathSupported);
    EXPECT_TRUE(LargeZ::kScalarFastPathSupported);
    EXPECT_TRUE(LargeZ::kSimdFastPathSupported);
}

TEST(Zp, SmallModulusMatchesReferenceArithmetic) {
    ExpectArithmeticAgreement<SmallZ>(10101);
}

TEST(Zp, LargeModulusMatchesReferenceArithmetic) {
    ExpectArithmeticAgreement<LargeZ>(20202);
}

TEST(Zp, RepeatedTwiddleMultiplicationMatchesPow) {
    ExpectTwiddleChain<SmallZ>(PrimitiveToy::omega_N);
    ExpectTwiddleChain<SmallZ>(PrimitiveToy::omega_O);
    ExpectTwiddleChain<LargeZ>(PrimitiveDefaultP::omega_N);
    ExpectTwiddleChain<LargeZ>(PrimitiveDefaultP::omega_O);
}

#if defined(BDF17_ENABLE_AVX512) && defined(__AVX512F__) && defined(__AVX512DQ__)
TEST(Zp, Avx512MulHiMatchesScalarReference) {
    std::mt19937_64 prng(30303);
    std::uniform_int_distribution<uint64_t> dist(0, 0xffffffffffffffffULL);

    for (size_t iter = 0; iter < 200; ++iter) {
        std::array<uint64_t, 8> x{};
        std::array<uint64_t, 8> y{};
        for (size_t lane = 0; lane < 8; ++lane) {
            x[lane] = dist(prng);
            y[lane] = dist(prng);
        }

        const __m512i x_vec = _mm512_loadu_si512(static_cast<const void *>(x.data()));
        const __m512i y_vec = _mm512_loadu_si512(static_cast<const void *>(y.data()));
        const __m512i hi_vec = bdf17::detail::simd::MulHiU64x8(x_vec, y_vec);

        std::array<uint64_t, 8> got{};
        _mm512_storeu_si512(static_cast<void *>(got.data()), hi_vec);

        for (size_t lane = 0; lane < 8; ++lane) {
            const uint64_t expected = static_cast<uint64_t>((static_cast<__uint128_t>(x[lane]) * y[lane]) >> 64);
            EXPECT_EQ(got[lane], expected);
        }
    }
}

TEST(Zp, Avx512MulConstMatchesScalarReference) {
    std::mt19937_64 prng(40404);
    std::uniform_int_distribution<uint64_t> dist_x(0, LargeZ::p - 1);
    std::uniform_int_distribution<uint64_t> dist_c(0, LargeZ::p - 1);

    for (size_t iter = 0; iter < 200; ++iter) {
        std::array<uint64_t, 8> x{};
        std::array<uint64_t, 8> c{};
        std::array<uint64_t, 8> c_shoup{};
        for (size_t lane = 0; lane < 8; ++lane) {
            x[lane] = dist_x(prng);
            c[lane] = dist_c(prng);
            c_shoup[lane] = LargeZ::MakeConstMultiplier(c[lane]).shoup;
        }

        const __m512i x_vec = _mm512_loadu_si512(static_cast<const void *>(x.data()));
        const __m512i c_vec = _mm512_loadu_si512(static_cast<const void *>(c.data()));
        const __m512i c_shoup_vec = _mm512_loadu_si512(static_cast<const void *>(c_shoup.data()));
        const __m512i got_vec = bdf17::detail::simd::MulConstU64x8<LargeZ>(x_vec, c_vec, c_shoup_vec);

        std::array<uint64_t, 8> got{};
        _mm512_storeu_si512(static_cast<void *>(got.data()), got_vec);

        for (size_t lane = 0; lane < 8; ++lane) {
            const uint64_t expected = LargeZ::MulConst(x[lane], {c[lane], c_shoup[lane]});
            EXPECT_EQ(got[lane], expected);
        }
    }
}
#endif

} // namespace
