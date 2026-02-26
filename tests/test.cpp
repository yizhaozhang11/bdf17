#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "ntt.h"
#include "poly.h"

namespace {

using PrimitiveToy = NTT<1093ULL, 5ULL, 13, 2>;
using CircToyP = CircNTT<1093ULL, 5ULL, 7, 3>;
using CircToyQ = CircNTT<1093ULL, 5ULL, 13, 2>;
using TensorToy = TensorNTTImpl<CircToyP, CircToyQ>;

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
    using PolyT = Poly<CircTransform>;

    auto lhs = RandomVector<CircTransform>(seed_lhs);
    auto rhs = RandomVector<CircTransform>(seed_rhs);
    auto expected = NaiveCircularConvolution<CircTransform>(lhs, rhs);

    auto a = PolyT::FromCoeff(lhs);
    auto b = PolyT::FromCoeff(rhs);
    a.ToNTT();
    b.ToNTT();
    auto c = a * b;
    c.ToCoeff();

    for (size_t i = 0; i < CircTransform::N; ++i) {
        EXPECT_EQ(c.a[i], expected[i]);
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

TEST(CircNTT, Linearity) {
    ExpectLinearity<CircToyP>(7, 8);
    ExpectLinearity<CircToyQ>(9, 10);
}

TEST(CircNTT, CircularConvolutionViaNTT) {
    ExpectCircularConvolutionViaNTT<CircToyP>(11, 12);
    ExpectCircularConvolutionViaNTT<CircToyQ>(13, 14);
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
