#ifndef NTT_H
#define NTT_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#if (defined(BDF17_ENABLE_AVX2) && defined(__AVX2__)) || (defined(BDF17_ENABLE_AVX512) && defined(__AVX512F__))
#include <immintrin.h>
#endif

#include "detail/simd_mod_arith.hpp"
#include "ntt_backend.hpp"
#include "zp.h"

template <size_t N_>
struct ConstMulTable {
    std::array<uint64_t, N_> value{};
    std::array<uint64_t, N_> shoup{};
};

template <uint64_t p_, uint64_t g_, size_t O_, size_t w_>
class NTT {
public:
    // parameter/meta and compile-time tables
    constexpr static uint64_t p = p_;
    constexpr static uint64_t g = g_;
    constexpr static size_t O = O_;
    constexpr static size_t w = w_;

    using Z = Zp<p>;

    static_assert(p % ((uint64_t)O * (O - 1)) == 1, "p must be 1 mod O * (O - 1)");

    constexpr static size_t ComputeTwoAdicity(size_t N_) {
        size_t u_ = 0;
        while (N_ % 2 == 0) {
            N_ >>= 1;
            u_++;
        }
        return u_;
    }

    constexpr static size_t ComputeTwoPower(size_t N_) {
        return (size_t)1 << ComputeTwoAdicity(N_);
    }

    constexpr static size_t ComputeThreeAdicity(size_t N_) {
        size_t v_ = 0;
        while (N_ % 3 == 0) {
            N_ /= 3;
            v_++;
        }
        return v_;
    }

    constexpr static size_t ComputeThreePower(size_t N_) {
        size_t V_ = 1;
        for (size_t i = 0; i < ComputeThreeAdicity(N_); i++) {
            V_ *= 3;
        }
        return V_;
    }

    constexpr static size_t N = O - 1;
    constexpr static size_t u = ComputeTwoAdicity(N);
    constexpr static size_t U = ComputeTwoPower(N);
    constexpr static size_t v = ComputeThreeAdicity(N);
    constexpr static size_t V = ComputeThreePower(N);

    static_assert(N == U * V, "MixedRadix23 NTT requires O - 1 to factor as 2^u * 3^v");

    constexpr static uint64_t ComputeOmegaO() {
        return Z::Pow(g, (p - 1) / O);
    }

    constexpr static uint64_t ComputeOmegaOInv() {
        return Z::Pow(ComputeOmegaO(), p - 2);
    }

    constexpr static uint64_t ComputeOmegaN() {
        return Z::Pow(g, (p - 1) / N);
    }

    constexpr static uint64_t ComputeOmegaNInv() {
        return Z::Pow(ComputeOmegaN(), p - 2);
    }

    constexpr static uint64_t omega_O = ComputeOmegaO();
    constexpr static uint64_t omega_O_inv = ComputeOmegaOInv();
    constexpr static uint64_t omega_N = ComputeOmegaN();
    constexpr static uint64_t omega_N_inv = ComputeOmegaNInv();

    constexpr static std::array<size_t, N> PrecomputeGi() {
        std::array<size_t, N> gi_{};
        gi_[0] = 1;
        for (size_t i = 1; i < N; i++) {
            gi_[i] = (uint64_t)gi_[i - 1] * w % O;
        }
        return gi_;
    }

    constexpr static std::array<size_t, O> PrecomputeGiInv() {
        std::array<size_t, O> gi_inv_{};
        gi_inv_[0] = (size_t)-1;
        for (size_t i = 0; i < N; i++) {
            gi_inv_[gi[i]] = i;
        }
        return gi_inv_;
    }

    constexpr static size_t BitReverse(size_t x, size_t l, size_t r) {
        size_t y = 0;
        for (size_t i = 0; i < l; i++) {
            y = y * r + x % r;
            x /= r;
        }
        return y;
    }

    constexpr static std::array<uint64_t, N> PrecomputeDigitReverseTable() {
        std::array<uint64_t, N> digit_reverse_table_{};
        for (size_t i = 0; i < V; i++) {
            for (size_t j = 0; j < U; j++) {
                digit_reverse_table_[i * U + j] = BitReverse(j, u, 2) * V + BitReverse(i, v, 3);
            }
        }
        return digit_reverse_table_;
    }

    constexpr static std::array<size_t, N> gi = PrecomputeGi();
    constexpr static std::array<size_t, O> gi_inv = PrecomputeGiInv();
    constexpr static std::array<uint64_t, N> digit_reverse_table = PrecomputeDigitReverseTable();

    void ForwardNTT(uint64_t a[]) {
        ForwardNTTWithBackend<Backend::Auto>(a);
    }

    void ForwardNTT(uint64_t a[], uint64_t scratch[]) {
        ForwardNTTWithBackend<Backend::Auto>(a, scratch);
    }

    template <Backend B = Backend::Auto>
    void ForwardNTTWithBackend(uint64_t a[]) {
        std::array<uint64_t, N> scratch{};
        ForwardNTTWithBackend<B>(a, scratch.data());
    }

    template <Backend B = Backend::Auto>
    void ForwardNTTWithBackend(uint64_t a[], uint64_t scratch[]) {
        for (size_t i = 0; i < N; i++) {
            scratch[i] = a[gi[i] - 1];
        }

        ForwardMixedRadix23NTT<B>(scratch, a);

        for (size_t i = 0; i < N; i++) {
            a[i] = Z::MulConst(a[i], {omega_o_fwd_.value[i], omega_o_fwd_.shoup[i]});
        }

        InverseMixedRadix23NTT<B>(a, scratch);

        for (size_t i = 0; i < N; i++) {
            a[i] = scratch[(N - gi_inv[i + 1]) % N];
        }
    }

    void InverseNTT(uint64_t a[]) {
        InverseNTTWithBackend<Backend::Auto>(a);
    }

    void InverseNTT(uint64_t a[], uint64_t scratch[]) {
        InverseNTTWithBackend<Backend::Auto>(a, scratch);
    }

    template <Backend B = Backend::Auto>
    void InverseNTTWithBackend(uint64_t a[]) {
        std::array<uint64_t, N> scratch{};
        InverseNTTWithBackend<B>(a, scratch.data());
    }

    template <Backend B = Backend::Auto>
    void InverseNTTWithBackend(uint64_t a[], uint64_t scratch[]) {
        for (size_t i = 0; i < N; i++) {
            scratch[i] = a[gi[(N - i) % N] - 1];
        }

        ForwardMixedRadix23NTT<B>(scratch, a);

        for (size_t i = 0; i < N; i++) {
            a[i] = Z::MulConst(a[i], {omega_o_inv_.value[i], omega_o_inv_.shoup[i]});
        }

        InverseMixedRadix23NTT<B>(a, scratch);

        for (size_t i = 0; i < N; i++) {
            a[i] = scratch[gi_inv[i + 1]];
        }
    }

    static NTT &GetInstance() {
        static NTT instance;
        return instance;
    }

    template <Backend B = Backend::Scalar>
    std::array<uint64_t, N> ComputeOmegaOTableValuesForTesting() {
        std::array<uint64_t, N> seed{};
        std::array<uint64_t, N> values{};
        uint64_t t = omega_O;
        for (size_t i = 1; i <= N; ++i) {
            seed[(N - gi_inv[i]) % N] = t;
            t = Z::Mul(t, omega_O);
        }
        ForwardMixedRadix23NTT<B>(seed.data(), values.data());

        const uint64_t n_inv = Z::Pow(N, p - 2);
        for (size_t i = 0; i < N; ++i) {
            values[i] = Z::Mul(values[i], n_inv);
        }
        return values;
    }

    const ConstMulTable<N> &OmegaOForwardTableForTesting() const {
        return omega_o_fwd_;
    }

private:
    static inline uint64_t MulConstRaw(uint64_t a, uint64_t value, uint64_t shoup) {
        return Z::MulConst(a, {value, shoup});
    }

    // scalar kernel
    void MixedRadix23NTTScalar(uint64_t __restrict__ a[], uint64_t __restrict__ b[], const ConstMulTable<N> &omega_table) {
        const auto &omega = omega_table.value;
        const auto &omega_shoup = omega_table.shoup;
        for (size_t i = 0; i < N; i++) {
            b[i] = a[digit_reverse_table[i]];
        }

        for (size_t i = 0; i < N; i += 2) {
            uint64_t t = b[i + 1];
            b[i + 1] = Z::Sub(b[i], t);
            b[i] = Z::Add(b[i], t);
        }

        size_t l0 = 1;
        size_t l1 = 2;
        size_t d = N / 2;

        for (size_t i = 1; i < u; i++) {
            l0 = (size_t)1 << i;
            l1 = (size_t)1 << (i + 1);
            d = N >> (i + 1);
            for (size_t j = 0; j < N; j += l1) {
                for (size_t k = 0; k < l0; k++) {
                    uint64_t t = MulConstRaw(b[j + l0 + k], omega[k * d], omega_shoup[k * d]);
                    b[j + l0 + k] = Z::Sub(b[j + k], t);
                    b[j + k] = Z::Add(b[j + k], t);
                }
            }
        }

        uint64_t z3 = omega[N / 3];
        uint64_t z3_shoup = omega_shoup[N / 3];
        uint64_t zz3 = omega[2 * N / 3];
        uint64_t zz3_shoup = omega_shoup[2 * N / 3];

        for (size_t i = 0; i < v; i++) {
            l0 = U;
            l1 = U * 3;
            for (size_t j = 0; j < i; j++) {
                l0 *= 3;
                l1 *= 3;
            }
            d = N / l1;
            for (size_t j = 0; j < N; j += l1) {
                for (size_t k = 0; k < l0; k++) {
                    uint64_t y1 = MulConstRaw(b[j + l0 + k], omega[k * d], omega_shoup[k * d]);
                    uint64_t y2 = MulConstRaw(b[j + l0 + l0 + k], omega[2 * k * d], omega_shoup[2 * k * d]);
                    uint64_t y0 = Z::Add(y1, y2);
                    uint64_t t = Z::Add(MulConstRaw(y1, z3, z3_shoup), MulConstRaw(y2, zz3, zz3_shoup));
                    b[j + l0 + k] = Z::Add(b[j + k], t);
                    b[j + l0 + l0 + k] = Z::Sub(b[j + k], Z::Add(y0, t));
                    b[j + k] = Z::Add(b[j + k], y0);
                }
            }
        }

        std::copy(b, b + N, a);
    }

#if defined(BDF17_ENABLE_AVX512) && defined(__AVX512F__) && defined(__AVX512DQ__)
    static_assert(Z::kSimdFastPathSupported, "SIMD fast path requires p <= 2^62.");
    // AVX-512 kernel
    void MixedRadix23NTTAVX512(uint64_t __restrict__ a[], uint64_t __restrict__ b[], const ConstMulTable<N> &omega_table) {
        const auto &omega = omega_table.value;
        const auto &omega_shoup = omega_table.shoup;
        for (size_t i = 0; i < N; ++i) {
            b[i] = a[digit_reverse_table[i]];
        }

        for (size_t i = 0; i < N; i += 2) {
            const uint64_t t = b[i + 1];
            b[i + 1] = Z::Sub(b[i], t);
            b[i] = Z::Add(b[i], t);
        }

        const __m512i pV = _mm512_set1_epi64((int64_t)p);
        const __m512i p_1V = _mm512_set1_epi64((int64_t)(p - 1));

        size_t l0 = 1;
        size_t l1 = 2;
        size_t d = N / 2;

        for (size_t i = 1; i < u; ++i) {
            l0 = (size_t)1 << i;
            l1 = (size_t)1 << (i + 1);
            d = N >> (i + 1);
            for (size_t j = 0; j < N; j += l1) {
                size_t k = 0;
                for (; k + 7 < l0; k += 8) {
                    const __m512i omegaV = _mm512_set_epi64(
                        (int64_t)omega[(k + 7) * d], (int64_t)omega[(k + 6) * d], (int64_t)omega[(k + 5) * d], (int64_t)omega[(k + 4) * d],
                        (int64_t)omega[(k + 3) * d], (int64_t)omega[(k + 2) * d], (int64_t)omega[(k + 1) * d], (int64_t)omega[(k + 0) * d]);
                    const __m512i omegaMuV = _mm512_set_epi64(
                        (int64_t)omega_shoup[(k + 7) * d], (int64_t)omega_shoup[(k + 6) * d], (int64_t)omega_shoup[(k + 5) * d], (int64_t)omega_shoup[(k + 4) * d],
                        (int64_t)omega_shoup[(k + 3) * d], (int64_t)omega_shoup[(k + 2) * d], (int64_t)omega_shoup[(k + 1) * d], (int64_t)omega_shoup[(k + 0) * d]);

                    const __m512i bl0V = _mm512_loadu_si512((const void *)&b[j + l0 + k]);
                    const __m512i tV = bdf17::detail::simd::MulConstU64x8<Z>(bl0V, omegaV, omegaMuV);
                    const __m512i bV = _mm512_loadu_si512((const void *)&b[j + k]);

                    const __m512i addV = bdf17::detail::simd::AddModU64x8(bV, tV, pV, p_1V);
                    const __m512i subV = bdf17::detail::simd::SubModU64x8(bV, tV, pV);

                    _mm512_storeu_si512((void *)&b[j + k], addV);
                    _mm512_storeu_si512((void *)&b[j + l0 + k], subV);
                }
                for (; k < l0; ++k) {
                    const uint64_t t = MulConstRaw(b[j + l0 + k], omega[k * d], omega_shoup[k * d]);
                    b[j + l0 + k] = Z::Sub(b[j + k], t);
                    b[j + k] = Z::Add(b[j + k], t);
                }
            }
        }

        const uint64_t z3 = omega[N / 3];
        const uint64_t z3_shoup = omega_shoup[N / 3];
        const uint64_t zz3 = omega[2 * N / 3];
        const uint64_t zz3_shoup = omega_shoup[2 * N / 3];
        const __m512i z3V = _mm512_set1_epi64((int64_t)z3);
        const __m512i z3MuV = _mm512_set1_epi64((int64_t)z3_shoup);
        const __m512i zz3V = _mm512_set1_epi64((int64_t)zz3);
        const __m512i zz3MuV = _mm512_set1_epi64((int64_t)zz3_shoup);

        for (size_t i = 0; i < v; ++i) {
            l0 = U;
            l1 = U * 3;
            for (size_t j = 0; j < i; ++j) {
                l0 *= 3;
                l1 *= 3;
            }
            d = N / l1;
            for (size_t j = 0; j < N; j += l1) {
                size_t k = 0;
                for (; k + 7 < l0; k += 8) {
                    const __m512i omegaV = _mm512_set_epi64(
                        (int64_t)omega[(k + 7) * d], (int64_t)omega[(k + 6) * d], (int64_t)omega[(k + 5) * d], (int64_t)omega[(k + 4) * d],
                        (int64_t)omega[(k + 3) * d], (int64_t)omega[(k + 2) * d], (int64_t)omega[(k + 1) * d], (int64_t)omega[(k + 0) * d]);
                    const __m512i omegaMuV = _mm512_set_epi64(
                        (int64_t)omega_shoup[(k + 7) * d], (int64_t)omega_shoup[(k + 6) * d], (int64_t)omega_shoup[(k + 5) * d], (int64_t)omega_shoup[(k + 4) * d],
                        (int64_t)omega_shoup[(k + 3) * d], (int64_t)omega_shoup[(k + 2) * d], (int64_t)omega_shoup[(k + 1) * d], (int64_t)omega_shoup[(k + 0) * d]);
                    const __m512i omega2V = _mm512_set_epi64(
                        (int64_t)omega[2 * (k + 7) * d], (int64_t)omega[2 * (k + 6) * d], (int64_t)omega[2 * (k + 5) * d], (int64_t)omega[2 * (k + 4) * d],
                        (int64_t)omega[2 * (k + 3) * d], (int64_t)omega[2 * (k + 2) * d], (int64_t)omega[2 * (k + 1) * d], (int64_t)omega[2 * (k + 0) * d]);
                    const __m512i omega2MuV = _mm512_set_epi64(
                        (int64_t)omega_shoup[2 * (k + 7) * d], (int64_t)omega_shoup[2 * (k + 6) * d], (int64_t)omega_shoup[2 * (k + 5) * d], (int64_t)omega_shoup[2 * (k + 4) * d],
                        (int64_t)omega_shoup[2 * (k + 3) * d], (int64_t)omega_shoup[2 * (k + 2) * d], (int64_t)omega_shoup[2 * (k + 1) * d], (int64_t)omega_shoup[2 * (k + 0) * d]);

                    const __m512i b1V = _mm512_loadu_si512((const void *)&b[j + l0 + k]);
                    const __m512i b2V = _mm512_loadu_si512((const void *)&b[j + l0 + l0 + k]);

                    const __m512i y1V = bdf17::detail::simd::MulConstU64x8<Z>(b1V, omegaV, omegaMuV);
                    const __m512i y2V = bdf17::detail::simd::MulConstU64x8<Z>(b2V, omega2V, omega2MuV);

                    const __m512i y0V = bdf17::detail::simd::AddModU64x8(y1V, y2V, pV, p_1V);

                    __m512i tV = _mm512_add_epi64(
                        bdf17::detail::simd::MulConstU64x8<Z>(y1V, z3V, z3MuV),
                        bdf17::detail::simd::MulConstU64x8<Z>(y2V, zz3V, zz3MuV));
                    tV = bdf17::detail::simd::ReduceOnceU64x8(tV, pV, p_1V);

                    const __m512i ytV = bdf17::detail::simd::AddModU64x8(y0V, tV, pV, p_1V);

                    const __m512i b0V = _mm512_loadu_si512((const void *)&b[j + k]);

                    const __m512i addV = bdf17::detail::simd::AddModU64x8(b0V, tV, pV, p_1V);
                    const __m512i subV = bdf17::detail::simd::SubModU64x8(b0V, ytV, pV);
                    const __m512i bNewV = bdf17::detail::simd::AddModU64x8(b0V, y0V, pV, p_1V);

                    _mm512_storeu_si512((void *)&b[j + l0 + k], addV);
                    _mm512_storeu_si512((void *)&b[j + l0 + l0 + k], subV);
                    _mm512_storeu_si512((void *)&b[j + k], bNewV);
                }
                for (; k < l0; ++k) {
                    const uint64_t y1 = MulConstRaw(b[j + l0 + k], omega[k * d], omega_shoup[k * d]);
                    const uint64_t y2 = MulConstRaw(b[j + l0 + l0 + k], omega[2 * k * d], omega_shoup[2 * k * d]);
                    const uint64_t y0 = Z::Add(y1, y2);
                    const uint64_t t = Z::Add(
                        MulConstRaw(y1, z3, z3_shoup),
                        MulConstRaw(y2, zz3, zz3_shoup));
                    b[j + l0 + k] = Z::Add(b[j + k], t);
                    b[j + l0 + l0 + k] = Z::Sub(b[j + k], Z::Add(y0, t));
                    b[j + k] = Z::Add(b[j + k], y0);
                }
            }
        }

        std::copy(b, b + N, a);
    }
#endif

#if defined(BDF17_ENABLE_AVX2) && defined(__AVX2__)
    static_assert(Z::kSimdFastPathSupported, "SIMD fast path requires p <= 2^62.");
    // AVX2 kernel
    void MixedRadix23NTTAVX2(uint64_t __restrict__ a[], uint64_t __restrict__ b[], const ConstMulTable<N> &omega_table) {
        const auto &omega = omega_table.value;
        const auto &omega_shoup = omega_table.shoup;
        for (size_t i = 0; i < N; i++) {
            b[i] = a[digit_reverse_table[i]];
        }

        for (size_t i = 0; i < N; i += 2) {
            uint64_t t = b[i + 1];
            b[i + 1] = Z::Sub(b[i], t);
            b[i] = Z::Add(b[i], t);
        }

        size_t l0 = 1;
        size_t l1 = 2;
        size_t d = N / 2;

        __m256i pV = _mm256_set1_epi64x(p);
        __m256i p_1V = _mm256_set1_epi64x(p - 1);

        for (size_t i = 1; i < u; i++) {
            l0 = (size_t)1 << i;
            l1 = (size_t)1 << (i + 1);
            d = N >> (i + 1);
            for (size_t j = 0; j < N; j += l1) {
                size_t k = 0;
                for (; k + 3 < l0; k += 4) {
                    uint64_t t0 = MulConstRaw(b[j + l0 + k], omega[k * d], omega_shoup[k * d]);
                    uint64_t t1 = MulConstRaw(b[j + l0 + k + 1], omega[(k + 1) * d], omega_shoup[(k + 1) * d]);
                    uint64_t t2 = MulConstRaw(b[j + l0 + k + 2], omega[(k + 2) * d], omega_shoup[(k + 2) * d]);
                    uint64_t t3 = MulConstRaw(b[j + l0 + k + 3], omega[(k + 3) * d], omega_shoup[(k + 3) * d]);

                    __m256i tV = _mm256_set_epi64x(t3, t2, t1, t0);
                    __m256i bV = _mm256_loadu_si256((__m256i *)&b[j + k]);

                    __m256i addV = _mm256_add_epi64(bV, tV);
                    __m256i add_maskV = _mm256_cmpgt_epi64(addV, p_1V);
                    __m256i add_adjustV = _mm256_sub_epi64(addV, _mm256_and_si256(add_maskV, pV));
                    _mm256_storeu_si256((__m256i *)&b[j + k], add_adjustV);

                    __m256i subV = _mm256_sub_epi64(bV, tV);
                    __m256i sub_maskV = _mm256_cmpgt_epi64(tV, bV);
                    __m256i sub_adjustV = _mm256_add_epi64(subV, _mm256_and_si256(sub_maskV, pV));
                    _mm256_storeu_si256((__m256i *)&b[j + l0 + k], sub_adjustV);
                }
                for (; k < l0; k++) {
                    uint64_t t = MulConstRaw(b[j + l0 + k], omega[k * d], omega_shoup[k * d]);
                    b[j + l0 + k] = Z::Sub(b[j + k], t);
                    b[j + k] = Z::Add(b[j + k], t);
                }
            }
        }

        uint64_t z3 = omega[N / 3];
        uint64_t z3_shoup = omega_shoup[N / 3];
        uint64_t zz3 = omega[2 * N / 3];
        uint64_t zz3_shoup = omega_shoup[2 * N / 3];

        for (size_t i = 0; i < v; i++) {
            l0 = U;
            l1 = U * 3;
            for (size_t j = 0; j < i; j++) {
                l0 *= 3;
                l1 *= 3;
            }
            d = N / l1;
            for (size_t j = 0; j < N; j += l1) {
                size_t k = 0;
                for (; k + 3 < l0; k += 4) {
                    uint64_t y01 = MulConstRaw(b[j + l0 + k], omega[k * d], omega_shoup[k * d]);
                    uint64_t y02 = MulConstRaw(b[j + l0 + l0 + k], omega[2 * k * d], omega_shoup[2 * k * d]);
                    uint64_t y11 = MulConstRaw(b[j + l0 + k + 1], omega[(k + 1) * d], omega_shoup[(k + 1) * d]);
                    uint64_t y12 = MulConstRaw(b[j + l0 + l0 + k + 1], omega[2 * (k + 1) * d], omega_shoup[2 * (k + 1) * d]);
                    uint64_t y21 = MulConstRaw(b[j + l0 + k + 2], omega[(k + 2) * d], omega_shoup[(k + 2) * d]);
                    uint64_t y22 = MulConstRaw(b[j + l0 + l0 + k + 2], omega[2 * (k + 2) * d], omega_shoup[2 * (k + 2) * d]);
                    uint64_t y31 = MulConstRaw(b[j + l0 + k + 3], omega[(k + 3) * d], omega_shoup[(k + 3) * d]);
                    uint64_t y32 = MulConstRaw(b[j + l0 + l0 + k + 3], omega[2 * (k + 3) * d], omega_shoup[2 * (k + 3) * d]);
                    uint64_t y00 = y01 + y02;
                    uint64_t y10 = y11 + y12;
                    uint64_t y20 = y21 + y22;
                    uint64_t y30 = y31 + y32;
                    uint64_t t0 = MulConstRaw(y01, z3, z3_shoup) + MulConstRaw(y02, zz3, zz3_shoup);
                    uint64_t t1 = MulConstRaw(y11, z3, z3_shoup) + MulConstRaw(y12, zz3, zz3_shoup);
                    uint64_t t2 = MulConstRaw(y21, z3, z3_shoup) + MulConstRaw(y22, zz3, zz3_shoup);
                    uint64_t t3 = MulConstRaw(y31, z3, z3_shoup) + MulConstRaw(y32, zz3, zz3_shoup);

                    __m256i tV = _mm256_set_epi64x(t3, t2, t1, t0);
                    __m256i t_maskV = _mm256_cmpgt_epi64(tV, p_1V);
                    tV = _mm256_sub_epi64(tV, _mm256_and_si256(t_maskV, pV));

                    __m256i yV = _mm256_set_epi64x(y30, y20, y10, y00);
                    __m256i y_maskV = _mm256_cmpgt_epi64(yV, p_1V);
                    yV = _mm256_sub_epi64(yV, _mm256_and_si256(y_maskV, pV));

                    __m256i ytV = _mm256_add_epi64(yV, tV);
                    __m256i yt_maskV = _mm256_cmpgt_epi64(ytV, p_1V);
                    ytV = _mm256_sub_epi64(ytV, _mm256_and_si256(yt_maskV, pV));

                    __m256i bV = _mm256_loadu_si256((__m256i *)&b[j + k]);

                    __m256i addV = _mm256_add_epi64(bV, tV);
                    __m256i add_maskV = _mm256_cmpgt_epi64(addV, p_1V);
                    __m256i add_adjustV = _mm256_sub_epi64(addV, _mm256_and_si256(add_maskV, pV));
                    _mm256_storeu_si256((__m256i *)&b[j + k + l0], add_adjustV);

                    __m256i subV = _mm256_sub_epi64(bV, ytV);
                    __m256i sub_maskV = _mm256_cmpgt_epi64(ytV, bV);
                    __m256i sub_adjustV = _mm256_add_epi64(subV, _mm256_and_si256(sub_maskV, pV));
                    _mm256_storeu_si256((__m256i *)&b[j + k + l0 + l0], sub_adjustV);

                    __m256i b0V = _mm256_add_epi64(bV, yV);
                    __m256i b0_maskV = _mm256_cmpgt_epi64(b0V, p_1V);
                    __m256i b0_adjustV = _mm256_sub_epi64(b0V, _mm256_and_si256(b0_maskV, pV));
                    _mm256_storeu_si256((__m256i *)&b[j + k], b0_adjustV);
                }
                for (; k < l0; k++) {
                    uint64_t y1 = MulConstRaw(b[j + l0 + k], omega[k * d], omega_shoup[k * d]);
                    uint64_t y2 = MulConstRaw(b[j + l0 + l0 + k], omega[2 * k * d], omega_shoup[2 * k * d]);
                    uint64_t y0 = Z::Add(y1, y2);
                    uint64_t t = Z::Add(MulConstRaw(y1, z3, z3_shoup), MulConstRaw(y2, zz3, zz3_shoup));
                    b[j + l0 + k] = Z::Add(b[j + k], t);
                    b[j + l0 + l0 + k] = Z::Sub(b[j + k], Z::Add(y0, t));
                    b[j + k] = Z::Add(b[j + k], y0);
                }
            }
        }

        std::copy(b, b + N, a);
    }
#endif

    void MixedRadix23NTT(uint64_t __restrict__ a[], uint64_t __restrict__ b[], const ConstMulTable<N> &omega_table) {
        MixedRadix23NTTWithBackend<Backend::Auto>(a, b, omega_table);
    }

    template <Backend B = Backend::Auto>
    void MixedRadix23NTTWithBackend(uint64_t __restrict__ a[], uint64_t __restrict__ b[], const ConstMulTable<N> &omega_table) {
        static_assert(N >= 2, "MixedRadix23 NTT kernel requires N >= 2 (equivalently O >= 3)");
        if constexpr (B == Backend::Scalar) {
            MixedRadix23NTTScalar(a, b, omega_table);
        } else if constexpr (B == Backend::Avx512) {
#if defined(BDF17_ENABLE_AVX512) && defined(__AVX512F__) && defined(__AVX512DQ__)
            MixedRadix23NTTAVX512(a, b, omega_table);
#elif defined(BDF17_ENABLE_AVX2) && defined(__AVX2__)
            MixedRadix23NTTAVX2(a, b, omega_table);
#else
            MixedRadix23NTTScalar(a, b, omega_table);
#endif
        } else if constexpr (B == Backend::Avx2) {
#if defined(BDF17_ENABLE_AVX2) && defined(__AVX2__)
            MixedRadix23NTTAVX2(a, b, omega_table);
#else
            MixedRadix23NTTScalar(a, b, omega_table);
#endif
        } else {
#if defined(BDF17_ENABLE_RUNTIME_BACKEND_DISPATCH)
            switch (ResolveAutoBackend()) {
                case Backend::Avx512:
#if defined(BDF17_ENABLE_AVX512) && defined(__AVX512F__) && defined(__AVX512DQ__)
                    MixedRadix23NTTAVX512(a, b, omega_table);
                    return;
#endif
                    break;
                case Backend::Avx2:
#if defined(BDF17_ENABLE_AVX2) && defined(__AVX2__)
                    MixedRadix23NTTAVX2(a, b, omega_table);
                    return;
#endif
                    break;
                case Backend::Scalar:
                case Backend::Auto:
                    break;
            }
            MixedRadix23NTTScalar(a, b, omega_table);
#else
#if defined(BDF17_ENABLE_AVX512) && defined(__AVX512F__) && defined(__AVX512DQ__)
            MixedRadix23NTTAVX512(a, b, omega_table);
#elif defined(BDF17_ENABLE_AVX2) && defined(__AVX2__)
            MixedRadix23NTTAVX2(a, b, omega_table);
#else
            MixedRadix23NTTScalar(a, b, omega_table);
#endif
#endif
        }
    }

    void ForwardMixedRadix23NTT(uint64_t a[], uint64_t b[]) {
        ForwardMixedRadix23NTT<Backend::Auto>(a, b);
    }

    template <Backend B = Backend::Auto>
    void ForwardMixedRadix23NTT(uint64_t a[], uint64_t b[]) {
        MixedRadix23NTTWithBackend<B>(a, b, omega_n_fwd_);
    }

    void InverseMixedRadix23NTT(uint64_t a[], uint64_t b[]) {
        InverseMixedRadix23NTT<Backend::Auto>(a, b);
    }

    template <Backend B = Backend::Auto>
    void InverseMixedRadix23NTT(uint64_t a[], uint64_t b[]) {
        MixedRadix23NTTWithBackend<B>(a, b, omega_n_inv_);
    }

    void ComputeOmegaNTable() {
        omega_n_fwd_.value[0] = 1;
        for (size_t i = 1; i < N; i++) {
            omega_n_fwd_.value[i] = Z::Mul(omega_n_fwd_.value[i - 1], omega_N);
        }
        omega_n_inv_.value[0] = 1;
        for (size_t i = 1; i < N; i++) {
            omega_n_inv_.value[i] = omega_n_fwd_.value[N - i];
        }
        for (size_t i = 0; i < N; i++) {
            omega_n_fwd_.shoup[i] = Z::MakeConstMultiplier(omega_n_fwd_.value[i]).shoup;
            omega_n_inv_.shoup[i] = Z::MakeConstMultiplier(omega_n_inv_.value[i]).shoup;
        }
    }

    void ComputeOmegaOTable() {
        std::array<uint64_t, N> seed{};
        uint64_t t = omega_O;
        for (size_t i = 1; i <= N; i++) {
            seed[(N - gi_inv[i]) % N] = t;
            t = Z::Mul(t, omega_O);
        }
        ForwardMixedRadix23NTT<Backend::Scalar>(seed.data(), omega_o_fwd_.value.data());

        uint64_t N_inv = Z::Pow(N, p - 2);
        for (size_t i = 0; i < N; i++) {
            omega_o_inv_.value[i] = Z::Pow(omega_o_fwd_.value[i], p - 2);
            omega_o_fwd_.value[i] = Z::Mul(omega_o_fwd_.value[i], N_inv);
            omega_o_inv_.value[i] = Z::Mul(omega_o_inv_.value[i], N_inv);
            omega_o_fwd_.shoup[i] = Z::MakeConstMultiplier(omega_o_fwd_.value[i]).shoup;
            omega_o_inv_.shoup[i] = Z::MakeConstMultiplier(omega_o_inv_.value[i]).shoup;
        }
    }

    [[nodiscard]] bool ValidateParams() const {
        if (N != U * V) {
            return false;
        }
        if (Z::Pow(omega_O, O) != 1) {
            return false;
        }
        if (Z::Pow(omega_N, N) != 1) {
            return false;
        }
        for (size_t i = 1; i <= N; ++i) {
            if (gi_inv[i] >= N) {
                return false;
            }
            if (gi[gi_inv[i]] != i) {
                return false;
            }
        }
        std::array<bool, N + 1> seen{};
        for (size_t i = 0; i < N; ++i) {
            const size_t value = gi[i];
            if (value == 0 || value > N || seen[value]) {
                return false;
            }
            seen[value] = true;
        }
        if (!Z::kScalarFastPathSupported) {
            return false;
        }
        return true;
    }

    NTT() {
#ifdef BDF17_VALIDATE_NTT_PARAMS
        if (!ValidateParams()) {
            throw "NTT parameter validation failed";
        }
#endif
        ComputeOmegaNTable();
        ComputeOmegaOTable();
    }

    ConstMulTable<N> omega_n_fwd_;
    ConstMulTable<N> omega_n_inv_;
    ConstMulTable<N> omega_o_fwd_;
    ConstMulTable<N> omega_o_inv_;
};

// wrappers (CircNTT, TensorNTTImpl)
template <uint64_t p_, uint64_t g_, size_t O_, size_t w_>
class CircNTT {
public:
    using PrimitiveNTT = NTT<p_, g_, O_, w_>;

    constexpr static uint64_t p = p_;
    constexpr static uint64_t g = g_;
    constexpr static size_t O = O_;
    constexpr static size_t w = w_;
    constexpr static size_t N = O;

    using Z = Zp<p>;

    constexpr static uint64_t N_inv = Z::Pow(N, p - 2);

    void ForwardNTT(uint64_t a[]) {
        ForwardNTTWithBackend<Backend::Auto>(a);
    }

    void ForwardNTT(uint64_t a[], uint64_t scratch[]) {
        ForwardNTTWithBackend<Backend::Auto>(a, scratch);
    }

    template <Backend B = Backend::Auto>
    void ForwardNTTWithBackend(uint64_t a[]) {
        std::array<uint64_t, PrimitiveNTT::N> scratch{};
        ForwardNTTWithBackend<B>(a, scratch.data());
    }

    template <Backend B = Backend::Auto>
    void ForwardNTTWithBackend(uint64_t a[], uint64_t scratch[]) {
        auto t = a[0];
        for (size_t i = 1; i < N; i++) {
            a[0] = Z::Add(a[0], a[i]);
        }

        PrimitiveNTT::GetInstance().template ForwardNTTWithBackend<B>(a + 1, scratch);

        for (size_t i = 1; i < N; i++) {
            a[i] = Z::Add(a[i], t);
        }
    }

    void InverseNTT(uint64_t a[]) {
        InverseNTTWithBackend<Backend::Auto>(a);
    }

    void InverseNTT(uint64_t a[], uint64_t scratch[]) {
        InverseNTTWithBackend<Backend::Auto>(a, scratch);
    }

    template <Backend B = Backend::Auto>
    void InverseNTTWithBackend(uint64_t a[]) {
        std::array<uint64_t, PrimitiveNTT::N> scratch{};
        InverseNTTWithBackend<B>(a, scratch.data());
    }

    template <Backend B = Backend::Auto>
    void InverseNTTWithBackend(uint64_t a[], uint64_t scratch[]) {
        PrimitiveNTT::GetInstance().template InverseNTTWithBackend<B>(a + 1, scratch);

        auto t = a[0];
        for (size_t i = 1; i < N; i++) {
            t = Z::Sub(t, a[i]);
        }
        a[0] = Z::Mul(t, N_inv);
        for (size_t i = 1; i < N; i++) {
            a[i] = Z::Add(a[i], a[0]);
        }
    }

    static CircNTT &GetInstance() {
        static CircNTT instance;
        return instance;
    }

private:
    CircNTT() = default;
};

template <typename NTTp_, typename NTTq_>
class TensorNTTImpl {
public:
    using NTTp = NTTp_;
    using NTTq = NTTq_;

    constexpr static size_t N = NTTp::N * NTTq::N;
    constexpr static size_t O = NTTp::O * NTTq::O;

    static_assert(NTTp::p == NTTq::p, "p must be the same");
    constexpr static uint64_t p = NTTp::p;

    using Z = Zp<p>;

    static_assert(NTTp::g == NTTq::g, "g must be the same");

    void ForwardNTT(uint64_t a[]) {
        ForwardNTTWithBackend<Backend::Auto>(a);
    }

    void ForwardNTT(uint64_t a[], uint64_t scratch[]) {
        ForwardNTTWithBackend<Backend::Auto>(a, scratch);
    }

    template <Backend B = Backend::Auto>
    void ForwardNTTWithBackend(uint64_t a[]) {
        auto scratch = std::make_unique<uint64_t[]>(N);
        ForwardNTTWithBackend<B>(a, scratch.get());
    }

    template <Backend B = Backend::Auto>
    void ForwardNTTWithBackend(uint64_t a[], uint64_t scratch[]) {
        for (size_t i = 0; i < NTTp::N; i++) {
            auto *b = scratch + NTTq::N * i;
            for (size_t j = 0; j < NTTq::N; j++) {
                b[j] = a[NTTq::N * i + j];
            }
            NTTq::GetInstance().template ForwardNTTWithBackend<B>(b);
        }

        std::copy(scratch, scratch + N, a);

        for (size_t j = 0; j < NTTq::N; j++) {
            auto *b = scratch + NTTp::N * j;
            for (size_t i = 0; i < NTTp::N; i++) {
                b[i] = a[NTTq::N * i + j];
            }
            NTTp::GetInstance().template ForwardNTTWithBackend<B>(b);
        }
        for (size_t j = 0; j < NTTq::N; j++) {
            for (size_t i = 0; i < NTTp::N; i++) {
                a[NTTq::N * i + j] = scratch[NTTp::N * j + i];
            }
        }
    }

    void InverseNTT(uint64_t a[]) {
        InverseNTTWithBackend<Backend::Auto>(a);
    }

    void InverseNTT(uint64_t a[], uint64_t scratch[]) {
        InverseNTTWithBackend<Backend::Auto>(a, scratch);
    }

    template <Backend B = Backend::Auto>
    void InverseNTTWithBackend(uint64_t a[]) {
        auto scratch = std::make_unique<uint64_t[]>(N);
        InverseNTTWithBackend<B>(a, scratch.get());
    }

    template <Backend B = Backend::Auto>
    void InverseNTTWithBackend(uint64_t a[], uint64_t scratch[]) {
        for (size_t j = 0; j < NTTq::N; j++) {
            auto *b = scratch + NTTp::N * j;
            for (size_t i = 0; i < NTTp::N; i++) {
                b[i] = a[NTTq::N * i + j];
            }
            NTTp::GetInstance().template InverseNTTWithBackend<B>(b);
        }
        for (size_t j = 0; j < NTTq::N; j++) {
            for (size_t i = 0; i < NTTp::N; i++) {
                a[NTTq::N * i + j] = scratch[NTTp::N * j + i];
            }
        }

        for (size_t i = 0; i < NTTp::N; i++) {
            auto *b = scratch + NTTq::N * i;
            for (size_t j = 0; j < NTTq::N; j++) {
                b[j] = a[NTTq::N * i + j];
            }
            NTTq::GetInstance().template InverseNTTWithBackend<B>(b);
        }

        std::copy(scratch, scratch + N, a);
    }

    static TensorNTTImpl &GetInstance() {
        static TensorNTTImpl instance;
        return instance;
    }

private:
    TensorNTTImpl() = default;
};

#endif // NTT_H
