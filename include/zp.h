#ifndef ZP_H
#define ZP_H

#include <cstdint>

#if defined(BDF17_ENABLE_AVX512) && defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
#endif

template <uint64_t p_>
class Zp {
public:

    constexpr static uint64_t p = p_;

    constexpr static uint64_t Add(uint64_t a, uint64_t b) {
        uint64_t sum = a + b;
        return sum >= p ? sum - p : sum;
    }

    constexpr static uint64_t Sub(uint64_t a, uint64_t b) {
        return a >= b ? a - b : p - b + a;
    }

    constexpr static uint64_t Mul(uint64_t a, uint64_t b) {
        return (uint64_t)((__uint128_t)a * b % p);
    }

    constexpr static uint64_t MulFastConst(uint64_t a, uint64_t b, uint64_t b_mu) {
        uint64_t q = (uint64_t)((__uint128_t)a * b_mu >> 64);
        uint64_t prod = a * b - q * p;
        return prod >= p ? prod - p : prod;
    }

    constexpr static uint64_t ComputeBarrettFactor(uint64_t x) {
        return (uint64_t)(((__uint128_t(x) << 64) / p));
    }

    constexpr static uint64_t Pow(uint64_t x, uint64_t e) {
        uint64_t res = 1;
        uint64_t base = x;

        while (e > 0) {
            if (e & 1) {
                res = Mul(res, base);
            }
            base = Mul(base, base);
            e >>= 1;
        }
        return res;
    }

#if defined(BDF17_ENABLE_AVX512) && defined(__AVX512F__) && defined(__AVX512DQ__)
    static inline __m512i MulHi512(__m512i x, __m512i y) {
        // 64x64->128 high lane emulation using 32-bit partial products.
        const __m512i lo_mask = _mm512_set1_epi64(0x00000000ffffffffULL);

        const __m512i x_hi = _mm512_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);
        const __m512i y_hi = _mm512_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);
        const __m512i z_lo_lo = _mm512_mul_epu32(x, y);
        const __m512i z_lo_hi = _mm512_mul_epu32(x, y_hi);
        const __m512i z_hi_lo = _mm512_mul_epu32(x_hi, y);
        const __m512i z_hi_hi = _mm512_mul_epu32(x_hi, y_hi);

        const __m512i z_lo_lo_shift = _mm512_srli_epi64(z_lo_lo, 32);
        const __m512i sum_tmp = _mm512_add_epi64(z_lo_hi, z_lo_lo_shift);
        const __m512i sum_lo = _mm512_and_si512(sum_tmp, lo_mask);
        const __m512i sum_mid = _mm512_srli_epi64(sum_tmp, 32);

        const __m512i sum_mid2 = _mm512_add_epi64(z_hi_lo, sum_lo);
        const __m512i sum_mid2_hi = _mm512_srli_epi64(sum_mid2, 32);
        const __m512i sum_hi = _mm512_add_epi64(z_hi_hi, sum_mid);
        return _mm512_add_epi64(sum_hi, sum_mid2_hi);
    }

    static inline __m512i MulConst512(__m512i x, __m512i y, __m512i y_mu) {
        // Barrett: (x*y - floor(x*y_mu / 2^64)*p) mod p
        const __m512i pV = _mm512_set1_epi64((int64_t)p);
        const __m512i xyV = _mm512_mullo_epi64(x, y);
        __m512i qV = MulHi512(x, y_mu);
        qV = _mm512_mullo_epi64(qV, pV);

        const __m512i subV = _mm512_sub_epi64(xyV, qV);
        return _mm512_min_epu64(subV, _mm512_sub_epi64(subV, pV));
    }
#endif
};

#endif // ZP_H
