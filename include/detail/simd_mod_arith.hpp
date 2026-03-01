#ifndef BDF17_DETAIL_SIMD_MOD_ARITH_HPP
#define BDF17_DETAIL_SIMD_MOD_ARITH_HPP

#include <cstdint>

#if defined(BDF17_ENABLE_AVX512) && defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
#endif

namespace bdf17::detail::simd {

#if defined(BDF17_ENABLE_AVX512) && defined(__AVX512F__) && defined(__AVX512DQ__)
inline __m512i MulHiU64x8(__m512i x, __m512i y) {
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

inline __m512i ReduceOnceU64x8(__m512i x, __m512i p, __m512i p_minus_1) {
    const __mmask8 reduce_mask = _mm512_cmpgt_epi64_mask(x, p_minus_1);
    return _mm512_mask_sub_epi64(x, reduce_mask, x, p);
}

inline __m512i AddModU64x8(__m512i a, __m512i b, __m512i p, __m512i p_minus_1) {
    return ReduceOnceU64x8(_mm512_add_epi64(a, b), p, p_minus_1);
}

inline __m512i SubModU64x8(__m512i a, __m512i b, __m512i p) {
    const __m512i diff = _mm512_sub_epi64(a, b);
    const __mmask8 underflow_mask = _mm512_cmpgt_epi64_mask(b, a);
    return _mm512_mask_add_epi64(diff, underflow_mask, diff, p);
}

template <class Z>
inline __m512i MulConstU64x8(__m512i x, __m512i y, __m512i y_shoup) {
    const __m512i p_vec = _mm512_set1_epi64(static_cast<int64_t>(Z::p));
    const __m512i p_minus_1_vec = _mm512_set1_epi64(static_cast<int64_t>(Z::p - 1));
    const __m512i low_prod = _mm512_mullo_epi64(x, y);
    __m512i q = MulHiU64x8(x, y_shoup);
    q = _mm512_mullo_epi64(q, p_vec);
    const __m512i r = _mm512_sub_epi64(low_prod, q);
    return ReduceOnceU64x8(r, p_vec, p_minus_1_vec);
}
#endif

} // namespace bdf17::detail::simd

#endif // BDF17_DETAIL_SIMD_MOD_ARITH_HPP
