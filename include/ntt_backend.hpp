#ifndef NTT_BACKEND_HPP
#define NTT_BACKEND_HPP

enum class Backend {
    // Auto picks the best backend compiled into this binary (AVX512, then AVX2, then scalar).
    // It does not perform runtime CPU feature detection.
    Auto,
    Scalar,
    Avx2,
    Avx512
};

#if defined(BDF17_ENABLE_AVX2) && defined(__AVX2__)
constexpr bool kAvx2BackendCompiled = true;
#else
constexpr bool kAvx2BackendCompiled = false;
#endif

#if defined(BDF17_ENABLE_AVX512) && defined(__AVX512F__) && defined(__AVX512DQ__)
constexpr bool kAvx512BackendCompiled = true;
#else
constexpr bool kAvx512BackendCompiled = false;
#endif

template <Backend B>
constexpr bool BackendCompiled() {
    if constexpr (B == Backend::Scalar) {
        return true;
    }
    if constexpr (B == Backend::Avx2) {
        return kAvx2BackendCompiled;
    }
    if constexpr (B == Backend::Avx512) {
        return kAvx512BackendCompiled;
    }
    return true;
}

#endif // NTT_BACKEND_HPP
