#ifndef NTT_BACKEND_HPP
#define NTT_BACKEND_HPP

enum class Backend {
    // Auto picks the best backend compiled into this binary (AVX512, then AVX2, then scalar).
    // If BDF17_ENABLE_RUNTIME_BACKEND_DISPATCH is defined, Auto resolves at runtime to the
    // best compiled backend supported by the current CPU.
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

constexpr Backend BestCompiledBackend() noexcept {
    if constexpr (kAvx512BackendCompiled) {
        return Backend::Avx512;
    }
    if constexpr (kAvx2BackendCompiled) {
        return Backend::Avx2;
    }
    return Backend::Scalar;
}

inline bool RuntimeSupportsAvx2() noexcept {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

inline bool RuntimeSupportsAvx512() noexcept {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
#else
    return false;
#endif
}

inline Backend BestRuntimeBackend() noexcept {
    if constexpr (kAvx512BackendCompiled) {
        if (RuntimeSupportsAvx512()) {
            return Backend::Avx512;
        }
    }
    if constexpr (kAvx2BackendCompiled) {
        if (RuntimeSupportsAvx2()) {
            return Backend::Avx2;
        }
    }
    return Backend::Scalar;
}

inline Backend ResolveAutoBackend() noexcept {
#if defined(BDF17_ENABLE_RUNTIME_BACKEND_DISPATCH)
    return BestRuntimeBackend();
#else
    return BestCompiledBackend();
#endif
}

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
