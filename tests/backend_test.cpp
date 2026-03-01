#include <gtest/gtest.h>

#include "ntt_backend.hpp"

namespace {

TEST(Backend, BestCompiledBackendMatchesCompileFlags) {
    if constexpr (kAvx512BackendCompiled) {
        EXPECT_EQ(BestCompiledBackend(), Backend::Avx512);
    } else if constexpr (kAvx2BackendCompiled) {
        EXPECT_EQ(BestCompiledBackend(), Backend::Avx2);
    } else {
        EXPECT_EQ(BestCompiledBackend(), Backend::Scalar);
    }
}

TEST(Backend, ResolveAutoBackendReturnsCompiledBackend) {
    const Backend resolved = ResolveAutoBackend();
    switch (resolved) {
        case Backend::Scalar:
            EXPECT_TRUE(BackendCompiled<Backend::Scalar>());
            break;
        case Backend::Avx2:
            EXPECT_TRUE(kAvx2BackendCompiled);
            break;
        case Backend::Avx512:
            EXPECT_TRUE(kAvx512BackendCompiled);
            break;
        case Backend::Auto:
            FAIL() << "ResolveAutoBackend() must return a concrete backend";
            break;
    }
}

TEST(Backend, BestRuntimeBackendRespectsCompiledBackends) {
    const Backend runtime_best = BestRuntimeBackend();
    if (!kAvx2BackendCompiled) {
        EXPECT_NE(runtime_best, Backend::Avx2);
    }
    if (!kAvx512BackendCompiled) {
        EXPECT_NE(runtime_best, Backend::Avx512);
    }
}

#if !defined(BDF17_ENABLE_RUNTIME_BACKEND_DISPATCH)
TEST(Backend, ResolveAutoMatchesBestCompiledWhenRuntimeDispatchIsDisabled) {
    EXPECT_EQ(ResolveAutoBackend(), BestCompiledBackend());
}
#endif

} // namespace
