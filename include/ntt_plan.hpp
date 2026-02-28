#ifndef NTT_PLAN_HPP
#define NTT_PLAN_HPP

#include <algorithm>
#include <cstdint>

#include "typed_poly.hpp"

enum class Backend {
    Auto,
    Scalar,
    Avx2,
    Avx512
};

template <class Transform, Backend B = Backend::Auto>
class CanonicalNttPlan {
public:
    EvalPoly<Transform> forward(const CoeffPoly<Transform> &coeff) const {
        EvalPoly<Transform> out;
        std::copy_n(coeff.data(), Transform::N, out.data());
        ForwardInPlace(out.data());
        return out;
    }

    CoeffPoly<Transform> inverse(const EvalPoly<Transform> &eval) const {
        CoeffPoly<Transform> out;
        std::copy_n(eval.data(), Transform::N, out.data());
        InverseInPlace(out.data());
        return out;
    }

private:
    static void ForwardInPlace(uint64_t *a) {
        if constexpr (requires(uint64_t *ptr) { Transform::GetInstance().ForwardNTT(ptr); }) {
            Transform::GetInstance().ForwardNTT(a);
        } else {
            Transform::ForwardNTT(a);
        }
    }

    static void InverseInPlace(uint64_t *a) {
        if constexpr (requires(uint64_t *ptr) { Transform::GetInstance().InverseNTT(ptr); }) {
            Transform::GetInstance().InverseNTT(a);
        } else {
            Transform::InverseNTT(a);
        }
    }
};

#endif // NTT_PLAN_HPP
