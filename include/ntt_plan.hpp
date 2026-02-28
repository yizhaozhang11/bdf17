#ifndef NTT_PLAN_HPP
#define NTT_PLAN_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

#include "ntt_backend.hpp"
#include "typed_poly.hpp"

template <class Transform, Backend B = Backend::Auto>
class CanonicalNttPlan {
public:
    struct Workspace {
        std::vector<uint64_t> scratch;
    };

    EvalPoly<Transform> forward(const CoeffPoly<Transform> &coeff) const {
        Workspace workspace;
        return forward(coeff, workspace);
    }

    EvalPoly<Transform> forward(const CoeffPoly<Transform> &coeff, Workspace &workspace) const {
        EvalPoly<Transform> out;
        std::copy_n(coeff.data(), Transform::N, out.data());
        ForwardInPlace(out.data(), workspace);
        return out;
    }

    CoeffPoly<Transform> inverse(const EvalPoly<Transform> &eval) const {
        Workspace workspace;
        return inverse(eval, workspace);
    }

    CoeffPoly<Transform> inverse(const EvalPoly<Transform> &eval, Workspace &workspace) const {
        CoeffPoly<Transform> out;
        std::copy_n(eval.data(), Transform::N, out.data());
        InverseInPlace(out.data(), workspace);
        return out;
    }

private:
    static uint64_t *Scratch(Workspace &workspace) {
        if (workspace.scratch.size() < Transform::N) {
            workspace.scratch.resize(Transform::N);
        }
        return workspace.scratch.data();
    }

    static void ForwardInPlace(uint64_t *a, Workspace &workspace) {
        if constexpr (requires(uint64_t *ptr, uint64_t *scratch) { Transform::GetInstance().template ForwardNTTWithBackend<B>(ptr, scratch); }) {
            Transform::GetInstance().template ForwardNTTWithBackend<B>(a, Scratch(workspace));
        } else if constexpr (requires(uint64_t *ptr, uint64_t *scratch) { Transform::template ForwardNTTWithBackend<B>(ptr, scratch); }) {
            Transform::template ForwardNTTWithBackend<B>(a, Scratch(workspace));
        } else if constexpr (requires(uint64_t *ptr) { Transform::GetInstance().template ForwardNTTWithBackend<B>(ptr); }) {
            Transform::GetInstance().template ForwardNTTWithBackend<B>(a);
        } else if constexpr (requires(uint64_t *ptr) { Transform::template ForwardNTTWithBackend<B>(ptr); }) {
            Transform::template ForwardNTTWithBackend<B>(a);
        } else if constexpr (requires(uint64_t *ptr, uint64_t *scratch) { Transform::GetInstance().ForwardNTT(ptr, scratch); }) {
            Transform::GetInstance().ForwardNTT(a, Scratch(workspace));
        } else if constexpr (requires(uint64_t *ptr, uint64_t *scratch) { Transform::ForwardNTT(ptr, scratch); }) {
            Transform::ForwardNTT(a, Scratch(workspace));
        } else if constexpr (requires(uint64_t *ptr) { Transform::GetInstance().ForwardNTT(ptr); }) {
            Transform::GetInstance().ForwardNTT(a);
        } else {
            Transform::ForwardNTT(a);
        }
    }

    static void InverseInPlace(uint64_t *a, Workspace &workspace) {
        if constexpr (requires(uint64_t *ptr, uint64_t *scratch) { Transform::GetInstance().template InverseNTTWithBackend<B>(ptr, scratch); }) {
            Transform::GetInstance().template InverseNTTWithBackend<B>(a, Scratch(workspace));
        } else if constexpr (requires(uint64_t *ptr, uint64_t *scratch) { Transform::template InverseNTTWithBackend<B>(ptr, scratch); }) {
            Transform::template InverseNTTWithBackend<B>(a, Scratch(workspace));
        } else if constexpr (requires(uint64_t *ptr) { Transform::GetInstance().template InverseNTTWithBackend<B>(ptr); }) {
            Transform::GetInstance().template InverseNTTWithBackend<B>(a);
        } else if constexpr (requires(uint64_t *ptr) { Transform::template InverseNTTWithBackend<B>(ptr); }) {
            Transform::template InverseNTTWithBackend<B>(a);
        } else if constexpr (requires(uint64_t *ptr, uint64_t *scratch) { Transform::GetInstance().InverseNTT(ptr, scratch); }) {
            Transform::GetInstance().InverseNTT(a, Scratch(workspace));
        } else if constexpr (requires(uint64_t *ptr, uint64_t *scratch) { Transform::InverseNTT(ptr, scratch); }) {
            Transform::InverseNTT(a, Scratch(workspace));
        } else if constexpr (requires(uint64_t *ptr) { Transform::GetInstance().InverseNTT(ptr); }) {
            Transform::GetInstance().InverseNTT(a);
        } else {
            Transform::InverseNTT(a);
        }
    }
};

#endif // NTT_PLAN_HPP
