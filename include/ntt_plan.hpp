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
        Transform::GetInstance().template ForwardNTTWithBackend<B>(a, Scratch(workspace));
    }

    static void InverseInPlace(uint64_t *a, Workspace &workspace) {
        Transform::GetInstance().template InverseNTTWithBackend<B>(a, Scratch(workspace));
    }
};

#endif // NTT_PLAN_HPP
