#ifndef BDF17_EXPCRT_HPP
#define BDF17_EXPCRT_HPP

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "experiment_config.hpp"
#include "expcrt_variant.hpp"
#include "params.hpp"
#include "typed_poly.hpp"

namespace bdf17 {

template <typename PolyPT, typename PolyQT>
void Tensor(
    EvalPoly<TensorNTTImpl<PolyPT, PolyQT>> &out,
    const EvalPoly<PolyPT> &lhs,
    const EvalPoly<PolyQT> &rhs) {
    using TensorNTT = TensorNTTImpl<PolyPT, PolyQT>;
    using Z = typename TensorNTT::Z;

    for (size_t i = 0; i < EvalPoly<PolyPT>::N; ++i) {
        for (size_t j = 0; j < EvalPoly<PolyQT>::N; ++j) {
            out[i * EvalPoly<PolyQT>::N + j] = Z::Mul(lhs[i], rhs[j]);
        }
    }
}

template <typename PolyPT, typename PolyQT>
[[nodiscard]] EvalPoly<TensorNTTImpl<PolyPT, PolyQT>> Tensor(const EvalPoly<PolyPT> &lhs, const EvalPoly<PolyQT> &rhs) {
    EvalPoly<TensorNTTImpl<PolyPT, PolyQT>> out;
    Tensor(out, lhs, rhs);
    return out;
}

template <typename PolyPT, typename PolyQT>
[[nodiscard]] EvalPoly<TensorNTTImpl<PolyPT, PolyQT>> GenKey(const std::vector<int64_t> &sk) {
    auto sk_p_coeff = CoeffPoly<PolyPT>::FromSigned(std::span<const int64_t>(sk));
    EvalPoly<PolyQT> one_q;
    for (size_t i = 0; i < EvalPoly<PolyQT>::N; ++i) {
        one_q[i] = 1;
    }

    CanonicalNttPlan<PolyPT> plan_p;
    return Tensor(plan_p.forward(sk_p_coeff), one_q);
}

template <typename Params = DefaultParams>
[[nodiscard]] typename Params::SchemePQ::RLWEKey TensorKey(
    const typename Params::SchemePt::RLWEKey &sk_p,
    const typename Params::SchemeQt::RLWEKey &sk_q) {
    using SchemePQ = typename Params::SchemePQ;
    using EvalPt = typename Params::SchemePt::Eval;
    using EvalQt = typename Params::SchemeQt::Eval;
    using EvalPQ = typename Params::SchemePQ::Eval;
    using Z = typename Params::Z;

    typename SchemePQ::RLWEKey sk_pq;
    sk_pq.reserve(3);

    EvalPt skp0 = sk_p[0];
    EvalQt skq0 = sk_q[0];
    EvalPt p_one;
    EvalQt q_one;

    for (size_t i = 0; i < EvalPt::N; ++i) {
        p_one[i] = 1;
    }
    for (size_t i = 0; i < EvalQt::N; ++i) {
        q_one[i] = 1;
    }

    EvalPQ skpq0 = Tensor(skp0, skq0);
    for (size_t i = 0; i < EvalPQ::N; ++i) {
        skpq0[i] = Z::Sub(0, skpq0[i]);
    }

    sk_pq.push_back(skpq0);
    sk_pq.push_back(Tensor(skp0, q_one));
    sk_pq.push_back(Tensor(p_one, skq0));
    return sk_pq;
}

template <typename Params = DefaultParams>
[[nodiscard]] typename Params::SchemePQ::RLWECiphertext TensorCt(
    const typename Params::SchemePt::RLWECiphertext &ct_p,
    const typename Params::SchemeQt::RLWECiphertext &ct_q) {
    using SchemePQ = typename Params::SchemePQ;
    using SchemePt = typename Params::SchemePt;
    using SchemeQt = typename Params::SchemeQt;

    const auto ct2_p = SchemePt::ToRlweCt2(ct_p);
    const auto ct2_q = SchemeQt::ToRlweCt2(ct_q);
    const uint64_t scaling_factor = SchemePQ::Q - Params::kPlainModulus;
    typename SchemePQ::TensorCt4 ct4{
        Tensor(ct2_p.a, ct2_q.a) * scaling_factor,
        Tensor(ct2_p.a, ct2_q.b) * scaling_factor,
        Tensor(ct2_p.b, ct2_q.a) * scaling_factor,
        Tensor(ct2_p.b, ct2_q.b) * scaling_factor,
    };
    return SchemePQ::FromTensorCt4(ct4);
}

template <typename Params = DefaultParams>
struct TensorExpCrtState {
    using SchemePt = typename Params::SchemePt;
    using SchemeQt = typename Params::SchemeQt;
    using SchemePQ = typename Params::SchemePQ;

    SchemePQ scheme_pq;
    typename SchemePQ::RLWESwitchingKey tensor_bk;

    TensorExpCrtState(
        const std::vector<int64_t> &lwe_secret,
        const std::vector<int64_t> &sk_p,
        const std::vector<int64_t> &sk_q,
        RandomContext &rng,
        double rlwe_noise_variance) {
        SchemePt scheme_pt(sk_p);
        SchemeQt scheme_qt(sk_q);

        auto sk_pq = TensorKey<Params>(scheme_pt.sk, scheme_qt.sk);
        typename SchemePQ::RLWEKey sk_p0{GenKey<typename SchemePt::Transform, typename SchemeQt::Transform>(lwe_secret)};
        tensor_bk = scheme_pq.KeySwitchGen(sk_pq, sk_p0, rlwe_noise_variance, rng.engine);
    }
};

template <typename Params = DefaultParams>
[[nodiscard]] typename Params::SchemePQ::RLWECiphertext ApplyTensorExpCRT(
    TensorExpCrtState<Params> &state,
    const typename Params::SchemePt::RLWECiphertext &ct_p,
    const typename Params::SchemeQt::RLWECiphertext &ct_q) {
    auto ct_pq = TensorCt<Params>(ct_p, ct_q);
    return state.scheme_pq.KeySwitch(ct_pq, state.tensor_bk);
}

template <typename Params = DefaultParams>
[[nodiscard]] typename Params::SchemePQ::RLWECiphertext ExpCRT(
    TensorExpCrtState<Params> &state,
    const typename Params::SchemePt::RLWECiphertext &ct_p,
    const typename Params::SchemeQt::RLWECiphertext &ct_q,
    ExpCrtVariant variant = ExpCrtVariant::TensorTrick) {
    switch (variant) {
        case ExpCrtVariant::TensorTrick:
            return ApplyTensorExpCRT<Params>(state, ct_p, ct_q);
        case ExpCrtVariant::Paper:
            throw std::runtime_error("Paper ExpCRT is not implemented yet");
    }
    throw std::runtime_error("Unknown ExpCRT variant");
}

} // namespace bdf17

#endif // BDF17_EXPCRT_HPP
