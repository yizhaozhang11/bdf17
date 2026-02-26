#ifndef BDF17_EXPCRT_HPP
#define BDF17_EXPCRT_HPP

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "params.hpp"

namespace bdf17 {

enum class ExpCrtVariant {
    TensorTrick,
    Paper,
};

template <typename PolyPT, typename PolyQT>
void Tensor(Poly<TensorNTTImpl<typename PolyPT::NTT, typename PolyQT::NTT>> &out, const PolyPT &lhs, const PolyQT &rhs) {
    using TensorNTT = TensorNTTImpl<typename PolyPT::NTT, typename PolyQT::NTT>;
    using Z = typename TensorNTT::Z;

    if (lhs.is_coeff || rhs.is_coeff) {
        throw std::runtime_error("Tensor product is not supported for coefficient domain");
    }

    out.is_coeff = false;
    for (size_t i = 0; i < PolyPT::N; ++i) {
        for (size_t j = 0; j < PolyQT::N; ++j) {
            out.a[i * PolyQT::N + j] = Z::Mul(lhs.a[i], rhs.a[j]);
        }
    }
}

template <typename PolyPT, typename PolyQT>
Poly<TensorNTTImpl<typename PolyPT::NTT, typename PolyQT::NTT>> Tensor(const PolyPT &lhs, const PolyQT &rhs) {
    Poly<TensorNTTImpl<typename PolyPT::NTT, typename PolyQT::NTT>> out(false);
    Tensor(out, lhs, rhs);
    return out;
}

template <typename PolyPT, typename PolyQT>
Poly<TensorNTTImpl<typename PolyPT::NTT, typename PolyQT::NTT>> GenKey(const std::vector<int64_t> &sk) {
    PolyPT sk_p(true);
    PolyQT one_q(false);

    for (size_t i = 0; i < sk.size(); ++i) {
        sk_p.a[i] = sk[i] < 0 ? sk[i] + PolyPT::p : sk[i];
    }
    for (size_t i = 0; i < PolyQT::N; ++i) {
        one_q.a[i] = 1;
    }

    sk_p.ToNTT();
    return Tensor(sk_p, one_q);
}

template <typename Params = DefaultParams>
typename Params::SchemePQ::RLWEKey TensorKey(const typename Params::SchemePt::RLWEKey &sk_p, const typename Params::SchemeQt::RLWEKey &sk_q) {
    using PolyPt = typename Params::PolyPt;
    using PolyQt = typename Params::PolyQt;
    using PolyPQ = typename Params::PolyPQ;
    using SchemePQ = typename Params::SchemePQ;
    using Z = typename Params::Z;

    typename SchemePQ::RLWEKey sk_pq;

    PolyPt skp0 = sk_p[0];
    PolyQt skq0 = sk_q[0];
    PolyPt p_one(false);
    PolyQt q_one(false);

    for (size_t i = 0; i < PolyPt::N; ++i) {
        p_one.a[i] = 1;
    }
    for (size_t i = 0; i < PolyQt::N; ++i) {
        q_one.a[i] = 1;
    }

    PolyPQ skpq0 = Tensor(skp0, skq0);
    for (size_t i = 0; i < PolyPQ::N; ++i) {
        skpq0.a[i] = Z::Sub(0, skpq0.a[i]);
    }

    sk_pq.push_back(skpq0);
    sk_pq.push_back(Tensor(skp0, q_one));
    sk_pq.push_back(Tensor(p_one, skq0));
    return sk_pq;
}

template <typename Params = DefaultParams>
typename Params::SchemePQ::RLWECiphertext TensorCt(const typename Params::SchemePt::RLWECiphertext &ct_p, const typename Params::SchemeQt::RLWECiphertext &ct_q) {
    using SchemePQ = typename Params::SchemePQ;

    typename SchemePQ::RLWECiphertext ct;
    const uint64_t scaling_factor = SchemePQ::Q - Params::kPlainModulus;
    ct.push_back(Tensor(ct_p[0], ct_q[0]) * scaling_factor);
    ct.push_back(Tensor(ct_p[0], ct_q[1]) * scaling_factor);
    ct.push_back(Tensor(ct_p[1], ct_q[0]) * scaling_factor);
    ct.push_back(Tensor(ct_p[1], ct_q[1]) * scaling_factor);
    return ct;
}

template <typename Params = DefaultParams>
struct TensorExpCrtState {
    using SchemePt = typename Params::SchemePt;
    using SchemeQt = typename Params::SchemeQt;
    using SchemePQ = typename Params::SchemePQ;

    SchemePQ scheme_pq;
    typename SchemePQ::RLWESwitchingKey tensor_bk;

    TensorExpCrtState(const std::vector<int64_t> &lwe_secret, const std::vector<int64_t> &sk_p, const std::vector<int64_t> &sk_q) {
        SchemePt scheme_pt(sk_p);
        SchemeQt scheme_qt(sk_q);

        auto sk_pq = TensorKey<Params>(scheme_pt.sk, scheme_qt.sk);
        typename SchemePQ::RLWEKey sk_p0{GenKey<typename Params::PolyPt, typename Params::PolyQt>(lwe_secret)};
        tensor_bk = scheme_pq.KeySwitchGen(sk_pq, sk_p0);
    }
};

template <typename Params = DefaultParams>
typename Params::SchemePQ::RLWECiphertext ApplyTensorExpCRT(
    TensorExpCrtState<Params> &state,
    const typename Params::SchemePt::RLWECiphertext &ct_p,
    const typename Params::SchemeQt::RLWECiphertext &ct_q) {
    auto ct_pq = TensorCt<Params>(ct_p, ct_q);
    return state.scheme_pq.KeySwitch(ct_pq, state.tensor_bk);
}

template <typename Params = DefaultParams>
typename Params::SchemePQ::RLWECiphertext ExpCRT(
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
