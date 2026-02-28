#ifndef BDF17_ACCUMULATOR_HPP
#define BDF17_ACCUMULATOR_HPP

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "params.hpp"

namespace bdf17 {

template <typename Params = DefaultParams>
struct AccumulatorState {
    using PolyP = typename Params::PolyP;
    using PolyQ = typename Params::PolyQ;
    using SchemeP = typename Params::SchemeP;
    using SchemeQ = typename Params::SchemeQ;

    using BootstrappingKeyP = std::vector<typename SchemeP::RGSWCiphertext>;
    using BootstrappingKeyQ = std::vector<typename SchemeQ::RGSWCiphertext>;

    std::vector<int64_t> sk_p;
    std::vector<int64_t> sk_q;
    SchemeP scheme_p;
    SchemeQ scheme_q;
    BootstrappingKeyP bk_p;
    BootstrappingKeyQ bk_q;

    explicit AccumulatorState(const std::vector<int64_t> &lwe_secret)
        : sk_p(GaussianSampler<PolyP::N>::GetInstance().SampleSk(Params::kAccumulatorSecretDensity)),
          sk_q(GaussianSampler<PolyQ::N>::GetInstance().SampleSk(Params::kAccumulatorSecretDensity)),
          scheme_p(sk_p),
          scheme_q(sk_q) {
        if (lwe_secret.size() != Params::kLweInputDimension) {
            throw std::runtime_error("lwe_secret size mismatch");
        }
        scheme_p.GaloisKeyGen();
        scheme_q.GaloisKeyGen();
        bk_p = scheme_p.BootstrappingKeyGen(lwe_secret);
        bk_q = scheme_q.BootstrappingKeyGen(lwe_secret);
    }
};

template <typename Params = DefaultParams>
using AccumulatorOutput = std::pair<typename Params::SchemePt::RLWECiphertext, typename Params::SchemeQt::RLWECiphertext>;

template <typename Params = DefaultParams>
AccumulatorOutput<Params> ExtExpInner(AccumulatorState<Params> &state, const std::vector<int64_t> &a, int64_t b) {
    using SchemeP = typename Params::SchemeP;
    using SchemeQ = typename Params::SchemeQ;
    using SchemePt = typename Params::SchemePt;
    using SchemeQt = typename Params::SchemeQt;

    if (a.size() != Params::kLweInputDimension) {
        throw std::runtime_error("LWE input dimension mismatch");
    }

    auto ct_p = SchemeP::template ModSwitch<SchemePt>(state.scheme_p.Process(state.bk_p, a, b, Params::kPlainModulus));
    auto ct_q = SchemeQ::template ModSwitch<SchemeQt>(state.scheme_q.Process(state.bk_q, a, b, Params::kPlainModulus));

    return {std::move(ct_p), std::move(ct_q)};
}

template <typename Params = DefaultParams>
AccumulatorOutput<Params> Process(AccumulatorState<Params> &state, const std::vector<int64_t> &a, int64_t b) {
    return ExtExpInner<Params>(state, a, b);
}

} // namespace bdf17

#endif // BDF17_ACCUMULATOR_HPP
