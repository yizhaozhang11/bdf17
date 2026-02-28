#ifndef BDF17_FUN_EXTRACT_HPP
#define BDF17_FUN_EXTRACT_HPP

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "params.hpp"

namespace bdf17 {

template <typename Params = DefaultParams>
std::vector<size_t> BuildParityLut() {
    std::vector<size_t> lut(Params::kPlainModulus, 0);
    for (size_t i = 0; i < Params::kPlainModulus; ++i) {
        lut[i] = i & 1;
    }
    return lut;
}

template <typename Params = DefaultParams>
std::vector<size_t> BuildTensorLutSamples(const std::vector<size_t> &plain_lut) {
    if (plain_lut.size() != Params::kPlainModulus) {
        throw std::runtime_error("plain_lut size mismatch");
    }

    std::vector<size_t> samples(Params::kTensorDimension, 0);
    for (size_t i = 0; i < Params::kTensorDimension; ++i) {
        samples[i] = plain_lut[(size_t)(0.5 + (double)Params::kPlainModulus * i / Params::kTensorDimension) % Params::kPlainModulus];
    }

    for (size_t i = 1, j = Params::kTensorDimension - 1; i < j; ++i, --j) {
        std::swap(samples[i], samples[j]);
    }

    return samples;
}

template <typename Params = DefaultParams>
void ConstructLutPoly(typename Params::CoeffPQ &out, const std::vector<size_t> &samples) {
    using CoeffP = typename Params::CoeffP;
    using CoeffQ = typename Params::CoeffQ;
    using CoeffPQ = typename Params::CoeffPQ;

    for (size_t i = 0; i < CoeffPQ::N; ++i) {
        out[i] = 0;
    }

    for (size_t k = 0; k < samples.size(); ++k) {
        const size_t i = k % CoeffP::O;
        const size_t j = k % CoeffQ::O;
        out[i * CoeffQ::N + j] = samples[k];
    }
}

template <typename Params = DefaultParams>
typename Params::CoeffPQ ConstructLutPoly(const std::vector<size_t> &samples) {
    typename Params::CoeffPQ out;
    ConstructLutPoly<Params>(out, samples);
    return out;
}

template <typename Params = DefaultParams>
typename Params::SchemePt::Eval TracePQtoP(const typename Params::SchemePQ::Eval &a) {
    using EvalP = typename Params::SchemePt::Eval;
    using EvalQ = typename Params::SchemeQt::Eval;
    using Z = typename Params::Z;

    const uint64_t z = Z::Pow(EvalQ::N, Z::p - 2);
    EvalP b;
    for (size_t i = 0; i < EvalP::N; ++i) {
        for (size_t j = 0; j < EvalQ::N; ++j) {
            b[i] = Z::Add(b[i], a[i * EvalQ::N + j]);
        }
        b[i] = Z::Mul(b[i], z);
    }
    return b;
}

template <typename Params = DefaultParams>
typename Params::SchemePt::Coeff TracePQtoP(const typename Params::SchemePQ::Coeff &a) {
    using CoeffP = typename Params::SchemePt::Coeff;
    using CoeffQ = typename Params::SchemeQt::Coeff;

    CoeffP b;
    for (size_t i = 0; i < CoeffP::N; ++i) {
        b[i] = a[i * CoeffQ::N];
    }
    return b;
}

template <typename Params = DefaultParams>
uint64_t TracePtoZ(const typename Params::SchemePt::Eval &a) {
    using Z = typename Params::Z;

    uint64_t z = 0;
    for (size_t i = 0; i < Params::SchemePt::Eval::N; ++i) {
        z = Z::Add(z, a[i]);
    }
    return Z::Mul(z, Z::Pow(Params::SchemePt::Eval::N, Z::p - 2));
}

template <typename Params = DefaultParams>
uint64_t TracePtoZ(const typename Params::SchemePt::Coeff &a) {
    return a[0];
}

template <typename Params = DefaultParams>
struct ExtractedLweSample {
    std::vector<uint64_t> a;
    uint64_t b;
};

template <typename Params = DefaultParams>
typename Params::SchemePt::RLWECiphertext ApplyLutAndTrace(
    typename Params::SchemePQ::RLWECiphertext tensor_ct,
    const typename Params::SchemePQ::Eval &lut_eval) {
    tensor_ct[0] = lut_eval * tensor_ct[0];
    tensor_ct[1] = lut_eval * tensor_ct[1];

    return {TracePQtoP<Params>(tensor_ct[0]), TracePQtoP<Params>(tensor_ct[1])};
}

template <typename Params = DefaultParams>
ExtractedLweSample<Params> ExtractLwe(typename Params::SchemePt::RLWECiphertext ct_trace) {
    using SchemePt = typename Params::SchemePt;
    using EvalP = typename SchemePt::Eval;

    ExtractedLweSample<Params> out;
    out.a.resize(Params::kLweFrontendDimension);
    out.b = TracePtoZ<Params>(ct_trace[1]);

    typename SchemePt::Plan plan_p;
    auto a_coeff = plan_p.inverse(ct_trace[0]);
    out.a[0] = a_coeff[0];
    for (size_t i = 1; i < Params::kLweFrontendDimension; ++i) {
        out.a[i] = a_coeff[EvalP::N - i];
    }
    return out;
}

template <typename Params = DefaultParams>
ExtractedLweSample<Params> FunExtract(
    typename Params::SchemePQ::RLWECiphertext tensor_ct,
    const typename Params::SchemePQ::Eval &lut_eval) {
    auto ct_trace = ApplyLutAndTrace<Params>(std::move(tensor_ct), lut_eval);
    return ExtractLwe<Params>(std::move(ct_trace));
}

} // namespace bdf17

#endif // BDF17_FUN_EXTRACT_HPP
