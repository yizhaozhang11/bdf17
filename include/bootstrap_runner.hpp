#ifndef BDF17_BOOTSTRAP_RUNNER_HPP
#define BDF17_BOOTSTRAP_RUNNER_HPP

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "accumulator.hpp"
#include "experiment_config.hpp"
#include "expcrt.hpp"
#include "fun_extract.hpp"
#include "lwe_frontend.hpp"
#include "params.hpp"

namespace bdf17 {

struct StageStats {
    double ms = 0.0;
    uint64_t forward_ntt = 0;
    uint64_t inverse_ntt = 0;
    uint64_t galois = 0;
    uint64_t keyswitch = 0;
    uint64_t extmult = 0;
    uint64_t trace_calls = 0;
    uint64_t approx_bytes = 0;
};

inline void MergeStageStats(StageStats &dst, const StageStats &src) {
    dst.ms += src.ms;
    dst.forward_ntt += src.forward_ntt;
    dst.inverse_ntt += src.inverse_ntt;
    dst.galois += src.galois;
    dst.keyswitch += src.keyswitch;
    dst.extmult += src.extmult;
    dst.trace_calls += src.trace_calls;
    dst.approx_bytes += src.approx_bytes;
}

inline void MergeStageCounters(StageStats &dst, const StageStats &src) {
    dst.forward_ntt += src.forward_ntt;
    dst.inverse_ntt += src.inverse_ntt;
    dst.galois += src.galois;
    dst.keyswitch += src.keyswitch;
    dst.extmult += src.extmult;
    dst.trace_calls += src.trace_calls;
    dst.approx_bytes += src.approx_bytes;
}

struct BootstrapMetrics {
    StageStats encrypt_bits;
    StageStats pack_bits;
    StageStats lwe_keyswitch;
    StageStats modswitch_in;
    StageStats accum_p;
    StageStats accum_q;
    StageStats expcrt;
    StageStats fun_extract;
    StageStats modswitch_out;
    StageStats decode_and_check;
    StageStats total;
};

inline void MergeBootstrapMetrics(BootstrapMetrics &dst, const BootstrapMetrics &src) {
    MergeStageStats(dst.encrypt_bits, src.encrypt_bits);
    MergeStageStats(dst.pack_bits, src.pack_bits);
    MergeStageStats(dst.lwe_keyswitch, src.lwe_keyswitch);
    MergeStageStats(dst.modswitch_in, src.modswitch_in);
    MergeStageStats(dst.accum_p, src.accum_p);
    MergeStageStats(dst.accum_q, src.accum_q);
    MergeStageStats(dst.expcrt, src.expcrt);
    MergeStageStats(dst.fun_extract, src.fun_extract);
    MergeStageStats(dst.modswitch_out, src.modswitch_out);
    MergeStageStats(dst.decode_and_check, src.decode_and_check);
    MergeStageStats(dst.total, src.total);
}

struct BootstrapRequest {
    size_t trial_index = 0;
};

struct BootstrapResult {
    size_t trial_index = 0;
    std::string bits_le;
    uint64_t packed_plain = 0;
    uint64_t phase = 0;
    size_t expected = 0;
    size_t got = 0;
    bool ok = false;
};

template <typename Params = DefaultParams>
class BootstrapRunner {
public:
    explicit BootstrapRunner(ExperimentConfig config) : config_(std::move(config)), rng_(config_.seed), bit_dist_(0, 1) {
        config_.profile_name = Params::kProfileName;
        dim_reduction_enabled_ =
            config_.has_enable_lwe_dim_reduction_override ? config_.enable_lwe_dim_reduction_override : Params::kEnableLweDimReduction;
        dim_reduction_active_ = dim_reduction_enabled_ && Params::kLweFrontendDimension != Params::kLweAccumulatorDimension;

        ValidateProfileOrThrow<Params>(dim_reduction_enabled_, config_.expcrt_variant);

        lwe_secret_frontend_ = GaussianSampler<Params::kLweFrontendDimension>::SampleSk(Params::kLweSecretDensity, rng_.engine);
        if (dim_reduction_active_) {
            lwe_secret_accumulator_ =
                GaussianSampler<Params::kLweAccumulatorDimension>::SampleSk(Params::kLweSecretDensity, rng_.engine);
            frontend_to_accumulator_ksk_ = GenerateLweKeySwitchKey(
                lwe_secret_frontend_,
                lwe_secret_accumulator_,
                Params::kFrontendModulus,
                Params::kLweKeySwitchBase,
                Params::kLweNoiseVar,
                rng_.engine);
        } else {
            lwe_secret_accumulator_ = lwe_secret_frontend_;
        }

        plain_lut_ = BuildParityLut<Params>();
        const auto lut_samples = BuildTensorLutSamples<Params>(plain_lut_);
        const auto lut_coeff = ConstructLutPoly<Params>(lut_samples);
        typename Params::PlanPQ plan_pq;
        lut_eval_ = plan_pq.forward(lut_coeff);

        accumulator_ = std::make_unique<AccumulatorState<Params>>(lwe_secret_accumulator_, rng_, Params::kRlweNoiseVar);
        expcrt_ = std::make_unique<TensorExpCrtState<Params>>(
            lwe_secret_frontend_,
            accumulator_->sk_p,
            accumulator_->sk_q,
            rng_,
            Params::kRlweNoiseVar);

        approx_lwe_ksk_bytes_ = EstimateLweKeySwitchBytes(frontend_to_accumulator_ksk_);
        approx_accum_p_key_bytes_ = EstimateBootstrappingKeyBytes<typename Params::SchemeP>(accumulator_->bk_p);
        approx_accum_q_key_bytes_ = EstimateBootstrappingKeyBytes<typename Params::SchemeQ>(accumulator_->bk_q);
        approx_expcrt_key_bytes_ = EstimateSwitchingKeyBytes<typename Params::SchemePQ>(expcrt_->tensor_bk);
    }

    const ExperimentConfig &config() const {
        return config_;
    }

    bool dim_reduction_enabled() const {
        return dim_reduction_enabled_;
    }

    bool dim_reduction_active() const {
        return dim_reduction_active_;
    }

    uint64_t q_frontend() const {
        return Params::kFrontendModulus;
    }

    uint64_t q_accumulator_input() const {
        return Params::kAccumulatorInputModulus;
    }

    uint64_t q_extract_internal() const {
        return Params::kExtractModulus;
    }

    size_t packing_width() const {
        return MaxPackingBits(Params::kPlainModulus);
    }

    std::vector<LweCiphertext> EncryptInputBits(uint64_t &packed_plain, std::string &bits_le, StageStats &stats) {
        const auto start = Clock::now();
        packed_plain = 0;
        const size_t width = packing_width();
        std::vector<LweCiphertext> bit_ciphertexts;
        bit_ciphertexts.reserve(width);
        for (size_t i = 0; i < width; ++i) {
            const uint64_t bit = static_cast<uint64_t>(bit_dist_(rng_.engine));
            packed_plain |= bit << i;
            bit_ciphertexts.push_back(EncryptLwe(
                lwe_secret_frontend_,
                bit,
                Params::kPlainModulus,
                Params::kFrontendModulus,
                Params::kLweNoiseVar,
                rng_.engine));
        }
        bits_le = BitsLE(packed_plain, width);
        stats.ms = ElapsedMs(start);
        return bit_ciphertexts;
    }

    LweCiphertext PackBits(const std::vector<LweCiphertext> &bit_ciphertexts, StageStats &stats) const {
        const auto start = Clock::now();
        auto packed = PackBitsCiphertextsLE(bit_ciphertexts, Params::kPlainModulus, Params::kFrontendModulus);
        stats.ms = ElapsedMs(start);
        return packed;
    }

    LweCiphertext FrontendKeySwitchIfNeeded(const LweCiphertext &packed_frontend, StageStats &stats) const {
        const auto start = Clock::now();
        LweCiphertext packed_for_accumulator = packed_frontend;
        if (dim_reduction_active_) {
            packed_for_accumulator = ApplyLweKeySwitch(packed_for_accumulator, *frontend_to_accumulator_ksk_);
            stats.keyswitch = 1;
            stats.approx_bytes = approx_lwe_ksk_bytes_;
        }
        stats.ms = ElapsedMs(start);
        return packed_for_accumulator;
    }

    LweCiphertext FrontendModSwitchIn(const LweCiphertext &packed_for_accumulator, StageStats &stats) const {
        const auto start = Clock::now();
        auto packed_internal =
            ModSwitchLwe(packed_for_accumulator, Params::kFrontendModulus, Params::kAccumulatorInputModulus, Params::kPlainModulus);
        stats.ms = ElapsedMs(start);
        return packed_internal;
    }

    typename Params::SchemePt::RLWECiphertext RunAccumulatorP(const std::vector<int64_t> &a, int64_t b, StageStats &stats) {
        const auto start = Clock::now();
        const auto ct_p = Params::SchemeP::template ModSwitch<typename Params::SchemePt>(
            accumulator_->scheme_p.Process(accumulator_->bk_p, a, b, Params::kPlainModulus));
        FillAccumulatorApproxCounters(a, stats);
        stats.approx_bytes = approx_accum_p_key_bytes_;
        stats.ms = ElapsedMs(start);
        return ct_p;
    }

    typename Params::SchemeQt::RLWECiphertext RunAccumulatorQ(const std::vector<int64_t> &a, int64_t b, StageStats &stats) {
        const auto start = Clock::now();
        const auto ct_q = Params::SchemeQ::template ModSwitch<typename Params::SchemeQt>(
            accumulator_->scheme_q.Process(accumulator_->bk_q, a, b, Params::kPlainModulus));
        FillAccumulatorApproxCounters(a, stats);
        stats.approx_bytes = approx_accum_q_key_bytes_;
        stats.ms = ElapsedMs(start);
        return ct_q;
    }

    typename Params::SchemePQ::RLWECiphertext RunExpCRT(
        const typename Params::SchemePt::RLWECiphertext &ct_p,
        const typename Params::SchemeQt::RLWECiphertext &ct_q,
        StageStats &stats) {
        const auto start = Clock::now();
        auto tensor_ct = ExpCRT<Params>(*expcrt_, ct_p, ct_q, config_.expcrt_variant);
        stats.keyswitch = 1;
        stats.extmult = 1;
        stats.approx_bytes = approx_expcrt_key_bytes_;
        stats.ms = ElapsedMs(start);
        return tensor_ct;
    }

    ExtractedLweSample<Params> RunFunExtract(typename Params::SchemePQ::RLWECiphertext tensor_ct, StageStats &stats) const {
        const auto start = Clock::now();
        auto extracted = FunExtract<Params>(std::move(tensor_ct), lut_eval_);
        stats.inverse_ntt = 1;
        stats.trace_calls = 1;
        stats.ms = ElapsedMs(start);
        return extracted;
    }

    LweCiphertext FrontendModSwitchOut(const LweCiphertext &extracted_internal, StageStats &stats) const {
        const auto start = Clock::now();
        LweCiphertext final_frontend = extracted_internal;
        if (Params::kExtractModulus != Params::kFrontendModulus) {
            final_frontend = ModSwitchLwe(extracted_internal, Params::kExtractModulus, Params::kFrontendModulus, Params::kPlainModulus);
        }
        stats.ms = ElapsedMs(start);
        return final_frontend;
    }

    BootstrapResult DecodeAndCheck(size_t trial_index, uint64_t packed_plain, const LweCiphertext &final_frontend, StageStats &stats) const {
        const auto start = Clock::now();
        BootstrapResult out;
        out.trial_index = trial_index;
        out.packed_plain = packed_plain;
        out.bits_le = BitsLE(packed_plain, packing_width());
        out.phase = DecryptPhase(final_frontend, lwe_secret_frontend_, Params::kFrontendModulus);
        out.got = static_cast<size_t>(DecodeMessage(out.phase, Params::kPlainModulus, Params::kFrontendModulus));
        out.expected = plain_lut_[packed_plain];
        out.ok = out.got == out.expected;
        stats.ms = ElapsedMs(start);
        return out;
    }

    BootstrapResult RunTrial(const BootstrapRequest &request, BootstrapMetrics *metrics = nullptr) {
        BootstrapMetrics local_metrics;
        const auto total_start = Clock::now();

        uint64_t packed_plain = 0;
        std::string bits_le;
        auto bit_ciphertexts = EncryptInputBits(packed_plain, bits_le, local_metrics.encrypt_bits);
        auto packed_frontend = PackBits(bit_ciphertexts, local_metrics.pack_bits);
        auto packed_for_accumulator = FrontendKeySwitchIfNeeded(packed_frontend, local_metrics.lwe_keyswitch);
        auto packed_internal = FrontendModSwitchIn(packed_for_accumulator, local_metrics.modswitch_in);

        std::vector<int64_t> a(Params::kLweAccumulatorDimension, 0);
        for (size_t i = 0; i < Params::kLweAccumulatorDimension; ++i) {
            a[i] = static_cast<int64_t>(packed_internal.a[i]);
        }
        const int64_t b = static_cast<int64_t>(packed_internal.b);

        auto ct_p = RunAccumulatorP(a, b, local_metrics.accum_p);
        auto ct_q = RunAccumulatorQ(a, b, local_metrics.accum_q);
        auto tensor_ct = RunExpCRT(ct_p, ct_q, local_metrics.expcrt);
        auto extracted = RunFunExtract(std::move(tensor_ct), local_metrics.fun_extract);
        LweCiphertext extracted_internal{extracted.a, extracted.b};
        auto final_frontend = FrontendModSwitchOut(extracted_internal, local_metrics.modswitch_out);
        auto result = DecodeAndCheck(request.trial_index, packed_plain, final_frontend, local_metrics.decode_and_check);
        result.bits_le = bits_le;

        local_metrics.total.ms = ElapsedMs(total_start);
        MergeStageCounters(local_metrics.total, local_metrics.encrypt_bits);
        MergeStageCounters(local_metrics.total, local_metrics.pack_bits);
        MergeStageCounters(local_metrics.total, local_metrics.lwe_keyswitch);
        MergeStageCounters(local_metrics.total, local_metrics.modswitch_in);
        MergeStageCounters(local_metrics.total, local_metrics.accum_p);
        MergeStageCounters(local_metrics.total, local_metrics.accum_q);
        MergeStageCounters(local_metrics.total, local_metrics.expcrt);
        MergeStageCounters(local_metrics.total, local_metrics.fun_extract);
        MergeStageCounters(local_metrics.total, local_metrics.modswitch_out);
        MergeStageCounters(local_metrics.total, local_metrics.decode_and_check);

        if (metrics != nullptr) {
            *metrics = local_metrics;
        }
        return result;
    }

private:
    using Clock = std::chrono::steady_clock;

    static double ElapsedMs(const Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    static std::string BitsLE(const uint64_t value, const size_t width) {
        std::string bits;
        bits.reserve(width);
        for (size_t i = 0; i < width; ++i) {
            bits.push_back(((value >> i) & 1ULL) != 0 ? '1' : '0');
        }
        return bits;
    }

    template <typename SchemeT>
    static uint64_t EstimateCiphertextBytes(const typename SchemeT::RLWECiphertext &ct) {
        return static_cast<uint64_t>(ct.size()) * SchemeT::Eval::N * sizeof(uint64_t);
    }

    template <typename SchemeT>
    static uint64_t EstimateGadgetCiphertextBytes(const typename SchemeT::RLWEGadgetCiphertext &ct) {
        uint64_t bytes = 0;
        for (const auto &digit_ct : ct) {
            bytes += EstimateCiphertextBytes<SchemeT>(digit_ct);
        }
        return bytes;
    }

    template <typename SchemeT>
    static uint64_t EstimateRgswBytes(const typename SchemeT::RGSWCiphertext &ct) {
        return EstimateGadgetCiphertextBytes<SchemeT>(ct.first) + EstimateGadgetCiphertextBytes<SchemeT>(ct.second);
    }

    template <typename SchemeT>
    static uint64_t EstimateBootstrappingKeyBytes(const std::vector<typename SchemeT::RGSWCiphertext> &bk) {
        uint64_t bytes = 0;
        for (const auto &entry : bk) {
            bytes += EstimateRgswBytes<SchemeT>(entry);
        }
        return bytes;
    }

    template <typename SchemeT>
    static uint64_t EstimateSwitchingKeyBytes(const typename SchemeT::RLWESwitchingKey &ksk) {
        uint64_t bytes = 0;
        for (const auto &input_dim_entry : ksk) {
            bytes += EstimateGadgetCiphertextBytes<SchemeT>(input_dim_entry);
        }
        return bytes;
    }

    static uint64_t EstimateLweKeySwitchBytes(const std::optional<LweKeySwitchKey> &ksk) {
        if (!ksk.has_value()) {
            return 0;
        }
        if (ksk->data.empty()) {
            return 0;
        }
        const uint64_t per_ct_words = static_cast<uint64_t>(ksk->n_out) + 1;
        return static_cast<uint64_t>(ksk->data.size()) * per_ct_words * sizeof(uint64_t);
    }

    static void FillAccumulatorApproxCounters(const std::vector<int64_t> &a, StageStats &stats) {
        const uint64_t nonzero = static_cast<uint64_t>(std::count_if(a.begin(), a.end(), [](int64_t x) { return x != 0; }));
        stats.forward_ntt += nonzero * 2;
        stats.inverse_ntt += nonzero;
        stats.extmult += nonzero;
        stats.galois += nonzero;
        stats.keyswitch += nonzero;
    }

    ExperimentConfig config_;
    bool dim_reduction_enabled_ = false;
    bool dim_reduction_active_ = false;
    RandomContext rng_;
    std::uniform_int_distribution<int> bit_dist_;

    std::vector<int64_t> lwe_secret_frontend_;
    std::vector<int64_t> lwe_secret_accumulator_;
    std::optional<LweKeySwitchKey> frontend_to_accumulator_ksk_;

    std::vector<size_t> plain_lut_;
    typename Params::SchemePQ::Eval lut_eval_;

    std::unique_ptr<AccumulatorState<Params>> accumulator_;
    std::unique_ptr<TensorExpCrtState<Params>> expcrt_;

    uint64_t approx_lwe_ksk_bytes_ = 0;
    uint64_t approx_accum_p_key_bytes_ = 0;
    uint64_t approx_accum_q_key_bytes_ = 0;
    uint64_t approx_expcrt_key_bytes_ = 0;
};

} // namespace bdf17

#endif // BDF17_BOOTSTRAP_RUNNER_HPP
