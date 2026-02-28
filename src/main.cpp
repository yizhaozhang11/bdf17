#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

#include "accumulator.hpp"
#include "expcrt.hpp"
#include "fun_extract.hpp"
#include "lwe_frontend.hpp"
#include "params.hpp"

int main() {
    using Params = bdf17::DefaultParams;
    constexpr size_t kMaxAttemptsPerTrial = 8;
    const uint64_t q_in = Params::kTensorDimension;
    const uint64_t q_out = Params::Z::p;
    const size_t packing_width = bdf17::MaxPackingBits(Params::kPlainModulus);

    auto stage_start = std::chrono::system_clock::now();
    auto start_timer = [&stage_start]() { stage_start = std::chrono::system_clock::now(); };
    auto end_timer = [&stage_start]() {
        std::cout << "Time: " << (std::chrono::duration<double>(std::chrono::system_clock::now() - stage_start).count()) << std::endl;
    };

    std::mt19937_64 engine(std::random_device{}());
    std::uniform_int_distribution<int> bit_dist(0, 1);

    start_timer();

    std::vector<int64_t> lwe_secret_in = GaussianSampler<Params::kLweInputDimension>::GetInstance().SampleSk(Params::kLweSecretDensity);
    std::vector<int64_t> lwe_secret_out;
    std::optional<bdf17::LweKeySwitchKey> lwe_reduction_key;

    const bool use_lwe_dim_reduction = Params::kEnableLweDimReduction && Params::kLweInputDimension != Params::kLweOutputDimension;
    if (use_lwe_dim_reduction) {
        lwe_secret_out = GaussianSampler<Params::kLweOutputDimension>::GetInstance().SampleSk(Params::kLweSecretDensity);
        lwe_reduction_key = bdf17::GenerateLweKeySwitchKey(
            lwe_secret_in, lwe_secret_out, q_out, Params::kLweKeySwitchBase, Params::kLweNoiseVar, engine);
    }

    auto plain_lut = bdf17::BuildParityLut<Params>();
    auto lut_samples = bdf17::BuildTensorLutSamples<Params>(plain_lut);
    auto lut_poly = bdf17::ConstructLutPoly<Params>(lut_samples);
    lut_poly.ToNTT();

    bdf17::AccumulatorState<Params> accumulator(lwe_secret_in);
    bdf17::TensorExpCrtState<Params> expcrt(lwe_secret_in, accumulator.sk_p, accumulator.sk_q);

    end_timer();

    for (size_t test_index = 0; test_index < Params::kNumTrials; ++test_index) {
        bool trial_passed = false;
        for (size_t attempt = 0; attempt < kMaxAttemptsPerTrial; ++attempt) {
            uint64_t packed_plain = 0;
            std::vector<bdf17::LweCiphertext> bit_ciphertexts;
            bit_ciphertexts.reserve(packing_width);
            for (size_t i = 0; i < packing_width; ++i) {
                const uint64_t bit = (uint64_t)bit_dist(engine);
                packed_plain |= bit << i;
                bit_ciphertexts.push_back(bdf17::EncryptLwe(
                    lwe_secret_in, bit, Params::kPlainModulus, q_in, Params::kLweNoiseVar, engine));
            }

            auto packed_ct = bdf17::PackBitsCiphertextsLE(bit_ciphertexts, Params::kPlainModulus, q_in);

            std::vector<int64_t> a(Params::kLweInputDimension, 0);
            for (size_t i = 0; i < Params::kLweInputDimension; ++i) {
                a[i] = (int64_t)packed_ct.a[i];
            }
            int64_t b = (int64_t)packed_ct.b;

            start_timer();
            auto [ct_p, ct_q] = bdf17::Process<Params>(accumulator, a, b);
            end_timer();

            start_timer();
            auto tensor_ct = bdf17::ExpCRT<Params>(expcrt, ct_p, ct_q, bdf17::ExpCrtVariant::TensorTrick);
            auto extracted = bdf17::FunExtract<Params>(tensor_ct, lut_poly);
            end_timer();

            bdf17::LweCiphertext final_ct{extracted.a, extracted.b};
            const std::vector<int64_t> *decode_sk = &lwe_secret_in;
            if (use_lwe_dim_reduction) {
                final_ct = bdf17::ApplyLweKeySwitch(final_ct, *lwe_reduction_key);
                decode_sk = &lwe_secret_out;
            }

            const uint64_t phase = bdf17::DecryptPhase(final_ct, *decode_sk, q_out);
            const size_t result = (size_t)bdf17::DecodeMessage(phase, Params::kPlainModulus, q_out);
            const size_t expected = plain_lut[packed_plain];
            std::cout << "result: " << result << std::endl;
            std::cout << "expected: " << expected << std::endl;

            if (result == expected) {
                trial_passed = true;
                break;
            }
        }

        if (!trial_passed) {
            std::cout << "Mismatch after " << kMaxAttemptsPerTrial << " attempts on trial " << test_index << std::endl;
            return 1;
        }
    }

    return 0;
}
