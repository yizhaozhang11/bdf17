#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "accumulator.hpp"
#include "expcrt.hpp"
#include "fun_extract.hpp"
#include "lwe_frontend.hpp"
#include "params.hpp"

int main() {
    using Params = bdf17::DefaultParams;
    const uint64_t q_frontend = Params::kFrontendModulus;
    const uint64_t q_accumulator_input = Params::kAccumulatorInputModulus;
    const uint64_t q_extract_internal = Params::kExtractModulus;
    const size_t packing_width = bdf17::MaxPackingBits(Params::kPlainModulus);

    auto stage_start = std::chrono::system_clock::now();
    auto start_timer = [&stage_start]() { stage_start = std::chrono::system_clock::now(); };
    auto end_timer = [&stage_start]() {
        std::cout << "Time: " << (std::chrono::duration<double>(std::chrono::system_clock::now() - stage_start).count()) << std::endl;
    };

    std::mt19937_64 engine(std::random_device{}());
    std::uniform_int_distribution<int> bit_dist(0, 1);

    start_timer();

    std::vector<int64_t> lwe_secret_frontend =
        GaussianSampler<Params::kLweFrontendDimension>::GetInstance().SampleSk(Params::kLweSecretDensity);
    std::vector<int64_t> lwe_secret_accumulator;
    std::optional<bdf17::LweKeySwitchKey> frontend_to_accumulator_ksk;

    const bool use_lwe_dim_reduction =
        Params::kEnableLweDimReduction && Params::kLweFrontendDimension != Params::kLweAccumulatorDimension;
    if (use_lwe_dim_reduction) {
        lwe_secret_accumulator =
            GaussianSampler<Params::kLweAccumulatorDimension>::GetInstance().SampleSk(Params::kLweSecretDensity);
        frontend_to_accumulator_ksk = bdf17::GenerateLweKeySwitchKey(
            lwe_secret_frontend,
            lwe_secret_accumulator,
            q_frontend,
            Params::kLweKeySwitchBase,
            Params::kLweNoiseVar,
            engine);
    } else {
        if (Params::kLweFrontendDimension != Params::kLweAccumulatorDimension) {
            throw std::runtime_error("frontend/accumulator dimensions differ while dimension reduction is disabled");
        }
        lwe_secret_accumulator = lwe_secret_frontend;
    }

    auto plain_lut = bdf17::BuildParityLut<Params>();
    auto lut_samples = bdf17::BuildTensorLutSamples<Params>(plain_lut);
    auto lut_coeff = bdf17::ConstructLutPoly<Params>(lut_samples);
    Params::PlanPQ plan_pq;
    auto lut_eval = plan_pq.forward(lut_coeff);

    bdf17::AccumulatorState<Params> accumulator(lwe_secret_accumulator);
    bdf17::TensorExpCrtState<Params> expcrt(lwe_secret_frontend, accumulator.sk_p, accumulator.sk_q);

    end_timer();

    for (size_t test_index = 0; test_index < Params::kNumTrials; ++test_index) {
        uint64_t packed_plain = 0;
        std::vector<bdf17::LweCiphertext> bit_ciphertexts;
        bit_ciphertexts.reserve(packing_width);
        for (size_t i = 0; i < packing_width; ++i) {
            const uint64_t bit = (uint64_t)bit_dist(engine);
            packed_plain |= bit << i;
            bit_ciphertexts.push_back(bdf17::EncryptLwe(
                lwe_secret_frontend,
                bit,
                Params::kPlainModulus,
                q_frontend,
                Params::kLweNoiseVar,
                engine));
        }

        auto packed_frontend = bdf17::PackBitsCiphertextsLE(bit_ciphertexts, Params::kPlainModulus, q_frontend);
        bdf17::LweCiphertext packed_for_accumulator = packed_frontend;
        if (use_lwe_dim_reduction) {
            packed_for_accumulator = bdf17::ApplyLweKeySwitch(packed_for_accumulator, *frontend_to_accumulator_ksk);
        }

        auto packed_internal =
            bdf17::ModSwitchLwe(packed_for_accumulator, q_frontend, q_accumulator_input, Params::kPlainModulus);

        std::vector<int64_t> a(Params::kLweAccumulatorDimension, 0);
        for (size_t i = 0; i < Params::kLweAccumulatorDimension; ++i) {
            a[i] = (int64_t)packed_internal.a[i];
        }
        int64_t b = (int64_t)packed_internal.b;

        start_timer();
        auto [ct_p, ct_q] = bdf17::Process<Params>(accumulator, a, b);
        end_timer();

        start_timer();
        auto tensor_ct = bdf17::ExpCRT<Params>(expcrt, ct_p, ct_q, bdf17::ExpCrtVariant::TensorTrick);
        auto extracted = bdf17::FunExtract<Params>(tensor_ct, lut_eval);
        end_timer();

        bdf17::LweCiphertext extracted_internal{extracted.a, extracted.b};
        bdf17::LweCiphertext final_frontend = extracted_internal;
        if (q_extract_internal != q_frontend) {
            final_frontend =
                bdf17::ModSwitchLwe(extracted_internal, q_extract_internal, q_frontend, Params::kPlainModulus);
        }

        const uint64_t phase = bdf17::DecryptPhase(final_frontend, lwe_secret_frontend, q_frontend);
        const size_t result = (size_t)bdf17::DecodeMessage(phase, Params::kPlainModulus, q_frontend);
        const size_t expected = plain_lut[packed_plain];
        std::cout << "result: " << result << std::endl;
        std::cout << "expected: " << expected << std::endl;

        if (result != expected) {
            std::cout << "Failure on trial " << test_index << ", packed_plain=" << packed_plain << std::endl;
            return 1;
        }
    }

    return 0;
}
