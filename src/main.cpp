#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "accumulator.hpp"
#include "expcrt.hpp"
#include "fun_extract.hpp"
#include "params.hpp"

namespace {

template <typename Params>
size_t DecodeMessage(uint64_t phase) {
    using Z = typename Params::Z;
    return (size_t)(0.5 + (double)Params::kPlainModulus * phase / Z::p) % Params::kPlainModulus;
}

} // namespace

int main() {
    using Params = bdf17::DefaultParams;

    auto stage_start = std::chrono::system_clock::now();
    auto start_timer = [&stage_start]() { stage_start = std::chrono::system_clock::now(); };
    auto end_timer = [&stage_start]() {
        std::cout << "Time: " << (std::chrono::duration<double>(std::chrono::system_clock::now() - stage_start).count()) << std::endl;
    };

    std::mt19937 engine(std::random_device{}());
    std::uniform_int_distribution<size_t> sample_index_dist(0, Params::kTensorDimension - 1);
    std::uniform_int_distribution<size_t> plain_dist(0, Params::kPlainModulus - 1);

    start_timer();

    std::vector<int64_t> lwe_secret = GaussianSampler<Params::kLweDimension>::GetInstance().SampleSk(Params::kLweSecretDensity);

    auto plain_lut = bdf17::BuildParityLut<Params>();
    auto lut_samples = bdf17::BuildTensorLutSamples<Params>(plain_lut);
    auto lut_poly = bdf17::ConstructLutPoly<Params>(lut_samples);
    lut_poly.ToNTT();

    bdf17::AccumulatorState<Params> accumulator(lwe_secret);
    bdf17::TensorExpCrtState<Params> expcrt(lwe_secret, accumulator.sk_p, accumulator.sk_q);

    end_timer();

    for (size_t test_index = 0; test_index < Params::kNumTrials; ++test_index) {
        std::vector<int64_t> a(Params::kLweDimension);
        for (size_t i = 0; i < Params::kLweDimension; ++i) {
            a[i] = (int64_t)sample_index_dist(engine);
        }

        const size_t b0 = plain_dist(engine);
        int64_t b = (int64_t)(b0 * Params::kTensorDimension / Params::kPlainModulus);
        for (size_t i = 0; i < Params::kLweDimension; ++i) {
            b = (b + lwe_secret[i] * a[i]) % (int64_t)Params::kTensorDimension;
        }
        b = (b + (int64_t)Params::kTensorDimension) % (int64_t)Params::kTensorDimension;

        start_timer();
        auto [ct_p, ct_q] = bdf17::Process<Params>(accumulator, a, b);
        end_timer();

        start_timer();
        auto tensor_ct = bdf17::ExpCRT<Params>(expcrt, ct_p, ct_q, bdf17::ExpCrtVariant::TensorTrick);
        auto extracted = bdf17::FunExtract<Params>(tensor_ct, lut_poly);
        end_timer();

        auto phase = extracted.b;
        for (size_t i = 0; i < Params::kLweDimension; ++i) {
            phase = Params::Z::Sub(phase, Params::Z::Mul(lwe_secret[i] + Params::Z::p, extracted.a[i]));
        }

        const size_t result = DecodeMessage<Params>(phase);
        const size_t expected = plain_lut[b0];
        std::cout << "result: " << result << std::endl;
        std::cout << "expected: " << expected << std::endl;

        if (result != expected) {
            std::cout << "Error" << std::endl;
            return 1;
        }
    }

    return 0;
}
