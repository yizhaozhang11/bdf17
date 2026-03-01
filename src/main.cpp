#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "accumulator.hpp"
#include "experiment_config.hpp"
#include "expcrt.hpp"
#include "fun_extract.hpp"
#include "lwe_frontend.hpp"
#include "params.hpp"

namespace {

constexpr uint64_t kDefaultSeed = 0xBDF170001ULL;
constexpr size_t kDefaultTrials = 8;
constexpr const char *kDefaultProfileName = "default";
constexpr const char *kDefaultLutName = "lowbit";

using bdf17::ExperimentConfig;

struct TrialResult {
    size_t trial_index = 0;
    std::string bits_le;
    uint64_t packed_plain = 0;
    uint64_t phase = 0;
    size_t expected = 0;
    size_t got = 0;
    bool ok = false;
};

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

const char *VariantName(const bdf17::ExpCrtVariant variant) {
    switch (variant) {
        case bdf17::ExpCrtVariant::TensorTrick:
            return "tensortrick";
        case bdf17::ExpCrtVariant::Paper:
            return "paper";
    }
    return "unknown";
}

void PrintUsage(const char *argv0) {
    std::cout << "Usage: " << argv0
              << " [--seed <uint64>] [--trials <size_t>] [--profile <name>] [--expcrt-variant <tensortrick|paper>]"
              << " [--lut <lowbit|parity>] [--json] [--enable-lwe-dim-reduction <0|1>]" << std::endl;
}

uint64_t ParseUint64(const std::string &text, const std::string &flag) {
    try {
        size_t consumed = 0;
        const auto parsed = static_cast<uint64_t>(std::stoull(text, &consumed, 10));
        if (consumed != text.size()) {
            throw std::runtime_error("invalid " + flag + " value: " + text);
        }
        return parsed;
    } catch (const std::exception &) {
        throw std::runtime_error("invalid " + flag + " value: " + text);
    }
}

bool ParseBool01(const std::string &text, const std::string &flag) {
    if (text == "0") {
        return false;
    }
    if (text == "1") {
        return true;
    }
    throw std::runtime_error("invalid " + flag + " value: " + text + " (expected 0 or 1)");
}

bdf17::ExpCrtVariant ParseVariant(std::string text) {
    text = ToLower(std::move(text));
    if (text == "tensortrick" || text == "tensor_trick") {
        return bdf17::ExpCrtVariant::TensorTrick;
    }
    if (text == "paper") {
        return bdf17::ExpCrtVariant::Paper;
    }
    throw std::runtime_error("invalid --expcrt-variant value: " + text);
}

ExperimentConfig DefaultExperimentConfig() {
    ExperimentConfig config{};
    config.seed = kDefaultSeed;
    config.num_trials = kDefaultTrials;
    config.expcrt_variant = bdf17::ExpCrtVariant::TensorTrick;
    config.profile_name = kDefaultProfileName;
    config.lut_name = kDefaultLutName;
    config.enable_lwe_dim_reduction_override = false;
    config.has_enable_lwe_dim_reduction_override = false;
    config.emit_json = false;
    return config;
}

ExperimentConfig ParseExperimentConfig(const int argc, char **argv) {
    ExperimentConfig config = DefaultExperimentConfig();
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        if (arg == "--json") {
            config.emit_json = true;
            continue;
        }
        if (arg == "--seed") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --seed");
            }
            config.seed = ParseUint64(argv[++i], "--seed");
            continue;
        }
        if (arg.rfind("--seed=", 0) == 0) {
            config.seed = ParseUint64(arg.substr(7), "--seed");
            continue;
        }
        if (arg == "--trials") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --trials");
            }
            config.num_trials = static_cast<size_t>(ParseUint64(argv[++i], "--trials"));
            continue;
        }
        if (arg.rfind("--trials=", 0) == 0) {
            config.num_trials = static_cast<size_t>(ParseUint64(arg.substr(9), "--trials"));
            continue;
        }
        if (arg == "--profile") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --profile");
            }
            config.profile_name = ToLower(argv[++i]);
            continue;
        }
        if (arg.rfind("--profile=", 0) == 0) {
            config.profile_name = ToLower(arg.substr(10));
            continue;
        }
        if (arg == "--expcrt-variant") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --expcrt-variant");
            }
            config.expcrt_variant = ParseVariant(argv[++i]);
            continue;
        }
        if (arg.rfind("--expcrt-variant=", 0) == 0) {
            config.expcrt_variant = ParseVariant(arg.substr(17));
            continue;
        }
        if (arg == "--lut") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --lut");
            }
            config.lut_name = ToLower(argv[++i]);
            continue;
        }
        if (arg.rfind("--lut=", 0) == 0) {
            config.lut_name = ToLower(arg.substr(6));
            continue;
        }
        if (arg == "--enable-lwe-dim-reduction") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --enable-lwe-dim-reduction");
            }
            config.enable_lwe_dim_reduction_override = ParseBool01(argv[++i], "--enable-lwe-dim-reduction");
            config.has_enable_lwe_dim_reduction_override = true;
            continue;
        }
        if (arg.rfind("--enable-lwe-dim-reduction=", 0) == 0) {
            config.enable_lwe_dim_reduction_override = ParseBool01(arg.substr(27), "--enable-lwe-dim-reduction");
            config.has_enable_lwe_dim_reduction_override = true;
            continue;
        }
        throw std::runtime_error("unknown argument: " + arg);
    }
    return config;
}

template <typename Params>
void ValidateExperimentConfig(const ExperimentConfig &config) {
    if (config.num_trials == 0) {
        throw std::runtime_error("--trials must be greater than zero");
    }
    if (config.profile_name != "default" && config.profile_name != "smooth-ntt-1153x1297") {
        throw std::runtime_error("unsupported --profile value: " + config.profile_name);
    }
    if (config.expcrt_variant == bdf17::ExpCrtVariant::Paper) {
        throw std::runtime_error("unsupported --expcrt-variant value: paper (not implemented in this build)");
    }
    if (config.lut_name != "lowbit" && config.lut_name != "parity") {
        throw std::runtime_error("unsupported --lut value: " + config.lut_name);
    }

    const bool dim_reduction_enabled =
        config.has_enable_lwe_dim_reduction_override ? config.enable_lwe_dim_reduction_override : Params::kEnableLweDimReduction;
    if (!dim_reduction_enabled && Params::kLweFrontendDimension != Params::kLweAccumulatorDimension) {
        throw std::runtime_error("frontend/accumulator dimensions differ while dimension reduction is disabled");
    }
}

template <typename Params>
std::vector<size_t> BuildLut(const ExperimentConfig &config) {
    (void)config;
    return bdf17::BuildParityLut<Params>();
}

std::string BitsLE(const uint64_t value, const size_t width) {
    std::string bits;
    bits.reserve(width);
    for (size_t i = 0; i < width; ++i) {
        bits.push_back(((value >> i) & 1ULL) != 0 ? '1' : '0');
    }
    return bits;
}

template <typename Params>
void PrintConfigText(const ExperimentConfig &config, bool dim_reduction_enabled, bool dim_reduction_active) {
    std::cout << "resolved_config" << std::endl;
    std::cout << "  profile=" << config.profile_name << std::endl;
    std::cout << "  expcrt_variant=" << VariantName(config.expcrt_variant) << std::endl;
    std::cout << "  lut=" << config.lut_name << std::endl;
    std::cout << "  seed=" << config.seed << std::endl;
    std::cout << "  trials=" << config.num_trials << std::endl;
    std::cout << "  plain_modulus=" << Params::kPlainModulus << std::endl;
    std::cout << "  lwe_frontend_dim=" << Params::kLweFrontendDimension << std::endl;
    std::cout << "  lwe_accumulator_dim=" << Params::kLweAccumulatorDimension << std::endl;
    std::cout << "  lwe_noise_var=" << Params::kLweNoiseVar << std::endl;
    std::cout << "  rlwe_noise_var=" << Params::kRlweNoiseVar << std::endl;
    std::cout << "  dim_reduction_enabled=" << (dim_reduction_enabled ? "true" : "false") << std::endl;
    std::cout << "  dim_reduction_active=" << (dim_reduction_active ? "true" : "false") << std::endl;
    std::cout << std::endl;
}

void PrintTrialsText(const std::vector<TrialResult> &trials) {
    size_t pass = 0;
    size_t fail = 0;
    for (const auto &trial : trials) {
        std::cout << "trial=" << trial.trial_index << " bits_le=" << trial.bits_le << " packed=" << trial.packed_plain
                  << " expected=" << trial.expected << " got=" << trial.got << " phase=" << trial.phase
                  << " status=" << (trial.ok ? "OK" : "FAIL") << std::endl;
        if (trial.ok) {
            ++pass;
        } else {
            ++fail;
        }
    }
    std::cout << std::endl;
    std::cout << "summary pass=" << pass << " fail=" << fail << std::endl;
}

template <typename Params>
void PrintJson(const ExperimentConfig &config, bool dim_reduction_enabled, bool dim_reduction_active, const std::vector<TrialResult> &trials) {
    size_t pass = 0;
    size_t fail = 0;
    for (const auto &trial : trials) {
        if (trial.ok) {
            ++pass;
        } else {
            ++fail;
        }
    }

    std::cout << "{" << std::endl;
    std::cout << "  \"profile\":\"" << config.profile_name << "\"," << std::endl;
    std::cout << "  \"expcrt_variant\":\"" << VariantName(config.expcrt_variant) << "\"," << std::endl;
    std::cout << "  \"lut\":\"" << config.lut_name << "\"," << std::endl;
    std::cout << "  \"seed\":" << config.seed << "," << std::endl;
    std::cout << "  \"trials\":" << config.num_trials << "," << std::endl;
    std::cout << "  \"plain_modulus\":" << Params::kPlainModulus << "," << std::endl;
    std::cout << "  \"lwe_frontend_dim\":" << Params::kLweFrontendDimension << "," << std::endl;
    std::cout << "  \"lwe_accumulator_dim\":" << Params::kLweAccumulatorDimension << "," << std::endl;
    std::cout << "  \"lwe_noise_var\":" << Params::kLweNoiseVar << "," << std::endl;
    std::cout << "  \"rlwe_noise_var\":" << Params::kRlweNoiseVar << "," << std::endl;
    std::cout << "  \"dim_reduction_enabled\":" << (dim_reduction_enabled ? "true" : "false") << "," << std::endl;
    std::cout << "  \"dim_reduction_active\":" << (dim_reduction_active ? "true" : "false") << "," << std::endl;
    std::cout << "  \"trial_results\":[" << std::endl;
    for (size_t i = 0; i < trials.size(); ++i) {
        const auto &trial = trials[i];
        std::cout << "    {\"trial\":" << trial.trial_index << ",\"bits_le\":\"" << trial.bits_le
                  << "\",\"packed_plain\":" << trial.packed_plain << ",\"phase\":" << trial.phase
                  << ",\"expected\":" << trial.expected << ",\"got\":" << trial.got
                  << ",\"ok\":" << (trial.ok ? "true" : "false") << "}";
        if (i + 1 != trials.size()) {
            std::cout << ",";
        }
        std::cout << std::endl;
    }
    std::cout << "  ]," << std::endl;
    std::cout << "  \"summary\":{\"pass\":" << pass << ",\"fail\":" << fail << "}" << std::endl;
    std::cout << "}" << std::endl;
}

template <typename Params>
int RunExperiment(const ExperimentConfig &config) {
    const bool dim_reduction_enabled =
        config.has_enable_lwe_dim_reduction_override ? config.enable_lwe_dim_reduction_override : Params::kEnableLweDimReduction;
    const bool dim_reduction_active = dim_reduction_enabled && Params::kLweFrontendDimension != Params::kLweAccumulatorDimension;

    const uint64_t q_frontend = Params::kFrontendModulus;
    const uint64_t q_accumulator_input = Params::kAccumulatorInputModulus;
    const uint64_t q_extract_internal = Params::kExtractModulus;
    const size_t packing_width = bdf17::MaxPackingBits(Params::kPlainModulus);

    bdf17::RandomContext rng(config.seed);
    std::uniform_int_distribution<int> bit_dist(0, 1);

    const std::vector<int64_t> lwe_secret_frontend =
        GaussianSampler<Params::kLweFrontendDimension>::SampleSk(Params::kLweSecretDensity, rng.engine);

    std::vector<int64_t> lwe_secret_accumulator;
    std::optional<bdf17::LweKeySwitchKey> frontend_to_accumulator_ksk;
    if (dim_reduction_active) {
        lwe_secret_accumulator =
            GaussianSampler<Params::kLweAccumulatorDimension>::SampleSk(Params::kLweSecretDensity, rng.engine);
        frontend_to_accumulator_ksk = bdf17::GenerateLweKeySwitchKey(
            lwe_secret_frontend,
            lwe_secret_accumulator,
            q_frontend,
            Params::kLweKeySwitchBase,
            Params::kLweNoiseVar,
            rng.engine);
    } else {
        lwe_secret_accumulator = lwe_secret_frontend;
    }

    const auto plain_lut = BuildLut<Params>(config);
    const auto lut_samples = bdf17::BuildTensorLutSamples<Params>(plain_lut);
    const auto lut_coeff = bdf17::ConstructLutPoly<Params>(lut_samples);
    typename Params::PlanPQ plan_pq;
    const auto lut_eval = plan_pq.forward(lut_coeff);

    bdf17::AccumulatorState<Params> accumulator(lwe_secret_accumulator, rng, Params::kRlweNoiseVar);
    bdf17::TensorExpCrtState<Params> expcrt(
        lwe_secret_frontend,
        accumulator.sk_p,
        accumulator.sk_q,
        rng,
        Params::kRlweNoiseVar);

    std::vector<TrialResult> trials;
    trials.reserve(config.num_trials);
    bool has_failure = false;
    for (size_t trial_index = 0; trial_index < config.num_trials; ++trial_index) {
        TrialResult trial;
        trial.trial_index = trial_index;

        uint64_t packed_plain = 0;
        std::vector<bdf17::LweCiphertext> bit_ciphertexts;
        bit_ciphertexts.reserve(packing_width);
        for (size_t i = 0; i < packing_width; ++i) {
            const uint64_t bit = static_cast<uint64_t>(bit_dist(rng.engine));
            packed_plain |= bit << i;
            bit_ciphertexts.push_back(bdf17::EncryptLwe(
                lwe_secret_frontend,
                bit,
                Params::kPlainModulus,
                q_frontend,
                Params::kLweNoiseVar,
                rng.engine));
        }
        trial.packed_plain = packed_plain;
        trial.bits_le = BitsLE(packed_plain, packing_width);

        auto packed_frontend = bdf17::PackBitsCiphertextsLE(bit_ciphertexts, Params::kPlainModulus, q_frontend);
        bdf17::LweCiphertext packed_for_accumulator = packed_frontend;
        if (dim_reduction_active) {
            packed_for_accumulator = bdf17::ApplyLweKeySwitch(packed_for_accumulator, *frontend_to_accumulator_ksk);
        }

        auto packed_internal =
            bdf17::ModSwitchLwe(packed_for_accumulator, q_frontend, q_accumulator_input, Params::kPlainModulus);
        std::vector<int64_t> a(Params::kLweAccumulatorDimension, 0);
        for (size_t i = 0; i < Params::kLweAccumulatorDimension; ++i) {
            a[i] = static_cast<int64_t>(packed_internal.a[i]);
        }
        int64_t b = static_cast<int64_t>(packed_internal.b);

        auto ct_p = Params::SchemeP::template ModSwitch<typename Params::SchemePt>(
            accumulator.scheme_p.Process(accumulator.bk_p, a, b, Params::kPlainModulus));
        auto ct_q = Params::SchemeQ::template ModSwitch<typename Params::SchemeQt>(
            accumulator.scheme_q.Process(accumulator.bk_q, a, b, Params::kPlainModulus));

        auto tensor_ct = bdf17::ExpCRT<Params>(expcrt, ct_p, ct_q, config.expcrt_variant);
        auto extracted = bdf17::FunExtract<Params>(std::move(tensor_ct), lut_eval);

        bdf17::LweCiphertext extracted_internal{extracted.a, extracted.b};
        bdf17::LweCiphertext final_frontend = extracted_internal;
        if (q_extract_internal != q_frontend) {
            final_frontend = bdf17::ModSwitchLwe(extracted_internal, q_extract_internal, q_frontend, Params::kPlainModulus);
        }

        trial.phase = bdf17::DecryptPhase(final_frontend, lwe_secret_frontend, q_frontend);
        trial.got = static_cast<size_t>(bdf17::DecodeMessage(trial.phase, Params::kPlainModulus, q_frontend));
        trial.expected = plain_lut[packed_plain];
        trial.ok = trial.got == trial.expected;
        has_failure = has_failure || !trial.ok;
        trials.push_back(std::move(trial));
    }

    if (config.emit_json) {
        PrintJson<Params>(config, dim_reduction_enabled, dim_reduction_active, trials);
    } else {
        PrintConfigText<Params>(config, dim_reduction_enabled, dim_reduction_active);
        PrintTrialsText(trials);
    }
    return has_failure ? 1 : 0;
}

} // namespace

int main(int argc, char **argv) {
    using Params = bdf17::DefaultParams;
    try {
        const ExperimentConfig config = ParseExperimentConfig(argc, argv);
        ValidateExperimentConfig<Params>(config);
        return RunExperiment<Params>(config);
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
