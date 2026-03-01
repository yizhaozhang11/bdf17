#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "accumulator.hpp"
#include "expcrt.hpp"
#include "fun_extract.hpp"
#include "lwe_frontend.hpp"
#include "params.hpp"

namespace {

using Clock = std::chrono::steady_clock;

#if defined(BDF17_ENABLE_AVX2)
constexpr bool kBackendAvx2Enabled = true;
#else
constexpr bool kBackendAvx2Enabled = false;
#endif

#if defined(BDF17_ENABLE_AVX512)
constexpr bool kBackendAvx512Enabled = true;
#else
constexpr bool kBackendAvx512Enabled = false;
#endif

constexpr const char *kProfileName = "DefaultParams";
constexpr const char *kLutLabel = "lowbit";
constexpr uint64_t kDefaultSeed = 0xBDF170001ULL;
constexpr bdf17::ExpCrtVariant kVariant = bdf17::ExpCrtVariant::TensorTrick;

uint64_t MixSeed(const uint64_t seed, const uint64_t stream) {
    uint64_t x = seed + 0x9E3779B97F4A7C15ULL + (stream << 1);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

struct RunOptions {
    uint64_t seed = kDefaultSeed;
    size_t trials = 0;
    bool json = false;
};

struct SetupStats {
    double frontend_sk_gen_ms = 0.0;
    double frontend_to_accumulator_ksk_gen_ms = 0.0;
    double lut_build_ms = 0.0;
    double lut_eval_ms = 0.0;
    double accumulator_state_gen_ms = 0.0;
    double expcrt_state_gen_ms = 0.0;
    double total_ms = 0.0;
};

struct TrialStats {
    size_t trial_index = 0;
    std::string bits_le;
    uint64_t packed_plain = 0;
    size_t expected = 0;
    size_t got = 0;
    uint64_t phase = 0;
    uint64_t ideal_phase = 0;
    int64_t centered_phase_error = 0;
    double bucket_error_fraction = 0.0;
    double encrypt_bits_ms = 0.0;
    double pack_bits_ms = 0.0;
    double frontend_to_accumulator_ks_ms = 0.0;
    double input_modswitch_ms = 0.0;
    double acc_p_ms = 0.0;
    double acc_q_ms = 0.0;
    double expcrt_ms = 0.0;
    double fun_extract_ms = 0.0;
    double output_modswitch_ms = 0.0;
    double decrypt_ms = 0.0;
    double total_ms = 0.0;
    bool ok = false;
};

struct AggregateStats {
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

double ElapsedMs(const Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string BitsLE(const uint64_t value, const size_t width) {
    std::string bits;
    bits.reserve(width);
    for (size_t i = 0; i < width; ++i) {
        bits.push_back(((value >> i) & 1ULL) != 0 ? '1' : '0');
    }
    return bits;
}

int64_t CenteredResidue(const uint64_t value, const uint64_t mod) {
    if (mod == 0) {
        throw std::runtime_error("modulus must be non-zero");
    }
    const uint64_t reduced = value % mod;
    const uint64_t threshold = (mod + 1) / 2;
    if (reduced >= threshold) {
        return static_cast<int64_t>(reduced) - static_cast<int64_t>(mod);
    }
    return static_cast<int64_t>(reduced);
}

int64_t CenteredDifference(const uint64_t lhs, const uint64_t rhs, const uint64_t mod) {
    if (mod == 0) {
        throw std::runtime_error("modulus must be non-zero");
    }
    const uint64_t lhs_mod = lhs % mod;
    const uint64_t rhs_mod = rhs % mod;
    const uint64_t diff = lhs_mod >= rhs_mod ? lhs_mod - rhs_mod : mod - (rhs_mod - lhs_mod);
    return CenteredResidue(diff, mod);
}

AggregateStats ComputeStats(const std::vector<double> &values) {
    AggregateStats out;
    if (values.empty()) {
        return out;
    }

    out.min = *std::min_element(values.begin(), values.end());
    out.max = *std::max_element(values.begin(), values.end());
    double sum = 0.0;
    for (const double v : values) {
        sum += v;
    }
    out.mean = sum / static_cast<double>(values.size());

    double variance = 0.0;
    for (const double v : values) {
        const double d = v - out.mean;
        variance += d * d;
    }
    variance /= static_cast<double>(values.size());
    out.stddev = std::sqrt(variance);
    return out;
}

std::string FormatDouble(const double value, const int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

const char *BoolStr(const bool value) {
    return value ? "true" : "false";
}

const char *VariantName(const bdf17::ExpCrtVariant variant) {
    switch (variant) {
        case bdf17::ExpCrtVariant::TensorTrick:
            return "TensorTrick";
        case bdf17::ExpCrtVariant::Paper:
            return "Paper";
    }
    return "Unknown";
}

void PrintUsage(const char *argv0) {
    std::cout << "Usage: " << argv0 << " [--seed <uint64>] [--trials <size_t>] [--json]" << std::endl;
}

uint64_t ParseUint64(const std::string &text, const std::string &flag) {
    try {
        size_t consumed = 0;
        const unsigned long long parsed = std::stoull(text, &consumed, 10);
        if (consumed != text.size()) {
            throw std::runtime_error("invalid " + flag + " value: " + text);
        }
        return static_cast<uint64_t>(parsed);
    } catch (const std::exception &) {
        throw std::runtime_error("invalid " + flag + " value: " + text);
    }
}

RunOptions ParseRunOptions(const int argc, char **argv, const size_t default_trials) {
    RunOptions opts;
    opts.trials = default_trials;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--json") {
            opts.json = true;
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        if (arg == "--seed") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --seed");
            }
            opts.seed = ParseUint64(argv[++i], "--seed");
            continue;
        }
        if (arg == "--trials") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --trials");
            }
            opts.trials = static_cast<size_t>(ParseUint64(argv[++i], "--trials"));
            continue;
        }
        if (arg.rfind("--seed=", 0) == 0) {
            opts.seed = ParseUint64(arg.substr(7), "--seed");
            continue;
        }
        if (arg.rfind("--trials=", 0) == 0) {
            opts.trials = static_cast<size_t>(ParseUint64(arg.substr(9), "--trials"));
            continue;
        }
        throw std::runtime_error("unknown argument: " + arg);
    }

    if (opts.trials == 0) {
        throw std::runtime_error("--trials must be greater than zero");
    }
    return opts;
}

template <typename Params>
void PrintRunHeader(
    const RunOptions &opts,
    const size_t packing_width,
    const uint64_t q_frontend,
    const uint64_t q_accumulator_input,
    const uint64_t q_extract_internal,
    const bool dim_reduction_active) {
    std::cout << "bdf17 demo" << std::endl;
    std::cout << "profile=" << kProfileName << " variant=" << VariantName(kVariant) << " lut=" << kLutLabel << " seed=" << opts.seed
              << " trials=" << opts.trials << std::endl;
    std::cout << "plain_modulus=" << Params::kPlainModulus << " packing_width=" << packing_width << " bit_order=lsb-first"
              << std::endl;
    std::cout << "lwe_frontend_dim=" << Params::kLweFrontendDimension << " lwe_accumulator_dim=" << Params::kLweAccumulatorDimension
              << " dim_reduction_enabled=" << BoolStr(Params::kEnableLweDimReduction)
              << " dim_reduction_active=" << BoolStr(dim_reduction_active) << std::endl;
    std::cout << "p=" << Params::EvalP::N << " q=" << Params::EvalQ::N << " pq=" << Params::kTensorDimension << std::endl;
    std::cout << "q_frontend=" << q_frontend << " q_accumulator_input=" << q_accumulator_input
              << " q_extract_internal=" << q_extract_internal << std::endl;
    std::cout << "rlwe_keyswitch_base=" << Params::kKeySwitchBase << " lwe_keyswitch_base=" << Params::kLweKeySwitchBase << std::endl;
    std::cout << "backend_avx2=" << BoolStr(kBackendAvx2Enabled) << " backend_avx512=" << BoolStr(kBackendAvx512Enabled)
              << std::endl;
    std::cout << std::endl;
}

void PrintSetup(const SetupStats &setup) {
    std::cout << "setup_ms{frontend_sk=" << FormatDouble(setup.frontend_sk_gen_ms)
              << " lwe_ksk=" << FormatDouble(setup.frontend_to_accumulator_ksk_gen_ms)
              << " lut_build=" << FormatDouble(setup.lut_build_ms) << " lut_eval=" << FormatDouble(setup.lut_eval_ms)
              << " accumulator=" << FormatDouble(setup.accumulator_state_gen_ms)
              << " expcrt_state=" << FormatDouble(setup.expcrt_state_gen_ms) << " total=" << FormatDouble(setup.total_ms) << "}"
              << std::endl
              << std::endl;
}

void PrintTrial(const TrialStats &s) {
    std::cout << "trial=" << s.trial_index << " bits_le=" << s.bits_le << " packed=" << s.packed_plain << " expected=" << s.expected
              << " got=" << s.got << " status=" << (s.ok ? "OK" : "FAIL") << std::endl;
    std::cout << "  ms{enc=" << FormatDouble(s.encrypt_bits_ms) << " pack=" << FormatDouble(s.pack_bits_ms)
              << " ks=" << FormatDouble(s.frontend_to_accumulator_ks_ms) << " msw_in=" << FormatDouble(s.input_modswitch_ms)
              << " acc_p=" << FormatDouble(s.acc_p_ms) << " acc_q=" << FormatDouble(s.acc_q_ms)
              << " expcrt=" << FormatDouble(s.expcrt_ms) << " extract=" << FormatDouble(s.fun_extract_ms)
              << " msw_out=" << FormatDouble(s.output_modswitch_ms) << " dec=" << FormatDouble(s.decrypt_ms)
              << " total=" << FormatDouble(s.total_ms) << "}" << std::endl;
    std::cout << "  diag{phase=" << s.phase << " ideal=" << s.ideal_phase << " centered_err=" << s.centered_phase_error
              << " bucket_frac=" << FormatDouble(s.bucket_error_fraction, 6) << "}" << std::endl;
    std::cout << std::endl;
}

void PrintFailureDiagnostic(const uint64_t seed, const TrialStats &s) {
    std::cout << "failure_diagnostic{" << std::endl;
    std::cout << "  seed=" << seed << " trial=" << s.trial_index << " bits_le=" << s.bits_le << " packed=" << s.packed_plain
              << std::endl;
    std::cout << "  expected=" << s.expected << " got=" << s.got << " phase=" << s.phase << " ideal_phase=" << s.ideal_phase
              << std::endl;
    std::cout << "  centered_phase_error=" << s.centered_phase_error
              << " bucket_error_fraction=" << FormatDouble(s.bucket_error_fraction, 6) << std::endl;
    std::cout << "  timings_ms{enc=" << FormatDouble(s.encrypt_bits_ms) << " pack=" << FormatDouble(s.pack_bits_ms)
              << " ks=" << FormatDouble(s.frontend_to_accumulator_ks_ms) << " msw_in=" << FormatDouble(s.input_modswitch_ms)
              << " acc_p=" << FormatDouble(s.acc_p_ms) << " acc_q=" << FormatDouble(s.acc_q_ms)
              << " expcrt=" << FormatDouble(s.expcrt_ms) << " extract=" << FormatDouble(s.fun_extract_ms)
              << " msw_out=" << FormatDouble(s.output_modswitch_ms) << " dec=" << FormatDouble(s.decrypt_ms)
              << " total=" << FormatDouble(s.total_ms) << "}" << std::endl;
    std::cout << "}" << std::endl << std::endl;
}

void PrintSummary(const SetupStats &setup, const std::vector<TrialStats> &trials) {
    std::vector<double> acc_p_ms;
    std::vector<double> acc_q_ms;
    std::vector<double> expcrt_ms;
    std::vector<double> extract_ms;
    std::vector<double> total_ms;

    size_t pass = 0;
    size_t fail = 0;
    acc_p_ms.reserve(trials.size());
    acc_q_ms.reserve(trials.size());
    expcrt_ms.reserve(trials.size());
    extract_ms.reserve(trials.size());
    total_ms.reserve(trials.size());
    for (const auto &trial : trials) {
        acc_p_ms.push_back(trial.acc_p_ms);
        acc_q_ms.push_back(trial.acc_q_ms);
        expcrt_ms.push_back(trial.expcrt_ms);
        extract_ms.push_back(trial.fun_extract_ms);
        total_ms.push_back(trial.total_ms);
        if (trial.ok) {
            ++pass;
        } else {
            ++fail;
        }
    }

    const AggregateStats acc_p = ComputeStats(acc_p_ms);
    const AggregateStats acc_q = ComputeStats(acc_q_ms);
    const AggregateStats expcrt = ComputeStats(expcrt_ms);
    const AggregateStats extract = ComputeStats(extract_ms);
    const AggregateStats total = ComputeStats(total_ms);
    const double throughput = total.mean > 0.0 ? 1000.0 / total.mean : 0.0;

    std::cout << "summary" << std::endl;
    std::cout << "pass=" << pass << " fail=" << fail << std::endl;
    std::cout << "setup_ms{frontend_sk=" << FormatDouble(setup.frontend_sk_gen_ms)
              << " lwe_ksk=" << FormatDouble(setup.frontend_to_accumulator_ksk_gen_ms)
              << " lut_build=" << FormatDouble(setup.lut_build_ms) << " lut_eval=" << FormatDouble(setup.lut_eval_ms)
              << " accumulator=" << FormatDouble(setup.accumulator_state_gen_ms)
              << " expcrt_state=" << FormatDouble(setup.expcrt_state_gen_ms) << " total=" << FormatDouble(setup.total_ms) << "}"
              << std::endl;
    std::cout << "mean_ms{acc_p=" << FormatDouble(acc_p.mean) << " acc_q=" << FormatDouble(acc_q.mean)
              << " expcrt=" << FormatDouble(expcrt.mean) << " extract=" << FormatDouble(extract.mean)
              << " total=" << FormatDouble(total.mean) << "}" << std::endl;
    std::cout << "min_ms{acc_p=" << FormatDouble(acc_p.min) << " acc_q=" << FormatDouble(acc_q.min)
              << " expcrt=" << FormatDouble(expcrt.min) << " extract=" << FormatDouble(extract.min)
              << " total=" << FormatDouble(total.min) << "}" << std::endl;
    std::cout << "max_ms{acc_p=" << FormatDouble(acc_p.max) << " acc_q=" << FormatDouble(acc_q.max)
              << " expcrt=" << FormatDouble(expcrt.max) << " extract=" << FormatDouble(extract.max)
              << " total=" << FormatDouble(total.max) << "}" << std::endl;
    std::cout << "stddev_ms{acc_p=" << FormatDouble(acc_p.stddev) << " acc_q=" << FormatDouble(acc_q.stddev)
              << " expcrt=" << FormatDouble(expcrt.stddev) << " extract=" << FormatDouble(extract.stddev)
              << " total=" << FormatDouble(total.stddev) << "}" << std::endl;
    std::cout << "throughput_trials_per_sec=" << FormatDouble(throughput, 4) << std::endl;
}

void PrintJson(
    const RunOptions &opts,
    const SetupStats &setup,
    const std::vector<TrialStats> &trials,
    const size_t packing_width,
    const uint64_t q_frontend,
    const uint64_t q_accumulator_input,
    const uint64_t q_extract_internal,
    const bool dim_reduction_active) {
    using Params = bdf17::DefaultParams;

    std::vector<double> acc_p_ms;
    std::vector<double> acc_q_ms;
    std::vector<double> expcrt_ms;
    std::vector<double> extract_ms;
    std::vector<double> total_ms;

    size_t pass = 0;
    size_t fail = 0;
    for (const auto &trial : trials) {
        acc_p_ms.push_back(trial.acc_p_ms);
        acc_q_ms.push_back(trial.acc_q_ms);
        expcrt_ms.push_back(trial.expcrt_ms);
        extract_ms.push_back(trial.fun_extract_ms);
        total_ms.push_back(trial.total_ms);
        if (trial.ok) {
            ++pass;
        } else {
            ++fail;
        }
    }

    const AggregateStats acc_p = ComputeStats(acc_p_ms);
    const AggregateStats acc_q = ComputeStats(acc_q_ms);
    const AggregateStats expcrt = ComputeStats(expcrt_ms);
    const AggregateStats extract = ComputeStats(extract_ms);
    const AggregateStats total = ComputeStats(total_ms);
    const double throughput = total.mean > 0.0 ? 1000.0 / total.mean : 0.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{" << std::endl;
    std::cout << "  \"demo\":\"bdf17\"," << std::endl;
    std::cout << "  \"profile\":\"" << kProfileName << "\"," << std::endl;
    std::cout << "  \"variant\":\"" << VariantName(kVariant) << "\"," << std::endl;
    std::cout << "  \"lut\":\"" << kLutLabel << "\"," << std::endl;
    std::cout << "  \"seed\":" << opts.seed << "," << std::endl;
    std::cout << "  \"trials\":" << opts.trials << "," << std::endl;
    std::cout << "  \"config\":{" << std::endl;
    std::cout << "    \"plain_modulus\":" << Params::kPlainModulus << "," << std::endl;
    std::cout << "    \"packing_width\":" << packing_width << "," << std::endl;
    std::cout << "    \"bit_order\":\"lsb-first\"," << std::endl;
    std::cout << "    \"lwe_frontend_dim\":" << Params::kLweFrontendDimension << "," << std::endl;
    std::cout << "    \"lwe_accumulator_dim\":" << Params::kLweAccumulatorDimension << "," << std::endl;
    std::cout << "    \"dim_reduction_enabled\":" << BoolStr(Params::kEnableLweDimReduction) << "," << std::endl;
    std::cout << "    \"dim_reduction_active\":" << BoolStr(dim_reduction_active) << "," << std::endl;
    std::cout << "    \"p\":" << Params::EvalP::N << "," << std::endl;
    std::cout << "    \"q\":" << Params::EvalQ::N << "," << std::endl;
    std::cout << "    \"pq\":" << Params::kTensorDimension << "," << std::endl;
    std::cout << "    \"q_frontend\":" << q_frontend << "," << std::endl;
    std::cout << "    \"q_accumulator_input\":" << q_accumulator_input << "," << std::endl;
    std::cout << "    \"q_extract_internal\":" << q_extract_internal << "," << std::endl;
    std::cout << "    \"rlwe_keyswitch_base\":" << Params::kKeySwitchBase << "," << std::endl;
    std::cout << "    \"lwe_keyswitch_base\":" << Params::kLweKeySwitchBase << "," << std::endl;
    std::cout << "    \"backend_avx2\":" << BoolStr(kBackendAvx2Enabled) << "," << std::endl;
    std::cout << "    \"backend_avx512\":" << BoolStr(kBackendAvx512Enabled) << std::endl;
    std::cout << "  }," << std::endl;
    std::cout << "  \"setup_ms\":{" << std::endl;
    std::cout << "    \"frontend_sk_gen_ms\":" << setup.frontend_sk_gen_ms << "," << std::endl;
    std::cout << "    \"frontend_to_accumulator_ksk_gen_ms\":" << setup.frontend_to_accumulator_ksk_gen_ms << "," << std::endl;
    std::cout << "    \"lut_build_ms\":" << setup.lut_build_ms << "," << std::endl;
    std::cout << "    \"lut_eval_ms\":" << setup.lut_eval_ms << "," << std::endl;
    std::cout << "    \"accumulator_state_gen_ms\":" << setup.accumulator_state_gen_ms << "," << std::endl;
    std::cout << "    \"expcrt_state_gen_ms\":" << setup.expcrt_state_gen_ms << "," << std::endl;
    std::cout << "    \"setup_total_ms\":" << setup.total_ms << std::endl;
    std::cout << "  }," << std::endl;
    std::cout << "  \"trial_stats\":[" << std::endl;
    for (size_t i = 0; i < trials.size(); ++i) {
        const auto &t = trials[i];
        std::cout << "    {\"trial\":" << t.trial_index << ",\"bits_le\":\"" << t.bits_le << "\",\"packed_plain\":" << t.packed_plain
                  << ",\"expected\":" << t.expected << ",\"got\":" << t.got << ",\"ok\":" << BoolStr(t.ok) << ",\"phase\":" << t.phase
                  << ",\"ideal_phase\":" << t.ideal_phase << ",\"centered_phase_error\":" << t.centered_phase_error
                  << ",\"bucket_error_fraction\":" << t.bucket_error_fraction
                  << ",\"encrypt_bits_ms\":" << t.encrypt_bits_ms << ",\"pack_bits_ms\":" << t.pack_bits_ms
                  << ",\"frontend_to_accumulator_ks_ms\":" << t.frontend_to_accumulator_ks_ms
                  << ",\"input_modswitch_ms\":" << t.input_modswitch_ms << ",\"acc_p_ms\":" << t.acc_p_ms
                  << ",\"acc_q_ms\":" << t.acc_q_ms << ",\"expcrt_ms\":" << t.expcrt_ms << ",\"fun_extract_ms\":" << t.fun_extract_ms
                  << ",\"output_modswitch_ms\":" << t.output_modswitch_ms << ",\"decrypt_ms\":" << t.decrypt_ms
                  << ",\"trial_total_ms\":" << t.total_ms << "}";
        if (i + 1 != trials.size()) {
            std::cout << ",";
        }
        std::cout << std::endl;
    }
    std::cout << "  ]," << std::endl;
    std::cout << "  \"summary\":{" << std::endl;
    std::cout << "    \"pass\":" << pass << "," << std::endl;
    std::cout << "    \"fail\":" << fail << "," << std::endl;
    std::cout << "    \"mean_ms\":{\"acc_p\":" << acc_p.mean << ",\"acc_q\":" << acc_q.mean << ",\"expcrt\":" << expcrt.mean
              << ",\"extract\":" << extract.mean << ",\"total\":" << total.mean << "}," << std::endl;
    std::cout << "    \"min_ms\":{\"acc_p\":" << acc_p.min << ",\"acc_q\":" << acc_q.min << ",\"expcrt\":" << expcrt.min
              << ",\"extract\":" << extract.min << ",\"total\":" << total.min << "}," << std::endl;
    std::cout << "    \"max_ms\":{\"acc_p\":" << acc_p.max << ",\"acc_q\":" << acc_q.max << ",\"expcrt\":" << expcrt.max
              << ",\"extract\":" << extract.max << ",\"total\":" << total.max << "}," << std::endl;
    std::cout << "    \"stddev_ms\":{\"acc_p\":" << acc_p.stddev << ",\"acc_q\":" << acc_q.stddev << ",\"expcrt\":" << expcrt.stddev
              << ",\"extract\":" << extract.stddev << ",\"total\":" << total.stddev << "}," << std::endl;
    std::cout << "    \"throughput_trials_per_sec\":" << throughput << std::endl;
    std::cout << "  }" << std::endl;
    std::cout << "}" << std::endl;
}

} // namespace

int main(int argc, char **argv) {
    using Params = bdf17::DefaultParams;

    try {
        const RunOptions options = ParseRunOptions(argc, argv, Params::kNumTrials);
        const uint64_t q_frontend = Params::kFrontendModulus;
        const uint64_t q_accumulator_input = Params::kAccumulatorInputModulus;
        const uint64_t q_extract_internal = Params::kExtractModulus;
        const size_t packing_width = bdf17::MaxPackingBits(Params::kPlainModulus);
        const bool dim_reduction_active =
            Params::kEnableLweDimReduction && Params::kLweFrontendDimension != Params::kLweAccumulatorDimension;

        if (!options.json) {
            PrintRunHeader<Params>(
                options, packing_width, q_frontend, q_accumulator_input, q_extract_internal, dim_reduction_active);
        }

        std::mt19937_64 engine(options.seed);
        std::uniform_int_distribution<int> bit_dist(0, 1);

        // Seed all sampler dimensions used in this pipeline so seeded runs are reproducible.
        GaussianSampler<Params::kLweFrontendDimension>::GetInstance().Seed(MixSeed(options.seed, 1));
        if constexpr (Params::kLweAccumulatorDimension != Params::kLweFrontendDimension) {
            GaussianSampler<Params::kLweAccumulatorDimension>::GetInstance().Seed(MixSeed(options.seed, 2));
        }
        if constexpr (
            Params::EvalP::N != Params::kLweFrontendDimension && Params::EvalP::N != Params::kLweAccumulatorDimension) {
            GaussianSampler<Params::EvalP::N>::GetInstance().Seed(MixSeed(options.seed, 3));
        }
        if constexpr (
            Params::EvalQ::N != Params::kLweFrontendDimension && Params::EvalQ::N != Params::kLweAccumulatorDimension &&
            Params::EvalQ::N != Params::EvalP::N) {
            GaussianSampler<Params::EvalQ::N>::GetInstance().Seed(MixSeed(options.seed, 4));
        }
        if constexpr (
            Params::kTensorDimension != Params::kLweFrontendDimension &&
            Params::kTensorDimension != Params::kLweAccumulatorDimension && Params::kTensorDimension != Params::EvalP::N &&
            Params::kTensorDimension != Params::EvalQ::N) {
            GaussianSampler<Params::kTensorDimension>::GetInstance().Seed(MixSeed(options.seed, 5));
        }

        SetupStats setup;
        const auto setup_start = Clock::now();

        const auto frontend_stage = Clock::now();
        std::vector<int64_t> lwe_secret_frontend =
            GaussianSampler<Params::kLweFrontendDimension>::GetInstance().SampleSk(Params::kLweSecretDensity);
        setup.frontend_sk_gen_ms = ElapsedMs(frontend_stage);

        std::vector<int64_t> lwe_secret_accumulator;
        std::optional<bdf17::LweKeySwitchKey> frontend_to_accumulator_ksk;
        if (dim_reduction_active) {
            const auto ksk_stage = Clock::now();
            lwe_secret_accumulator =
                GaussianSampler<Params::kLweAccumulatorDimension>::GetInstance().SampleSk(Params::kLweSecretDensity);
            frontend_to_accumulator_ksk = bdf17::GenerateLweKeySwitchKey(
                lwe_secret_frontend,
                lwe_secret_accumulator,
                q_frontend,
                Params::kLweKeySwitchBase,
                Params::kLweNoiseVar,
                engine);
            setup.frontend_to_accumulator_ksk_gen_ms = ElapsedMs(ksk_stage);
        } else {
            if (Params::kLweFrontendDimension != Params::kLweAccumulatorDimension) {
                throw std::runtime_error("frontend/accumulator dimensions differ while dimension reduction is disabled");
            }
            lwe_secret_accumulator = lwe_secret_frontend;
        }

        const auto lut_build_stage = Clock::now();
        auto plain_lut = bdf17::BuildParityLut<Params>();
        auto lut_samples = bdf17::BuildTensorLutSamples<Params>(plain_lut);
        auto lut_coeff = bdf17::ConstructLutPoly<Params>(lut_samples);
        setup.lut_build_ms = ElapsedMs(lut_build_stage);

        const auto lut_eval_stage = Clock::now();
        Params::PlanPQ plan_pq;
        auto lut_eval = plan_pq.forward(lut_coeff);
        setup.lut_eval_ms = ElapsedMs(lut_eval_stage);

        const auto accumulator_stage = Clock::now();
        bdf17::AccumulatorState<Params> accumulator(lwe_secret_accumulator, MixSeed(options.seed, 11));
        setup.accumulator_state_gen_ms = ElapsedMs(accumulator_stage);

        const auto expcrt_stage = Clock::now();
        bdf17::TensorExpCrtState<Params> expcrt(
            lwe_secret_frontend, accumulator.sk_p, accumulator.sk_q, MixSeed(options.seed, 12));
        setup.expcrt_state_gen_ms = ElapsedMs(expcrt_stage);

        setup.total_ms = ElapsedMs(setup_start);

        if (!options.json) {
            PrintSetup(setup);
        }

        std::vector<TrialStats> trials;
        trials.reserve(options.trials);
        bool has_failure = false;
        for (size_t test_index = 0; test_index < options.trials; ++test_index) {
            TrialStats trial;
            trial.trial_index = test_index;
            const auto trial_start = Clock::now();

            uint64_t packed_plain = 0;
            std::vector<bdf17::LweCiphertext> bit_ciphertexts;
            bit_ciphertexts.reserve(packing_width);

            const auto enc_stage = Clock::now();
            for (size_t i = 0; i < packing_width; ++i) {
                const uint64_t bit = static_cast<uint64_t>(bit_dist(engine));
                packed_plain |= bit << i;
                bit_ciphertexts.push_back(bdf17::EncryptLwe(
                    lwe_secret_frontend,
                    bit,
                    Params::kPlainModulus,
                    q_frontend,
                    Params::kLweNoiseVar,
                    engine));
            }
            trial.encrypt_bits_ms = ElapsedMs(enc_stage);
            trial.packed_plain = packed_plain;
            trial.bits_le = BitsLE(packed_plain, packing_width);

            const auto pack_stage = Clock::now();
            auto packed_frontend = bdf17::PackBitsCiphertextsLE(bit_ciphertexts, Params::kPlainModulus, q_frontend);
            trial.pack_bits_ms = ElapsedMs(pack_stage);

            bdf17::LweCiphertext packed_for_accumulator = packed_frontend;
            if (dim_reduction_active) {
                const auto ks_stage = Clock::now();
                packed_for_accumulator = bdf17::ApplyLweKeySwitch(packed_for_accumulator, *frontend_to_accumulator_ksk);
                trial.frontend_to_accumulator_ks_ms = ElapsedMs(ks_stage);
            }

            std::vector<int64_t> a(Params::kLweAccumulatorDimension, 0);
            int64_t b = 0;
            const auto modswitch_stage = Clock::now();
            auto packed_internal =
                bdf17::ModSwitchLwe(packed_for_accumulator, q_frontend, q_accumulator_input, Params::kPlainModulus);
            for (size_t i = 0; i < Params::kLweAccumulatorDimension; ++i) {
                a[i] = static_cast<int64_t>(packed_internal.a[i]);
            }
            b = static_cast<int64_t>(packed_internal.b);
            trial.input_modswitch_ms = ElapsedMs(modswitch_stage);

            const auto acc_p_stage = Clock::now();
            auto ct_p = Params::SchemeP::template ModSwitch<Params::SchemePt>(
                accumulator.scheme_p.Process(accumulator.bk_p, a, b, Params::kPlainModulus));
            trial.acc_p_ms = ElapsedMs(acc_p_stage);

            const auto acc_q_stage = Clock::now();
            auto ct_q = Params::SchemeQ::template ModSwitch<Params::SchemeQt>(
                accumulator.scheme_q.Process(accumulator.bk_q, a, b, Params::kPlainModulus));
            trial.acc_q_ms = ElapsedMs(acc_q_stage);

            const auto expcrt_trial_stage = Clock::now();
            auto tensor_ct = bdf17::ExpCRT<Params>(expcrt, ct_p, ct_q, kVariant);
            trial.expcrt_ms = ElapsedMs(expcrt_trial_stage);

            const auto extract_stage = Clock::now();
            auto extracted = bdf17::FunExtract<Params>(std::move(tensor_ct), lut_eval);
            trial.fun_extract_ms = ElapsedMs(extract_stage);

            bdf17::LweCiphertext extracted_internal{extracted.a, extracted.b};
            bdf17::LweCiphertext final_frontend = extracted_internal;
            const auto output_msw_stage = Clock::now();
            if (q_extract_internal != q_frontend) {
                final_frontend =
                    bdf17::ModSwitchLwe(extracted_internal, q_extract_internal, q_frontend, Params::kPlainModulus);
            }
            trial.output_modswitch_ms = ElapsedMs(output_msw_stage);

            const auto decrypt_stage = Clock::now();
            const uint64_t phase = bdf17::DecryptPhase(final_frontend, lwe_secret_frontend, q_frontend);
            trial.decrypt_ms = ElapsedMs(decrypt_stage);

            trial.phase = phase;
            trial.got = static_cast<size_t>(bdf17::DecodeMessage(phase, Params::kPlainModulus, q_frontend));
            trial.expected = plain_lut[packed_plain];
            trial.ideal_phase = bdf17::EncodeMessage(trial.expected, Params::kPlainModulus, q_frontend);
            trial.centered_phase_error = CenteredDifference(trial.phase, trial.ideal_phase, q_frontend);
            const double bucket_width = static_cast<double>(q_frontend) / static_cast<double>(Params::kPlainModulus);
            trial.bucket_error_fraction = static_cast<double>(trial.centered_phase_error) / bucket_width;
            trial.ok = trial.got == trial.expected;
            trial.total_ms = ElapsedMs(trial_start);

            if (!trial.ok) {
                has_failure = true;
            }

            if (!options.json) {
                PrintTrial(trial);
                if (!trial.ok) {
                    PrintFailureDiagnostic(options.seed, trial);
                }
            }
            trials.push_back(std::move(trial));
        }

        if (options.json) {
            PrintJson(
                options,
                setup,
                trials,
                packing_width,
                q_frontend,
                q_accumulator_input,
                q_extract_internal,
                dim_reduction_active);
        } else {
            PrintSummary(setup, trials);
        }

        return has_failure ? 1 : 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
