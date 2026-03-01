#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bootstrap_runner.hpp"
#include "experiment_config.hpp"
#include "params.hpp"

namespace {

constexpr uint64_t kDefaultSeed = 0xBDF170001ULL;
constexpr size_t kDefaultTrials = 8;
constexpr const char *kDefaultProfileName = "default";
constexpr const char *kDefaultLutName = "lowbit";

using bdf17::BootstrapMetrics;
using bdf17::BootstrapResult;
using bdf17::ExperimentConfig;
using bdf17::StageStats;

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
              << " [--lut <lowbit|hamming-parity|parity>] [--json] [--enable-lwe-dim-reduction <0|1>]" << std::endl;
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

void ValidateExperimentConfig(const ExperimentConfig &config) {
    if (config.num_trials == 0) {
        throw std::runtime_error("--trials must be greater than zero");
    }
    (void)bdf17::ParseLutKindOrThrow(config.lut_name);
}

template <typename Params>
void ValidateRunConfigOrThrow(const ExperimentConfig &config) {
    const bool dim_reduction_enabled =
        config.has_enable_lwe_dim_reduction_override
            ? config.enable_lwe_dim_reduction_override
            : Params::kEnableLweDimReduction;
    bdf17::ValidateProfileOrThrow<Params>(dim_reduction_enabled, config.expcrt_variant);
    (void)bdf17::ParseLutKindOrThrow(config.lut_name);
}

void PrintStageText(const char *name, const StageStats &stats, size_t num_trials) {
    const double avg_ms = num_trials == 0 ? 0.0 : stats.ms / static_cast<double>(num_trials);
    std::cout << "  " << name << " ms_total=" << stats.ms << " ms_avg=" << avg_ms << " forward_ntt=" << stats.forward_ntt
              << " inverse_ntt=" << stats.inverse_ntt << " galois=" << stats.galois << " keyswitch=" << stats.keyswitch
              << " extmult=" << stats.extmult << " trace_calls=" << stats.trace_calls << " approx_bytes=" << stats.approx_bytes
              << std::endl;
}

template <typename Params>
void PrintConfigText(const ExperimentConfig &config, bool dim_reduction_enabled, bool dim_reduction_active) {
    std::cout << "resolved_config" << std::endl;
    std::cout << "  profile=" << config.profile_name << std::endl;
    std::cout << "  profile_intent=" << bdf17::ProfileIntentName(Params::kProfileIntent) << std::endl;
    std::cout << "  expcrt_variant=" << VariantName(config.expcrt_variant) << std::endl;
    std::cout << "  lut=" << config.lut_name << std::endl;
    std::cout << "  seed=" << config.seed << std::endl;
    std::cout << "  trials=" << config.num_trials << std::endl;
    std::cout << "  p_degree=" << Params::EvalP::N << std::endl;
    std::cout << "  q_degree=" << Params::EvalQ::N << std::endl;
    std::cout << "  tensor_dimension=" << Params::kTensorDimension << std::endl;
    std::cout << "  plain_modulus=" << Params::kPlainModulus << std::endl;
    std::cout << "  lwe_frontend_dim=" << Params::kLweFrontendDimension << std::endl;
    std::cout << "  lwe_accumulator_dim=" << Params::kLweAccumulatorDimension << std::endl;
    std::cout << "  lwe_noise_var=" << Params::kLweNoiseVar << std::endl;
    std::cout << "  rlwe_noise_var=" << Params::kRlweNoiseVar << std::endl;
    std::cout << "  q_frontend=" << Params::kFrontendModulus << std::endl;
    std::cout << "  q_accumulator_input=" << Params::kAccumulatorInputModulus << std::endl;
    std::cout << "  q_extract_internal=" << Params::kExtractModulus << std::endl;
    std::cout << "  dim_reduction_enabled=" << (dim_reduction_enabled ? "true" : "false") << std::endl;
    std::cout << "  dim_reduction_active=" << (dim_reduction_active ? "true" : "false") << std::endl;
    std::cout << std::endl;
}

void PrintTrialsText(const std::vector<BootstrapResult> &trials) {
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

void PrintMetricsText(const BootstrapMetrics &metrics, size_t num_trials) {
    std::cout << std::endl;
    std::cout << "stage_metrics" << std::endl;
    PrintStageText("encrypt_bits", metrics.encrypt_bits, num_trials);
    PrintStageText("pack_bits", metrics.pack_bits, num_trials);
    PrintStageText("lwe_keyswitch", metrics.lwe_keyswitch, num_trials);
    PrintStageText("modswitch_in", metrics.modswitch_in, num_trials);
    PrintStageText("accum_p", metrics.accum_p, num_trials);
    PrintStageText("accum_q", metrics.accum_q, num_trials);
    PrintStageText("expcrt", metrics.expcrt, num_trials);
    PrintStageText("fun_extract", metrics.fun_extract, num_trials);
    PrintStageText("modswitch_out", metrics.modswitch_out, num_trials);
    PrintStageText("decode_and_check", metrics.decode_and_check, num_trials);
    PrintStageText("total", metrics.total, num_trials);
}

void PrintStageJson(const char *name, const StageStats &stats, bool has_trailing_comma) {
    std::cout << "    \"" << name << "\":{"
              << "\"ms\":" << stats.ms << ","
              << "\"forward_ntt\":" << stats.forward_ntt << ","
              << "\"inverse_ntt\":" << stats.inverse_ntt << ","
              << "\"galois\":" << stats.galois << ","
              << "\"keyswitch\":" << stats.keyswitch << ","
              << "\"extmult\":" << stats.extmult << ","
              << "\"trace_calls\":" << stats.trace_calls << ","
              << "\"approx_bytes\":" << stats.approx_bytes << "}";
    if (has_trailing_comma) {
        std::cout << ",";
    }
    std::cout << std::endl;
}

template <typename Params>
void PrintJson(
    const ExperimentConfig &config,
    bool dim_reduction_enabled,
    bool dim_reduction_active,
    const std::vector<BootstrapResult> &trials,
    const BootstrapMetrics &metrics) {
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
    std::cout << "  \"profile_intent\":\"" << bdf17::ProfileIntentName(Params::kProfileIntent) << "\"," << std::endl;
    std::cout << "  \"expcrt_variant\":\"" << VariantName(config.expcrt_variant) << "\"," << std::endl;
    std::cout << "  \"lut\":\"" << config.lut_name << "\"," << std::endl;
    std::cout << "  \"seed\":" << config.seed << "," << std::endl;
    std::cout << "  \"trials\":" << config.num_trials << "," << std::endl;
    std::cout << "  \"p_degree\":" << Params::EvalP::N << "," << std::endl;
    std::cout << "  \"q_degree\":" << Params::EvalQ::N << "," << std::endl;
    std::cout << "  \"tensor_dimension\":" << Params::kTensorDimension << "," << std::endl;
    std::cout << "  \"plain_modulus\":" << Params::kPlainModulus << "," << std::endl;
    std::cout << "  \"lwe_frontend_dim\":" << Params::kLweFrontendDimension << "," << std::endl;
    std::cout << "  \"lwe_accumulator_dim\":" << Params::kLweAccumulatorDimension << "," << std::endl;
    std::cout << "  \"lwe_noise_var\":" << Params::kLweNoiseVar << "," << std::endl;
    std::cout << "  \"rlwe_noise_var\":" << Params::kRlweNoiseVar << "," << std::endl;
    std::cout << "  \"q_frontend\":" << Params::kFrontendModulus << "," << std::endl;
    std::cout << "  \"q_accumulator_input\":" << Params::kAccumulatorInputModulus << "," << std::endl;
    std::cout << "  \"q_extract_internal\":" << Params::kExtractModulus << "," << std::endl;
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
    std::cout << "  \"stage_metrics\":{" << std::endl;
    PrintStageJson("encrypt_bits", metrics.encrypt_bits, true);
    PrintStageJson("pack_bits", metrics.pack_bits, true);
    PrintStageJson("lwe_keyswitch", metrics.lwe_keyswitch, true);
    PrintStageJson("modswitch_in", metrics.modswitch_in, true);
    PrintStageJson("accum_p", metrics.accum_p, true);
    PrintStageJson("accum_q", metrics.accum_q, true);
    PrintStageJson("expcrt", metrics.expcrt, true);
    PrintStageJson("fun_extract", metrics.fun_extract, true);
    PrintStageJson("modswitch_out", metrics.modswitch_out, true);
    PrintStageJson("decode_and_check", metrics.decode_and_check, true);
    PrintStageJson("total", metrics.total, false);
    std::cout << "  }," << std::endl;
    std::cout << "  \"summary\":{\"pass\":" << pass << ",\"fail\":" << fail << "}" << std::endl;
    std::cout << "}" << std::endl;
}

template <typename Params>
int RunExperiment(const ExperimentConfig &config) {
    ValidateRunConfigOrThrow<Params>(config);
    bdf17::BootstrapRunner<Params> runner(config);
    const ExperimentConfig &resolved_config = runner.config();

    std::vector<BootstrapResult> trials;
    trials.reserve(resolved_config.num_trials);

    BootstrapMetrics aggregate_metrics;
    bool has_failure = false;
    for (size_t trial_index = 0; trial_index < resolved_config.num_trials; ++trial_index) {
        BootstrapMetrics trial_metrics;
        BootstrapResult trial = runner.RunTrial({trial_index}, &trial_metrics);
        bdf17::MergeBootstrapMetrics(aggregate_metrics, trial_metrics);
        has_failure = has_failure || !trial.ok;
        trials.push_back(std::move(trial));
    }

    if (resolved_config.emit_json) {
        PrintJson<Params>(
            resolved_config,
            runner.dim_reduction_enabled(),
            runner.dim_reduction_active(),
            trials,
            aggregate_metrics);
    } else {
        PrintConfigText<Params>(resolved_config, runner.dim_reduction_enabled(), runner.dim_reduction_active());
        PrintTrialsText(trials);
        PrintMetricsText(aggregate_metrics, trials.size());
    }

    return has_failure ? 1 : 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const ExperimentConfig config = ParseExperimentConfig(argc, argv);
        ValidateExperimentConfig(config);

        if (config.profile_name == "default" || config.profile_name == "smooth" || config.profile_name == "smooth-ntt-1153x1297") {
            return RunExperiment<bdf17::SmoothNtt1153x1297Profile>(config);
        }
        if (config.profile_name == "toy" || config.profile_name == "toy-equivalence") {
            return RunExperiment<bdf17::ToyEquivalenceProfile>(config);
        }

        throw std::runtime_error(
            "unsupported --profile value: " + config.profile_name +
            " (supported: default, smooth-ntt-1153x1297, toy-equivalence)");
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
