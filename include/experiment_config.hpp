#ifndef BDF17_EXPERIMENT_CONFIG_HPP
#define BDF17_EXPERIMENT_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

#include "expcrt_variant.hpp"

namespace bdf17 {

struct RandomContext {
    explicit RandomContext(uint64_t seed_value = 0) : engine(seed_value) {}

    std::mt19937_64 engine;
};

struct ExperimentConfig {
    uint64_t seed = 0;
    size_t num_trials = 0;
    ExpCrtVariant expcrt_variant;
    std::string profile_name;
    std::string lut_name;
    bool enable_lwe_dim_reduction_override = false;
    bool has_enable_lwe_dim_reduction_override = false;
    bool emit_json = false;
};

} // namespace bdf17

#endif // BDF17_EXPERIMENT_CONFIG_HPP
