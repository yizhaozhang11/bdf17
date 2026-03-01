#ifndef BDF17_LWE_FRONTEND_HPP
#define BDF17_LWE_FRONTEND_HPP

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace bdf17 {

struct LweCiphertext {
    std::vector<uint64_t> a;
    uint64_t b = 0;
};

struct LweKeySwitchKey {
    uint64_t modulus = 0;
    uint64_t base = 0;
    size_t digits = 0;
    size_t n_in = 0;
    size_t n_out = 0;
    std::vector<LweCiphertext> data;
};

inline uint64_t ModAdd(uint64_t lhs, uint64_t rhs, uint64_t mod) {
    if (mod == 0) {
        throw std::runtime_error("modulus must be non-zero");
    }
    const __uint128_t sum = (__uint128_t)lhs + rhs;
    return (uint64_t)(sum % mod);
}

inline uint64_t ModSub(uint64_t lhs, uint64_t rhs, uint64_t mod) {
    if (mod == 0) {
        throw std::runtime_error("modulus must be non-zero");
    }
    lhs %= mod;
    rhs %= mod;
    return lhs >= rhs ? lhs - rhs : mod - (rhs - lhs);
}

inline uint64_t ModMul(uint64_t lhs, uint64_t rhs, uint64_t mod) {
    if (mod == 0) {
        throw std::runtime_error("modulus must be non-zero");
    }
    return (uint64_t)((__uint128_t)lhs * rhs % mod);
}

inline uint64_t ModFromSigned(int64_t value, uint64_t mod) {
    if (mod == 0) {
        throw std::runtime_error("modulus must be non-zero");
    }
    if (value >= 0) {
        return (uint64_t)value % mod;
    }

    const uint64_t abs_value = (uint64_t)(-(value + 1)) + 1;
    const uint64_t rem = abs_value % mod;
    return rem == 0 ? 0 : mod - rem;
}

inline uint64_t ModMulSigned(uint64_t lhs, int64_t rhs, uint64_t mod) {
    if (rhs >= 0) {
        return ModMul(lhs, (uint64_t)rhs, mod);
    }
    const uint64_t t = ModMul(lhs, (uint64_t)(-rhs), mod);
    return t == 0 ? 0 : mod - t;
}

inline size_t MaxPackingBits(uint64_t plain_modulus) {
    if (plain_modulus < 2) {
        throw std::runtime_error("plain modulus must be >= 2");
    }
    size_t bits = 0;
    for (uint64_t value = plain_modulus; value > 1; value >>= 1) {
        ++bits;
    }
    return bits;
}

inline uint64_t EncodeMessage(uint64_t message, uint64_t plain_modulus, uint64_t mod) {
    if (plain_modulus == 0 || mod == 0) {
        throw std::runtime_error("modulus must be non-zero");
    }
    const uint64_t delta = (uint64_t)(((__uint128_t)mod + plain_modulus / 2) / plain_modulus);
    const uint64_t m = message % plain_modulus;
    return ModMul(m, delta, mod);
}

inline uint64_t DecodeMessage(uint64_t phase, uint64_t plain_modulus, uint64_t mod) {
    if (plain_modulus == 0 || mod == 0) {
        throw std::runtime_error("modulus must be non-zero");
    }
    const __uint128_t scaled = (__uint128_t)phase * plain_modulus + mod / 2;
    return (uint64_t)((scaled / mod) % plain_modulus);
}

inline int64_t SampleNoise(std::mt19937_64 &rng, double variance) {
    if (variance <= 0.0) {
        return 0;
    }

    const uint64_t trials = (uint64_t)(2.0 * variance + 0.5);
    std::uniform_int_distribution<uint64_t> bit_dist(0, 1);

    int64_t noise = 0;
    for (uint64_t i = 0; i < trials; ++i) {
        noise += (int64_t)bit_dist(rng);
        noise -= (int64_t)bit_dist(rng);
    }
    return noise;
}

inline LweCiphertext EncryptPhase(
    const std::vector<int64_t> &sk,
    uint64_t phase,
    uint64_t mod,
    double noise_variance,
    std::mt19937_64 &rng) {
    if (sk.empty()) {
        throw std::runtime_error("secret key must be non-empty");
    }
    if (mod <= 1) {
        throw std::runtime_error("modulus must be > 1");
    }

    LweCiphertext ct;
    ct.a.resize(sk.size(), 0);

    std::uniform_int_distribution<uint64_t> a_dist(0, mod - 1);
    uint64_t inner = 0;
    for (size_t i = 0; i < sk.size(); ++i) {
        ct.a[i] = a_dist(rng);
        inner = ModAdd(inner, ModMulSigned(ct.a[i], sk[i], mod), mod);
    }

    const uint64_t noise = ModFromSigned(SampleNoise(rng, noise_variance), mod);
    ct.b = ModAdd(ModAdd(phase % mod, inner, mod), noise, mod);
    return ct;
}

inline LweCiphertext EncryptLwe(
    const std::vector<int64_t> &sk,
    uint64_t message,
    uint64_t plain_modulus,
    uint64_t mod,
    double noise_variance,
    std::mt19937_64 &rng) {
    return EncryptPhase(sk, EncodeMessage(message, plain_modulus, mod), mod, noise_variance, rng);
}

inline uint64_t DecryptPhase(const LweCiphertext &ct, const std::vector<int64_t> &sk, uint64_t mod) {
    if (mod == 0) {
        throw std::runtime_error("modulus must be non-zero");
    }
    if (ct.a.size() != sk.size()) {
        throw std::runtime_error("ciphertext/secret dimension mismatch");
    }

    uint64_t phase = ct.b % mod;
    for (size_t i = 0; i < sk.size(); ++i) {
        phase = ModSub(phase, ModMulSigned(ct.a[i], sk[i], mod), mod);
    }
    return phase;
}

inline uint64_t DecryptLwe(const LweCiphertext &ct, const std::vector<int64_t> &sk, uint64_t plain_modulus, uint64_t mod) {
    return DecodeMessage(DecryptPhase(ct, sk, mod), plain_modulus, mod);
}

[[nodiscard]] inline uint64_t ModSwitchScalar(uint64_t value, uint64_t q_from, uint64_t q_to) {
    if (q_from == 0 || q_to == 0) {
        throw std::runtime_error("modulus must be non-zero");
    }

    const __uint128_t scaled = (__uint128_t)(value % q_from) * q_to + q_from / 2;
    uint64_t out = (uint64_t)(scaled / q_from);
    if (out >= q_to) {
        out -= q_to;
    }
    return out;
}

[[nodiscard]] inline LweCiphertext ModSwitchLwe(const LweCiphertext &ct, uint64_t q_from, uint64_t q_to, uint64_t plain_modulus) {
    if (plain_modulus == 0) {
        throw std::runtime_error("plain modulus must be non-zero");
    }

    LweCiphertext out;
    out.a.resize(ct.a.size(), 0);
    for (size_t i = 0; i < ct.a.size(); ++i) {
        out.a[i] = ModSwitchScalar(ct.a[i], q_from, q_to);
    }
    out.b = ModSwitchScalar(ct.b, q_from, q_to);
    return out;
}

inline LweCiphertext PackBitsCiphertextsLE(const std::vector<LweCiphertext> &bit_ciphertexts, uint64_t plain_modulus, uint64_t mod) {
    if (bit_ciphertexts.empty()) {
        throw std::runtime_error("bit ciphertext list must be non-empty");
    }

    const size_t max_bits = MaxPackingBits(plain_modulus);
    if (bit_ciphertexts.size() > max_bits) {
        throw std::runtime_error("too many bits for plaintext modulus");
    }

    const size_t dimension = bit_ciphertexts[0].a.size();
    if (dimension == 0) {
        throw std::runtime_error("ciphertext dimension must be non-zero");
    }

    LweCiphertext packed;
    packed.a.assign(dimension, 0);
    packed.b = 0;

    for (size_t bit_index = 0; bit_index < bit_ciphertexts.size(); ++bit_index) {
        const auto &ct = bit_ciphertexts[bit_index];
        if (ct.a.size() != dimension) {
            throw std::runtime_error("ciphertext dimension mismatch");
        }

        const uint64_t scale = 1ULL << bit_index;
        for (size_t i = 0; i < dimension; ++i) {
            packed.a[i] = ModAdd(packed.a[i], ModMul(ct.a[i], scale, mod), mod);
        }
        packed.b = ModAdd(packed.b, ModMul(ct.b, scale, mod), mod);
    }

    return packed;
}

inline size_t ComputeBaseDigits(uint64_t mod, uint64_t base) {
    if (mod <= 1) {
        throw std::runtime_error("modulus must be > 1");
    }
    if (base < 2) {
        throw std::runtime_error("base must be >= 2");
    }

    size_t digits = 0;
    __uint128_t value = 1;
    while (value < mod) {
        value *= base;
        ++digits;
    }
    return digits;
}

inline LweKeySwitchKey GenerateLweKeySwitchKey(
    const std::vector<int64_t> &sk_in,
    const std::vector<int64_t> &sk_out,
    uint64_t mod,
    uint64_t base,
    double noise_variance,
    std::mt19937_64 &rng) {
    if (sk_in.empty() || sk_out.empty()) {
        throw std::runtime_error("secret keys must be non-empty");
    }

    const size_t digits = ComputeBaseDigits(mod, base);

    LweKeySwitchKey ksk;
    ksk.modulus = mod;
    ksk.base = base;
    ksk.digits = digits;
    ksk.n_in = sk_in.size();
    ksk.n_out = sk_out.size();
    ksk.data.reserve(ksk.n_in * ksk.digits);

    std::vector<uint64_t> base_powers(digits, 1);
    for (size_t j = 1; j < digits; ++j) {
        base_powers[j] = ModMul(base_powers[j - 1], base, mod);
    }

    for (size_t i = 0; i < sk_in.size(); ++i) {
        for (size_t j = 0; j < digits; ++j) {
            const uint64_t phase = ModMulSigned(base_powers[j], sk_in[i], mod);
            ksk.data.push_back(EncryptPhase(sk_out, phase, mod, noise_variance, rng));
        }
    }

    return ksk;
}

[[nodiscard]] inline LweCiphertext ApplyLweKeySwitch(const LweCiphertext &ct_in, const LweKeySwitchKey &ksk) {
    if (ct_in.a.size() != ksk.n_in) {
        throw std::runtime_error("ciphertext/key-switch input dimension mismatch");
    }
    if (ksk.base < 2 || ksk.digits == 0) {
        throw std::runtime_error("invalid key-switch parameters");
    }
    if (ksk.data.size() != ksk.n_in * ksk.digits) {
        throw std::runtime_error("invalid key-switch key data size");
    }

    LweCiphertext out;
    out.a.assign(ksk.n_out, 0);
    out.b = ct_in.b % ksk.modulus;

    for (size_t i = 0; i < ksk.n_in; ++i) {
        uint64_t coeff = ct_in.a[i] % ksk.modulus;
        for (size_t j = 0; j < ksk.digits; ++j) {
            const uint64_t digit = coeff % ksk.base;
            coeff /= ksk.base;
            if (digit == 0) {
                continue;
            }

            const auto &entry = ksk.data[i * ksk.digits + j];
            if (entry.a.size() != ksk.n_out) {
                throw std::runtime_error("invalid key-switch key entry dimension");
            }
            for (size_t k = 0; k < ksk.n_out; ++k) {
                out.a[k] = ModSub(out.a[k], ModMul(entry.a[k], digit, ksk.modulus), ksk.modulus);
            }
            out.b = ModSub(out.b, ModMul(entry.b, digit, ksk.modulus), ksk.modulus);
        }
    }

    return out;
}

} // namespace bdf17

#endif // BDF17_LWE_FRONTEND_HPP
