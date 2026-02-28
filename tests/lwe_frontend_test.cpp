#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "lwe_frontend.hpp"

namespace {

constexpr uint64_t kPlainModulus = 64;
constexpr uint64_t kLweModulus = 1495441;

std::vector<int64_t> BuildDeterministicSecret(size_t n) {
    std::vector<int64_t> sk(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (i % 3 == 0) {
            sk[i] = 1;
        } else if (i % 3 == 1) {
            sk[i] = -1;
        } else {
            sk[i] = 0;
        }
    }
    return sk;
}

} // namespace

TEST(LweFrontend, EncodeDecodeRoundTripNoNoise) {
    for (uint64_t m = 0; m < kPlainModulus; ++m) {
        const uint64_t phase = bdf17::EncodeMessage(m, kPlainModulus, kLweModulus);
        EXPECT_EQ(bdf17::DecodeMessage(phase, kPlainModulus, kLweModulus), m);
    }
}

TEST(LweFrontend, EncryptDecryptDeterministic) {
    std::mt19937_64 rng(123456789ULL);
    const auto sk = BuildDeterministicSecret(32);

    for (uint64_t m = 0; m < kPlainModulus; m += 3) {
        auto ct = bdf17::EncryptLwe(sk, m, kPlainModulus, kLweModulus, 4.0, rng);
        EXPECT_EQ(bdf17::DecryptLwe(ct, sk, kPlainModulus, kLweModulus), m);
    }
}

TEST(LweFrontend, PackBitsCiphertextsLittleEndian) {
    std::mt19937_64 rng(42ULL);
    const auto sk = BuildDeterministicSecret(24);

    const std::array<uint64_t, 6> bits{1, 0, 1, 1, 0, 1}; // 45 in LE
    std::vector<bdf17::LweCiphertext> bit_ciphertexts;
    bit_ciphertexts.reserve(bits.size());

    uint64_t packed_plain = 0;
    for (size_t i = 0; i < bits.size(); ++i) {
        packed_plain |= bits[i] << i;
        bit_ciphertexts.push_back(bdf17::EncryptLwe(sk, bits[i], kPlainModulus, kLweModulus, 0.0, rng));
    }

    const auto packed_ct = bdf17::PackBitsCiphertextsLE(bit_ciphertexts, kPlainModulus, kLweModulus);
    EXPECT_EQ(bdf17::DecryptLwe(packed_ct, sk, kPlainModulus, kLweModulus), packed_plain);
}

TEST(LweFrontend, PackBitsRejectsTooManyInputs) {
    std::vector<bdf17::LweCiphertext> bit_ciphertexts(7);
    for (auto &ct : bit_ciphertexts) {
        ct.a = {0, 0, 0};
        ct.b = 0;
    }

    EXPECT_THROW(
        (void)bdf17::PackBitsCiphertextsLE(bit_ciphertexts, kPlainModulus, kLweModulus),
        std::runtime_error);
}

TEST(LweFrontend, KeySwitchReductionPreservesMessage) {
    std::mt19937_64 rng(20260228ULL);
    const auto sk_in = BuildDeterministicSecret(12);
    const auto sk_out = BuildDeterministicSecret(7);

    auto ksk = bdf17::GenerateLweKeySwitchKey(sk_in, sk_out, kLweModulus, 16, 0.0, rng);

    const std::array<uint64_t, 6> messages{0, 1, 5, 17, 33, 63};
    for (uint64_t m : messages) {
        auto ct_in = bdf17::EncryptLwe(sk_in, m, kPlainModulus, kLweModulus, 0.0, rng);
        auto ct_out = bdf17::ApplyLweKeySwitch(ct_in, ksk);
        EXPECT_EQ(ct_out.a.size(), sk_out.size());
        EXPECT_EQ(bdf17::DecryptLwe(ct_out, sk_out, kPlainModulus, kLweModulus), m);
    }
}
