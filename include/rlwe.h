#ifndef RLWE_H
#define RLWE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ntt_plan.hpp"
#include "typed_poly.hpp"

template <size_t l>
class GaussianSampler {
public:
    static std::vector<int64_t> SampleE(double var, std::mt19937_64 &engine) {
        std::vector<int64_t> a(l, 0);
        if (l == 0) {
            return a;
        }
        std::uniform_int_distribution<size_t> distribution(0, l - 1);
        for (size_t i = 0; i < var * l / 2; i++) {
            a[distribution(engine)]++;
            a[distribution(engine)]--;
        }
        return a;
    }

    static std::vector<int64_t> SampleSk(double density, std::mt19937_64 &engine) {
        std::vector<int64_t> a(l, 0);
        if (l == 0) {
            return a;
        }
        std::uniform_int_distribution<size_t> distribution(0, l - 1);
        size_t count1 = density * l;
        while (count1 > 0) {
            size_t i = distribution(engine);
            if (a[i] == 0) {
                a[i] = 1;
                count1--;
            }
        }
        count1 = density * l;
        while (count1 > 0) {
            size_t i = distribution(engine);
            if (a[i] == 0) {
                a[i] = -1;
                count1--;
            }
        }
        return a;
    }
};

template <class Transform_, uint64_t B, class Plan_ = CanonicalNttPlan<Transform_>>
class SchemeImpl {
public:
    using Transform = Transform_;
    using Plan = Plan_;

    using Coeff = CoeffPoly<Transform>;
    using Eval = EvalPoly<Transform>;
    using CoeffBuffer = CoeffPolyBuffer<Transform>;
    using EvalBuffer = EvalPolyBuffer<Transform>;

    using RLWEKey = std::vector<Eval>;
    using RLWECiphertext = std::vector<Eval>;
    // Optional contiguous buffer groundwork for future ciphertext layout refactors.
    using RLWECiphertextBuffer = EvalBuffer;
    using RLWEGadgetCiphertext = std::vector<RLWECiphertext>;
    using RLWESwitchingKey = std::vector<RLWEGadgetCiphertext>;
    using RGSWCiphertext = std::pair<RLWEGadgetCiphertext, RLWEGadgetCiphertext>;

    constexpr static uint64_t Q = Eval::p;

    constexpr static size_t G = []() {
        size_t g = 0;
        for (__uint128_t t = 1; t <= Q; t *= B) {
            g++;
        }
        return g;
    }();

    constexpr static auto gadget = []() {
        std::array<uint64_t, G> g{};
        uint64_t t = 1;
        for (size_t i = 0; i < G; i++) {
            g[i] = t;
            t *= B;
        }
        return g;
    }();

    RLWEKey sk;
    Eval skp;

    std::vector<RLWESwitchingKey> ksk_galois;

    SchemeImpl();
    explicit SchemeImpl(std::vector<int64_t> skVec);

    void GaloisKeyGen(std::mt19937_64 &rng, double rlwe_noise_variance);

    template <typename T>
    static std::vector<T> GaloisConjugate(const std::vector<T> &x, const size_t &a);

    template <typename T1, typename T2>
    static std::pair<T1, T2> GaloisConjugate(const std::pair<T1, T2> &x, const size_t &a);

    static Eval GaloisConjugate(const Eval &x, const size_t &a);
    static Coeff GaloisConjugate(const Coeff &x, const size_t &a);

    static void ModSwitch(Coeff &x, uint64_t q);

    template <typename S>
    static typename S::RLWECiphertext ModSwitch(const RLWECiphertext &ct);

    RLWECiphertext RLWEEncrypt(const Eval &m, const RLWEKey &sk, uint64_t q_plain, double noise_variance, std::mt19937_64 &rng);
    RLWEGadgetCiphertext RLWEGadgetEncrypt(const Eval &m, const RLWEKey &sk, uint64_t q_plain, double noise_variance, std::mt19937_64 &rng);
    RGSWCiphertext RGSWEncrypt(const Eval &m, const RLWEKey &sk, double noise_variance, std::mt19937_64 &rng);
    Coeff RLWEDecrypt(const RLWECiphertext &ct, const RLWEKey &sk, uint64_t q_plain) const;

    static RLWECiphertext Mult(Eval a, const RLWEGadgetCiphertext &ct);
    static RLWECiphertext ExtMult(const RLWECiphertext &ct, const RGSWCiphertext &ctGSW);

    RLWESwitchingKey KeySwitchGen(const RLWEKey &sk, const RLWEKey &skN, double noise_variance, std::mt19937_64 &rng);
    static RLWECiphertext KeySwitch(const RLWECiphertext &ct, const RLWESwitchingKey &k);

    std::vector<RGSWCiphertext> BootstrappingKeyGen(std::vector<int64_t> z, double noise_variance, std::mt19937_64 &rng);
    RLWECiphertext Process(const std::vector<RGSWCiphertext> &bk, std::vector<int64_t> a, int64_t b, uint64_t q_plain);

private:
    static std::array<Coeff, G> BaseDecompose(const Coeff &a);
    static std::array<Eval, G> BaseDecomposeToEval(const Eval &a, const Plan &plan);

    Plan plan_;
};

#include "rlwe-impl.h"

#endif // RLWE_H
