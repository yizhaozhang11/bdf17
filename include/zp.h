#ifndef ZP_H
#define ZP_H

#include <cstdint>

template <uint64_t p_>
class Zp {
public:
    static constexpr uint64_t p = p_;
    static constexpr bool kScalarFastPathSupported = (p < (1ULL << 63));
    static constexpr bool kSimdFastPathSupported = (p <= (1ULL << 62));

    static_assert(kScalarFastPathSupported,
                  "Current Zp fast path requires p < 2^63. Add a generic fallback before using larger moduli.");

    struct ConstMultiplier {
        uint64_t value;
        uint64_t shoup;
    };

    static constexpr uint64_t Add(uint64_t a, uint64_t b) noexcept {
        uint64_t sum = a + b;
        return sum >= p ? sum - p : sum;
    }

    static constexpr uint64_t Sub(uint64_t a, uint64_t b) noexcept {
        return a >= b ? a - b : p - b + a;
    }

    static constexpr uint64_t Mul(uint64_t a, uint64_t b) noexcept {
        return (uint64_t)((__uint128_t)a * b % p);
    }

    static constexpr ConstMultiplier MakeConstMultiplier(uint64_t x) noexcept {
        const uint64_t value = x % p;
        return ConstMultiplier{
            value,
            static_cast<uint64_t>((static_cast<__uint128_t>(value) << 64) / p),
        };
    }

    // Preconditions:
    // - a < p
    // - c.value < p
    // - c.shoup = floor(c.value * 2^64 / p)
    // - p < 2^63
    // Then r = a*c.value - floor(a*c.shoup / 2^64)*p lies in [0, 2p),
    // so a single conditional subtract is sufficient.
    static constexpr uint64_t MulConst(uint64_t a, ConstMultiplier c) noexcept {
        const uint64_t q = static_cast<uint64_t>((static_cast<__uint128_t>(a) * c.shoup) >> 64);
        const uint64_t r = a * c.value - q * p;
        return r >= p ? r - p : r;
    }

    static constexpr uint64_t Pow(uint64_t x, uint64_t e) noexcept {
        uint64_t res = 1;
        uint64_t base = x;

        while (e > 0) {
            if (e & 1) {
                res = Mul(res, base);
            }
            base = Mul(base, base);
            e >>= 1;
        }
        return res;
    }

    // Compatibility wrappers kept during migration.
    static constexpr uint64_t MulFastConst(uint64_t a, uint64_t b, uint64_t b_mu) noexcept {
        return MulConst(a, ConstMultiplier{b, b_mu});
    }

    static constexpr uint64_t ComputeBarrettFactor(uint64_t x) noexcept {
        return MakeConstMultiplier(x).shoup;
    }
};

#endif // ZP_H
