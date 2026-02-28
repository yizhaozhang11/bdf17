#ifndef TYPED_POLY_HPP
#define TYPED_POLY_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

#include "ntt.h"

struct CoeffTag {};
struct CanonicalEvalTag {};

template <class Transform, class DomainTag>
class PolyRep;

template <class Transform>
using CoeffPoly = PolyRep<Transform, CoeffTag>;

template <class Transform>
using EvalPoly = PolyRep<Transform, CanonicalEvalTag>;

template <class Transform, class DomainTag>
class PolyRep {
public:
    static constexpr uint64_t p = Transform::p;
    static constexpr size_t O = Transform::O;
    static constexpr size_t N = Transform::N;

    using Z = Zp<p>;
    using Domain = DomainTag;

    PolyRep() : a_(std::make_unique<uint64_t[]>(N)) {
        std::fill_n(a_.get(), N, 0);
    }

    PolyRep(const PolyRep &rhs) : a_(std::make_unique<uint64_t[]>(N)) {
        std::copy_n(rhs.a_.get(), N, a_.get());
    }

    PolyRep &operator=(const PolyRep &rhs) {
        if (this == &rhs) {
            return *this;
        }
        if (!a_) {
            a_ = std::make_unique<uint64_t[]>(N);
        }
        std::copy_n(rhs.a_.get(), N, a_.get());
        return *this;
    }

    PolyRep(PolyRep &&) noexcept = default;
    PolyRep &operator=(PolyRep &&) noexcept = default;

    constexpr size_t size() const {
        return N;
    }

    uint64_t *data() {
        return a_.get();
    }

    const uint64_t *data() const {
        return a_.get();
    }

    std::span<uint64_t, N> span() {
        return std::span<uint64_t, N>(a_.get(), N);
    }

    std::span<const uint64_t, N> span() const {
        return std::span<const uint64_t, N>(a_.get(), N);
    }

    uint64_t &operator[](size_t idx) {
        return a_[idx];
    }

    const uint64_t &operator[](size_t idx) const {
        return a_[idx];
    }

    template <typename T>
    requires std::signed_integral<T> && std::same_as<DomainTag, CoeffTag>
    static PolyRep FromSigned(const std::vector<T> &v) {
        PolyRep ret;
        constexpr int64_t mod = static_cast<int64_t>(p);
        const size_t bound = std::min(N, v.size());
        for (size_t i = 0; i < bound; ++i) {
            int64_t residue = static_cast<int64_t>(v[i]) % mod;
            if (residue < 0) {
                residue += mod;
            }
            ret.a_[i] = static_cast<uint64_t>(residue);
        }
        return ret;
    }

    template <typename T>
    requires std::unsigned_integral<T> && std::same_as<DomainTag, CoeffTag>
    static PolyRep FromUnsigned(const std::vector<T> &v) {
        PolyRep ret;
        const size_t bound = std::min(N, v.size());
        for (size_t i = 0; i < bound; ++i) {
            ret.a_[i] = static_cast<uint64_t>(v[i]) % p;
        }
        return ret;
    }

    static PolyRep Monomial(size_t index, uint64_t value = 1)
    requires std::same_as<DomainTag, CoeffTag>
    {
        PolyRep ret;
        if constexpr (N > 0) {
            ret.a_[index % N] = value % p;
        }
        return ret;
    }

    PolyRep operator+(const PolyRep &rhs) const {
        PolyRep ret;
        for (size_t i = 0; i < N; ++i) {
            ret.a_[i] = Z::Add(a_[i], rhs.a_[i]);
        }
        return ret;
    }

    PolyRep operator-(const PolyRep &rhs) const {
        PolyRep ret;
        for (size_t i = 0; i < N; ++i) {
            ret.a_[i] = Z::Sub(a_[i], rhs.a_[i]);
        }
        return ret;
    }

    PolyRep operator*(uint64_t rhs) const {
        PolyRep ret;
        for (size_t i = 0; i < N; ++i) {
            ret.a_[i] = Z::Mul(a_[i], rhs);
        }
        return ret;
    }

    friend PolyRep operator*(uint64_t lhs, const PolyRep &rhs) {
        return rhs * lhs;
    }

    PolyRep operator*(const PolyRep &rhs) const
    requires std::same_as<DomainTag, CanonicalEvalTag>
    {
        PolyRep ret;
        for (size_t i = 0; i < N; ++i) {
            ret.a_[i] = Z::Mul(a_[i], rhs.a_[i]);
        }
        return ret;
    }

    bool operator==(const PolyRep &rhs) const {
        for (size_t i = 0; i < N; ++i) {
            if (a_[i] != rhs.a_[i]) {
                return false;
            }
        }
        return true;
    }

private:
    std::unique_ptr<uint64_t[]> a_;
};

template <class Transform>
CoeffPoly<Transform> GaloisApply(const CoeffPoly<Transform> &x, const size_t a) {
    static_assert(Transform::N == Transform::O, "Coeff-domain Galois action requires ring transforms with N == O");

    CoeffPoly<Transform> ret;
    ret[0] = x[0];
    for (size_t i = 1; i < Transform::N; ++i) {
        ret[i * a % Transform::O] = x[i];
    }
    return ret;
}

template <class Transform>
EvalPoly<Transform> GaloisApply(const EvalPoly<Transform> &x, const size_t a) {
    static_assert(Transform::N == Transform::O, "Eval-domain Galois action requires ring transforms with N == O");

    EvalPoly<Transform> ret;
    ret[0] = x[0];
    for (size_t i = 1; i < Transform::N; ++i) {
        ret[i] = x[i * a % Transform::O];
    }
    return ret;
}

#endif // TYPED_POLY_HPP
