#ifndef TYPED_POLY_HPP
#define TYPED_POLY_HPP

#include <algorithm>
#include <cassert>
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

template <class Transform, class DomainTag>
class ConstPolyView;

template <class Transform, class DomainTag>
class PolyView {
public:
    static constexpr uint64_t p = Transform::p;
    static constexpr size_t O = Transform::O;
    static constexpr size_t N = Transform::N;

    using TransformType = Transform;
    using Domain = DomainTag;

    explicit PolyView(uint64_t *data) noexcept : data_(data) {
        assert(data_ != nullptr);
    }

    explicit PolyView(std::span<uint64_t> data) noexcept : data_(data.data()) {
        assert(data.size() >= N);
        assert(data_ != nullptr);
    }

    size_t size() const noexcept {
        return N;
    }

    uint64_t *data() noexcept {
        return data_;
    }

    const uint64_t *data() const noexcept {
        return data_;
    }

    std::span<uint64_t, N> span() const noexcept {
        return std::span<uint64_t, N>(data_, N);
    }

    uint64_t &operator[](size_t idx) noexcept {
        return data_[idx];
    }

    const uint64_t &operator[](size_t idx) const noexcept {
        return data_[idx];
    }

    operator ConstPolyView<Transform, DomainTag>() const noexcept {
        return ConstPolyView<Transform, DomainTag>(data_);
    }

private:
    uint64_t *data_;
};

template <class Transform, class DomainTag>
class ConstPolyView {
public:
    static constexpr uint64_t p = Transform::p;
    static constexpr size_t O = Transform::O;
    static constexpr size_t N = Transform::N;

    using TransformType = Transform;
    using Domain = DomainTag;

    explicit ConstPolyView(const uint64_t *data) noexcept : data_(data) {
        assert(data_ != nullptr);
    }

    explicit ConstPolyView(std::span<const uint64_t> data) noexcept : data_(data.data()) {
        assert(data.size() >= N);
        assert(data_ != nullptr);
    }

    size_t size() const noexcept {
        return N;
    }

    const uint64_t *data() const noexcept {
        return data_;
    }

    std::span<const uint64_t, N> span() const noexcept {
        return std::span<const uint64_t, N>(data_, N);
    }

    const uint64_t &operator[](size_t idx) const noexcept {
        return data_[idx];
    }

private:
    const uint64_t *data_;
};

template <class Transform, class DomainTag>
class PolyBuffer {
public:
    static constexpr size_t N = Transform::N;

    explicit PolyBuffer(size_t poly_count) : storage_(poly_count * N, 0) {}

    size_t poly_count() const noexcept {
        return storage_.size() / N;
    }

    size_t size() const noexcept {
        return storage_.size();
    }

    std::span<uint64_t> raw_span() noexcept {
        return std::span<uint64_t>(storage_.data(), storage_.size());
    }

    std::span<const uint64_t> raw_span() const noexcept {
        return std::span<const uint64_t>(storage_.data(), storage_.size());
    }

    PolyView<Transform, DomainTag> operator[](size_t idx) noexcept {
        assert(idx < poly_count());
        return PolyView<Transform, DomainTag>(storage_.data() + idx * N);
    }

    ConstPolyView<Transform, DomainTag> operator[](size_t idx) const noexcept {
        assert(idx < poly_count());
        return ConstPolyView<Transform, DomainTag>(storage_.data() + idx * N);
    }

private:
    std::vector<uint64_t> storage_;
};

template <class Transform>
using CoeffPoly = PolyRep<Transform, CoeffTag>;

template <class Transform>
using EvalPoly = PolyRep<Transform, CanonicalEvalTag>;

template <class Transform>
using CoeffPolyView = PolyView<Transform, CoeffTag>;

template <class Transform>
using EvalPolyView = PolyView<Transform, CanonicalEvalTag>;

template <class Transform>
using ConstCoeffPolyView = ConstPolyView<Transform, CoeffTag>;

template <class Transform>
using ConstEvalPolyView = ConstPolyView<Transform, CanonicalEvalTag>;

template <class Transform>
using CoeffPolyBuffer = PolyBuffer<Transform, CoeffTag>;

template <class Transform>
using EvalPolyBuffer = PolyBuffer<Transform, CanonicalEvalTag>;

template <class Transform, class DomainTag>
class PolyRep {
public:
    static constexpr uint64_t p = Transform::p;
    static constexpr size_t O = Transform::O;
    static constexpr size_t N = Transform::N;

    using TransformType = Transform;
    using Z = Zp<p>;
    using Domain = DomainTag;

    PolyRep() : a_(std::make_unique<uint64_t[]>(N)) {
        std::fill_n(a_.get(), N, 0);
    }

    PolyRep(const PolyRep &rhs) : a_(std::make_unique<uint64_t[]>(N)) {
        std::copy_n(rhs.a_.get(), N, a_.get());
    }

    explicit PolyRep(ConstPolyView<Transform, DomainTag> rhs_view) : a_(std::make_unique<uint64_t[]>(N)) {
        std::copy_n(rhs_view.data(), N, a_.get());
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

    constexpr size_t size() const noexcept {
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

    PolyView<Transform, DomainTag> view() noexcept {
        return PolyView<Transform, DomainTag>(a_.get());
    }

    ConstPolyView<Transform, DomainTag> view() const noexcept {
        return ConstPolyView<Transform, DomainTag>(a_.get());
    }

    void Fill(uint64_t x) noexcept {
        std::fill_n(a_.get(), N, x % p);
    }

    void SetZero() noexcept {
        Fill(0);
    }

    static PolyRep Zero() {
        return PolyRep{};
    }

    template <typename T>
    requires std::signed_integral<T> && std::same_as<DomainTag, CoeffTag>
    static PolyRep FromSigned(std::span<const T> v) {
        PolyRep ret;
        const size_t bound = std::min(N, v.size());
        for (size_t i = 0; i < bound; ++i) {
            ret.a_[i] = CanonicalizeSigned(v[i]);
        }
        return ret;
    }

    template <typename T>
    requires std::unsigned_integral<T> && std::same_as<DomainTag, CoeffTag>
    static PolyRep FromUnsigned(std::span<const T> v) {
        PolyRep ret;
        const size_t bound = std::min(N, v.size());
        for (size_t i = 0; i < bound; ++i) {
            ret.a_[i] = static_cast<uint64_t>(v[i]) % p;
        }
        return ret;
    }

    template <typename T>
    requires std::signed_integral<T> && std::same_as<DomainTag, CoeffTag>
    static PolyRep FromSigned(const std::vector<T> &v) {
        return FromSigned<T>(std::span<const T>(v.data(), v.size()));
    }

    template <typename T>
    requires std::unsigned_integral<T> && std::same_as<DomainTag, CoeffTag>
    static PolyRep FromUnsigned(const std::vector<T> &v) {
        return FromUnsigned<T>(std::span<const T>(v.data(), v.size()));
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

    PolyRep &operator+=(const PolyRep &rhs) noexcept {
        for (size_t i = 0; i < N; ++i) {
            a_[i] = Z::Add(a_[i], rhs.a_[i]);
        }
        return *this;
    }

    PolyRep &operator-=(const PolyRep &rhs) noexcept {
        for (size_t i = 0; i < N; ++i) {
            a_[i] = Z::Sub(a_[i], rhs.a_[i]);
        }
        return *this;
    }

    PolyRep &operator*=(uint64_t rhs) noexcept {
        for (size_t i = 0; i < N; ++i) {
            a_[i] = Z::Mul(a_[i], rhs);
        }
        return *this;
    }

    PolyRep &operator*=(const PolyRep &rhs) noexcept
    requires std::same_as<DomainTag, CanonicalEvalTag>
    {
        for (size_t i = 0; i < N; ++i) {
            a_[i] = Z::Mul(a_[i], rhs.a_[i]);
        }
        return *this;
    }

    PolyRep operator+(const PolyRep &rhs) const {
        PolyRep ret(*this);
        ret += rhs;
        return ret;
    }

    PolyRep operator-(const PolyRep &rhs) const {
        PolyRep ret(*this);
        ret -= rhs;
        return ret;
    }

    PolyRep operator*(uint64_t rhs) const {
        PolyRep ret(*this);
        ret *= rhs;
        return ret;
    }

    friend PolyRep operator*(uint64_t lhs, const PolyRep &rhs) {
        return rhs * lhs;
    }

    PolyRep operator*(const PolyRep &rhs) const
    requires std::same_as<DomainTag, CanonicalEvalTag>
    {
        PolyRep ret(*this);
        ret *= rhs;
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
    template <typename T>
    static uint64_t CanonicalizeSigned(T x) noexcept
    requires std::signed_integral<T>
    {
        if (x >= 0) {
            using UnsignedT = std::make_unsigned_t<T>;
            return static_cast<uint64_t>(static_cast<__uint128_t>(static_cast<UnsignedT>(x)) % p);
        }

        using UnsignedT = std::make_unsigned_t<T>;
        const UnsignedT ux = static_cast<UnsignedT>(x);
        const __uint128_t mag = (static_cast<__uint128_t>(~ux) + 1) % p;
        return mag == 0 ? 0 : p - static_cast<uint64_t>(mag);
    }

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
