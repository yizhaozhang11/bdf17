#ifndef RLWE_IMPL_H
#define RLWE_IMPL_H

template <class Transform, uint64_t B, class Plan>
SchemeImpl<Transform, B, Plan>::SchemeImpl()
    : sk(), skp(), plan_() {}

template <class Transform, uint64_t B, class Plan>
SchemeImpl<Transform, B, Plan>::SchemeImpl(std::vector<int64_t> skVec)
    : sk(1), skp(), plan_() {
    Coeff sk_coeff = Coeff::FromSigned(std::span<const int64_t>(skVec));
    skp = plan_.forward(sk_coeff);
    sk[0] = skp;
}

template <class Transform, uint64_t B, class Plan>
void SchemeImpl<Transform, B, Plan>::GaloisKeyGen(std::mt19937_64 &rng, double rlwe_noise_variance) {
    ksk_galois.resize(Eval::O);
    for (size_t a = 2; a < Eval::O; a++) {
        auto sk_a = GaloisConjugate(sk, a);
        auto ksk = KeySwitchGen(sk_a, sk, rlwe_noise_variance, rng);
        ksk_galois[a] = std::move(ksk);
    }
}

template <class Transform, uint64_t B, class Plan>
template <typename T>
std::vector<T> SchemeImpl<Transform, B, Plan>::GaloisConjugate(const std::vector<T> &x, const size_t &a) {
    std::vector<T> ret;
    ret.reserve(x.size());
    for (const auto &elem : x) {
        ret.push_back(GaloisConjugate(elem, a));
    }
    return ret;
}

template <class Transform, uint64_t B, class Plan>
template <typename T1, typename T2>
std::pair<T1, T2> SchemeImpl<Transform, B, Plan>::GaloisConjugate(const std::pair<T1, T2> &x, const size_t &a) {
    return {GaloisConjugate(x.first, a), GaloisConjugate(x.second, a)};
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RlweCt2 SchemeImpl<Transform, B, Plan>::GaloisConjugate(const RlweCt2 &x, const size_t &a) {
    return {GaloisConjugate(x.a, a), GaloisConjugate(x.b, a)};
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::Eval SchemeImpl<Transform, B, Plan>::GaloisConjugate(const Eval &x, const size_t &a) {
    return GaloisApply(x, a);
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::Coeff SchemeImpl<Transform, B, Plan>::GaloisConjugate(const Coeff &x, const size_t &a) {
    return GaloisApply(x, a);
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RLWECiphertext SchemeImpl<Transform, B, Plan>::FromRlweCt2(const RlweCt2 &ct) {
    RLWECiphertext out;
    out.reserve(2);
    out.push_back(ct.a);
    out.push_back(ct.b);
    return out;
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RlweCt2 SchemeImpl<Transform, B, Plan>::ToRlweCt2(const RLWECiphertext &ct) {
    if (ct.size() != 2) {
        throw std::runtime_error("expected RLWE ciphertext of size 2");
    }
    return {ct[0], ct[1]};
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RLWECiphertext SchemeImpl<Transform, B, Plan>::FromTensorCt4(const TensorCt4 &ct) {
    RLWECiphertext out;
    out.reserve(4);
    out.push_back(ct.c0);
    out.push_back(ct.c1);
    out.push_back(ct.c2);
    out.push_back(ct.c3);
    return out;
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::TensorCt4 SchemeImpl<Transform, B, Plan>::ToTensorCt4(const RLWECiphertext &ct) {
    if (ct.size() != 4) {
        throw std::runtime_error("expected tensor ciphertext of size 4");
    }
    return {ct[0], ct[1], ct[2], ct[3]};
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RLWECiphertext SchemeImpl<Transform, B, Plan>::RLWEEncrypt(
    const Eval &m,
    const RLWEKey &sk,
    uint64_t q_plain,
    double noise_variance,
    std::mt19937_64 &rng) {
    const size_t k = sk.size();
    RLWECiphertext ct;
    ct.reserve(k + 1);

    std::uniform_int_distribution<uint64_t> coeff_dist(0, Q - 1);
    Eval result;
    for (size_t i = 0; i < k; i++) {
        // BDF17-style CLWE sampling: choose a in the sum-zero subspace.
        Coeff a_coeff;
        uint64_t sum = 0;
        for (size_t j = 1; j < Eval::N; j++) {
            a_coeff[j] = coeff_dist(rng);
            sum = Coeff::Z::Add(sum, a_coeff[j]);
        }
        a_coeff[0] = Coeff::Z::Sub(0, sum);

        Eval a_eval = plan_.forward(a_coeff);
        result = result + a_eval * sk[i];
        ct.push_back(a_eval);
    }

    const auto rand = GaussianSampler<Eval::N>::SampleE(noise_variance, rng);
    Coeff e_coeff = Coeff::FromSigned(std::span<const int64_t>(rand));
    Eval e_eval = plan_.forward(e_coeff);

    ct.push_back(result + e_eval + m * (Q / q_plain));
    return ct;
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RLWEGadgetCiphertext SchemeImpl<Transform, B, Plan>::RLWEGadgetEncrypt(
    const Eval &m,
    const RLWEKey &sk,
    uint64_t q_plain,
    double noise_variance,
    std::mt19937_64 &rng) {
    RLWEGadgetCiphertext ct(G);
    for (size_t i = 0; i < G; i++) {
        ct[i] = RLWEEncrypt(m * gadget[i], sk, q_plain, noise_variance, rng);
    }
    return ct;
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RGSWCiphertext SchemeImpl<Transform, B, Plan>::RGSWEncrypt(
    const Eval &m,
    const RLWEKey &sk,
    double noise_variance,
    std::mt19937_64 &rng) {
    if (sk.size() != 1) {
        throw std::runtime_error("RGSW encryption requires a secret key of size 1");
    }
    const Eval s = sk[0];
    return std::make_pair(
        RLWEGadgetEncrypt(m * s, sk, Q, noise_variance, rng),
        RLWEGadgetEncrypt(m, sk, Q, noise_variance, rng));
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::Coeff SchemeImpl<Transform, B, Plan>::RLWEDecrypt(
    const RLWECiphertext &ct,
    const RLWEKey &sk,
    uint64_t q_plain) const {
    const size_t k = sk.size();
    Eval result = ct[k];
    for (size_t i = 0; i < k; i++) {
        result = result - ct[i] * sk[i];
    }
    Coeff coeff = plan_.inverse(result);
    ModSwitch(coeff, q_plain);
    return coeff;
}

template <class Transform, uint64_t B, class Plan>
void SchemeImpl<Transform, B, Plan>::ModSwitch(Coeff &x, uint64_t q) {
    const size_t n = Eval::N;
    const __uint128_t big_q = Eval::p;
    const __uint128_t half_q = big_q / 2;
    const __uint128_t qq = q;
    for (size_t i = 0; i < n; ++i) {
        __uint128_t xi = x[i];
        xi = (xi * qq + half_q) / big_q;
        if (xi >= qq) {
            xi -= qq;
        }
        x[i] = static_cast<uint64_t>(xi);
    }
}

template <class Transform, uint64_t B, class Plan>
template <typename S>
typename S::RLWECiphertext SchemeImpl<Transform, B, Plan>::ModSwitch(const RLWECiphertext &ct) {
    static_assert(S::Coeff::N == Coeff::N, "Cross-scheme ModSwitch requires equal ring degree");

    Plan source_plan;
    typename S::Plan target_plan;

    typename S::RLWECiphertext result;
    result.reserve(ct.size());
    for (const auto &elem : ct) {
        Coeff coeff = source_plan.inverse(elem);
        ModSwitch(coeff, S::Q);

        typename S::Coeff coeff_switched;
        for (size_t i = 0; i < S::Coeff::N; ++i) {
            coeff_switched[i] = coeff[i];
        }
        result.push_back(target_plan.forward(coeff_switched));
    }
    return result;
}

template <class Transform, uint64_t B, class Plan>
std::array<typename SchemeImpl<Transform, B, Plan>::Coeff, SchemeImpl<Transform, B, Plan>::G>
SchemeImpl<Transform, B, Plan>::BaseDecompose(const Coeff &a) {
    std::array<Coeff, G> out;
    for (size_t j = 0; j < G; ++j) {
        const uint64_t t = gadget[j];
        for (size_t l = 0; l < Eval::N; ++l) {
            out[j][l] = (a[l] / t) % B;
        }
    }
    return out;
}

template <class Transform, uint64_t B, class Plan>
std::array<typename SchemeImpl<Transform, B, Plan>::Eval, SchemeImpl<Transform, B, Plan>::G>
SchemeImpl<Transform, B, Plan>::BaseDecomposeToEval(const Eval &a, const Plan &plan) {
    const Coeff coeff = plan.inverse(a);
    const auto digits_coeff = BaseDecompose(coeff);

    std::array<Eval, G> out;
    for (size_t j = 0; j < G; ++j) {
        out[j] = plan.forward(digits_coeff[j]);
    }
    return out;
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RLWESwitchingKey SchemeImpl<Transform, B, Plan>::KeySwitchGen(
    const RLWEKey &sk,
    const RLWEKey &skN,
    double noise_variance,
    std::mt19937_64 &rng) {
    RLWESwitchingKey result;
    result.reserve(sk.size());
    for (size_t i = 0; i < sk.size(); i++) {
        result.push_back(RLWEGadgetEncrypt(sk[i], skN, Q, noise_variance, rng));
    }
    return result;
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RLWECiphertext SchemeImpl<Transform, B, Plan>::KeySwitch(
    const RLWECiphertext &ct,
    const RLWESwitchingKey &k) {
    const Plan plan;
    return KeySwitchImpl(ct, k, plan);
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RlweCt2 SchemeImpl<Transform, B, Plan>::KeySwitch(
    const RlweCt2 &ct,
    const RLWESwitchingKey &k) {
    const Plan plan;
    return KeySwitchCt2(ct, k, plan);
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RLWECiphertext SchemeImpl<Transform, B, Plan>::KeySwitchImpl(
    const RLWECiphertext &ct,
    const RLWESwitchingKey &k,
    const Plan &plan) {
    if (ct.empty()) {
        throw std::runtime_error("KeySwitch requires non-empty ciphertext");
    }
    if (k.empty() || k[0].empty() || k[0][0].empty()) {
        throw std::runtime_error("KeySwitch requires non-empty switching key");
    }

    const size_t dim_in = ct.size() - 1;
    const size_t dim_out = k[0][0].size() - 1;
    if (k.size() != dim_in) {
        throw std::runtime_error("KeySwitch input dimension mismatch");
    }

    RLWECiphertext result(dim_out + 1);
    result[dim_out] = ct[dim_in];

    for (size_t i = 0; i < dim_in; i++) {
        const auto digits = BaseDecomposeToEval(ct[i], plan);
        for (size_t j = 0; j < G; j++) {
            for (size_t l = 0; l <= dim_out; l++) {
                result[l] = result[l] - digits[j] * k[i][j][l];
            }
        }
    }
    return result;
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RlweCt2 SchemeImpl<Transform, B, Plan>::KeySwitchCt2(
    const RlweCt2 &ct,
    const RLWESwitchingKey &k,
    const Plan &plan) {
    const auto switched = KeySwitchImpl(FromRlweCt2(ct), k, plan);
    return ToRlweCt2(switched);
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RLWECiphertext SchemeImpl<Transform, B, Plan>::Mult(
    Eval a,
    const RLWEGadgetCiphertext &ct) {
    const Plan plan;
    return MultImpl(std::move(a), ct, plan);
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RLWECiphertext SchemeImpl<Transform, B, Plan>::MultImpl(
    Eval a,
    const RLWEGadgetCiphertext &ct,
    const Plan &plan) {
    const size_t dim_out = ct[0].size() - 1;

    const auto digits = BaseDecomposeToEval(a, plan);

    RLWECiphertext result(dim_out + 1);
    for (size_t i = 0; i <= dim_out; i++) {
        Eval c;
        for (size_t j = 0; j < G; j++) {
            c = c + digits[j] * ct[j][i];
        }
        result[i] = c;
    }
    return result;
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RlweCt2 SchemeImpl<Transform, B, Plan>::ExtMult(
    const RlweCt2 &ct,
    const RGSWCiphertext &ctGSW) {
    const Plan plan;
    return ExtMultImpl(ct, ctGSW, plan);
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RLWECiphertext SchemeImpl<Transform, B, Plan>::ExtMult(
    const RLWECiphertext &ct,
    const RGSWCiphertext &ctGSW) {
    return FromRlweCt2(ExtMult(ToRlweCt2(ct), ctGSW));
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RlweCt2 SchemeImpl<Transform, B, Plan>::ExtMultImpl(
    const RlweCt2 &ct,
    const RGSWCiphertext &ctGSW,
    const Plan &plan) {
    Eval ra;
    Eval rb;

    Eval a = ct.a;
    Eval b = ct.b;
    for (size_t i = 0; i < Eval::N; i++) {
        a[i] = Eval::Z::Sub(0, a[i]);
    }

    const auto a_digits = BaseDecomposeToEval(a, plan);
    const auto b_digits = BaseDecomposeToEval(b, plan);

    for (size_t i = 0; i < G; i++) {
        ra = ra + b_digits[i] * ctGSW.second[i][0] + a_digits[i] * ctGSW.first[i][0];
        rb = rb + b_digits[i] * ctGSW.second[i][1] + a_digits[i] * ctGSW.first[i][1];
    }
    return RlweCt2{ra, rb};
}

template <class Transform, uint64_t B, class Plan>
std::vector<typename SchemeImpl<Transform, B, Plan>::RGSWCiphertext> SchemeImpl<Transform, B, Plan>::BootstrappingKeyGen(
    std::vector<int64_t> z,
    double noise_variance,
    std::mt19937_64 &rng) {
    std::vector<RGSWCiphertext> result;
    result.reserve(z.size());
    for (size_t i = 0; i < z.size(); i++) {
        int64_t idx = z[i] % static_cast<int64_t>(Eval::O);
        if (idx < 0) {
            idx += static_cast<int64_t>(Eval::O);
        }
        Coeff m_coeff = Coeff::Monomial(static_cast<size_t>(idx), 1);
        Eval m_eval = plan_.forward(m_coeff);
        result.push_back(RGSWEncrypt(m_eval, sk, noise_variance, rng));
    }
    return result;
}

template <class Transform, uint64_t B, class Plan>
typename SchemeImpl<Transform, B, Plan>::RLWECiphertext SchemeImpl<Transform, B, Plan>::Process(
    const std::vector<RGSWCiphertext> &bk,
    std::vector<int64_t> a,
    int64_t b,
    uint64_t q_plain) {
    Eval ca;
    Coeff cb_coeff;

    if (b < 0) {
        b = -b;
        b = b % static_cast<int64_t>(Eval::O);
        b = static_cast<int64_t>(Eval::O) - b;
    }
    b = b % static_cast<int64_t>(Eval::O);
    cb_coeff[static_cast<size_t>(b)] = Q / q_plain;

    Eval cb = plan_.forward(cb_coeff);
    RlweCt2 ct{ca, cb};

    uint64_t t = 1;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] < 0) {
            a[i] = -a[i];
            a[i] = a[i] % static_cast<int64_t>(Eval::O);
            a[i] = static_cast<int64_t>(Eval::O) - a[i];
        }
        a[i] = a[i] % static_cast<int64_t>(Eval::O);
        a[i] = static_cast<int64_t>(Eval::O) - a[i];
        if (a[i] != static_cast<int64_t>(Eval::O)) {
            t = Zp<Eval::O>::Mul(t, Zp<Eval::O>::Pow(static_cast<uint64_t>(a[i]), Eval::O - 2));
            if (t != 1) {
                ct = GaloisConjugate(ct, t);
                ct = KeySwitchCt2(ct, ksk_galois[t], plan_);
            }
            ct = ExtMultImpl(ct, bk[i], plan_);
            t = static_cast<uint64_t>(a[i]);
        }
    }
    if (t != 1) {
        ct = GaloisConjugate(ct, t);
        ct = KeySwitchCt2(ct, ksk_galois[t], plan_);
    }
    return FromRlweCt2(ct);
}

#endif // RLWE_IMPL_H
