#ifndef RLWE_IMPL_H
#define RLWE_IMPL_H

template <typename Poly, uint64_t B>
SchemeImpl<Poly, B>::SchemeImpl() : sk(), skp(), engine(std::random_device{}()), distribution(0, Q - 1) {}

template <typename Poly, uint64_t B>
SchemeImpl<Poly, B>::SchemeImpl(std::vector<int64_t> skVec) : sk(1), skp(true), engine(std::random_device{}()), distribution(0, Q - 1) {
    skp = Poly::FromCoeff(skVec);
    skp.ToNTT();
    sk[0] = skp;
}

template <typename Poly, uint64_t B>
void SchemeImpl<Poly, B>::GaloisKeyGen() {
    ksk_galois.resize(Poly::O);
    skp.ToNTT();
    for (size_t a = 2; a < Poly::O; a++) {
        auto sk_a = GaloisConjugate(sk, a);
        auto ksk = KeySwitchGen(sk_a, sk);
        ksk_galois[a] = ksk;
    }
}

template <typename Poly, uint64_t B>
template <typename T>
std::vector<T> SchemeImpl<Poly, B>::GaloisConjugate(const std::vector<T> &x, const size_t &a) {
    std::vector<T> ret;
    ret.reserve(x.size());
    for (const auto &elem : x) {
        ret.push_back(Poly::GaloisConjugate(elem, a));
    }
    return ret;
}

template <typename Poly, uint64_t B>
template <typename T1, typename T2>
std::pair<T1, T2> SchemeImpl<Poly, B>::GaloisConjugate(const std::pair<T1, T2> &x, const size_t &a) {
    return {Poly::GaloisConjugate(x.first, a), Poly::GaloisConjugate(x.second, a)};
}

template <typename Poly, uint64_t B>
typename SchemeImpl<Poly, B>::RLWECiphertext SchemeImpl<Poly, B>::RLWEEncrypt(const Poly &m, const RLWEKey &sk, uint64_t q_plain) {
    size_t k = sk.size();
    RLWECiphertext ct;

    Poly result(false);
    for (size_t i = 0; i < k; i++) {
        // BDF17-style CLWE sampling: choose a in the sum-zero subspace.
        Poly a(true);
        uint64_t sum = 0;
        for (size_t j = 1; j < Poly::N; j++) {
            a.a[j] = distribution(engine);
            sum = Poly::Z::Add(sum, a.a[j]);
        }
        a.a[0] = Poly::Z::Sub(0, sum);
        a.ToNTT();
        result = result + a * sk[i];
        ct.push_back(a);
    }
    auto rand = GaussianSampler<Poly::N>::GetInstance().SampleE(4.0);
    Poly e = Poly::FromCoeff(rand);
    e.ToNTT();
    ct.push_back(result + e + m * (Q / q_plain));
    return ct;
}

template <typename Poly, uint64_t B>
typename SchemeImpl<Poly, B>::RLWEGadgetCiphertext SchemeImpl<Poly, B>::RLWEGadgetEncrypt(const Poly &m, const RLWEKey &sk, uint64_t q_plain) {
    RLWEGadgetCiphertext ct(G);

    for (size_t i = 0; i < G; i++) {
        ct[i] = RLWEEncrypt(m * gadget[i], sk, q_plain);
    }
    return ct;
}

template <typename Poly, uint64_t B>
typename SchemeImpl<Poly, B>::RGSWCiphertext SchemeImpl<Poly, B>::RGSWEncrypt(const Poly &m, const typename SchemeImpl<Poly, B>::RLWEKey &sk) {
    if (sk.size() != 1) {
        throw std::runtime_error("RGSW encryption requires a secret key of size 1");
    }
    Poly s = sk[0];
    return std::make_pair(RLWEGadgetEncrypt(m * s, sk, Q), RLWEGadgetEncrypt(m, sk, Q));
}

template <typename Poly, uint64_t B>
Poly SchemeImpl<Poly, B>::RLWEDecrypt(const RLWECiphertext &ct, const RLWEKey &sk, uint64_t q_plain) {
    size_t k = sk.size();
    Poly result = ct[k];
    for (size_t i = 0; i < k; i++) {
        result = result - ct[i] * sk[i];
    }
    result.ToCoeff();
    ModSwitch(result, q_plain);
    return result;
}

template <typename Poly, uint64_t B>
void SchemeImpl<Poly, B>::ModSwitch(Poly &x, uint64_t q) {
    size_t n = Poly::N;
    __uint128_t Q = Poly::p;
    __uint128_t halfQ = Q / 2;
    __uint128_t qq = q;
    for (size_t i = 0; i < n; ++i) {
        __uint128_t xi = x.a[i];
        xi = (xi * qq + halfQ) / Q;
        if (xi >= qq) {
            xi -= qq;
        }
        x.a[i] = (uint64_t)xi;
    }
}

template <typename Poly, uint64_t B> template <typename S>
typename S::RLWECiphertext SchemeImpl<Poly, B>::ModSwitch(const RLWECiphertext &ct) {
    auto q = S::Q;
    typename S::RLWECiphertext result;
    for (auto elem : ct) {
        elem.ToCoeff();
        ModSwitch(elem, q);
        typename S::Poly tmp;
        for (size_t i = 0; i < S::Poly::N; i++) {
            tmp.a[i] = elem.a[i];
        }
        tmp.ToNTT();
        result.push_back(tmp);
    }
    return result;
}

template <typename Poly, uint64_t B>
typename SchemeImpl<Poly, B>::RLWESwitchingKey SchemeImpl<Poly, B>::KeySwitchGen(const RLWEKey &sk, const RLWEKey &skN) {
    RLWESwitchingKey result;
    for (size_t i = 0; i < sk.size(); i++) {
        result.push_back(RLWEGadgetEncrypt(sk[i], skN, Q));
    }
    return result;
}

template <typename Poly, uint64_t B>
typename SchemeImpl<Poly, B>::RLWECiphertext SchemeImpl<Poly, B>::KeySwitch(const RLWECiphertext &ct, const RLWESwitchingKey &K) {
    size_t k = ct.size() - 1;
    size_t kN = K[0][0].size() - 1;

    RLWECiphertext result(kN + 1);
    for (size_t i = 0; i <= kN; i++) {
        result[i].is_coeff = false;
    }
    result[kN] = ct[k];

    for (size_t i = 0; i < k; i++) {
        auto a = ct[i];
        a.ToCoeff();
        for (size_t j = 0; j < G; j++) {
            uint64_t t = gadget[j];
            Poly a0(true);
            for (size_t l = 0; l < Poly::N; l++) {
                a0.a[l] = a.a[l] / t % B;
            }
            a0.ToNTT();
            for (size_t l = 0; l <= kN; l++) {
                result[l] = result[l] - a0 * K[i][j][l];
            }
        }
    }
    return result;
}

template <typename Poly, uint64_t B>
typename SchemeImpl<Poly, B>::RLWECiphertext SchemeImpl<Poly, B>::Mult(Poly a, RLWEGadgetCiphertext ct) {
    size_t kN = ct[0].size() - 1;
    a.ToCoeff();
    RLWECiphertext result;
    for (size_t i = 0; i <= kN; i++) {
        Poly c(false);
        for (size_t j = 0; j < G; j++) {
            uint64_t t = gadget[j];
            Poly a0(true);
            for (size_t l = 0; l < Poly::N; l++) {
                a0.a[l] = a.a[l] / t % B;
            }
            a0.ToNTT();
            c = c + a0 * ct[j][i];
        }
        result.push_back(c);
    }
    return result;
}

template <typename Poly, uint64_t B>
typename SchemeImpl<Poly, B>::RLWECiphertext SchemeImpl<Poly, B>::ExtMult(const RLWECiphertext &ct, const RGSWCiphertext &ctGSW) {
    if (ct.size() != 2) {
        throw std::runtime_error("RGSW multiplication requires a ciphertext of size 2");
    }

    Poly ra(false);
    Poly rb(false);

    Poly a = ct[0];
    Poly b = ct[1];

    for (size_t i = 0; i < Poly::N; i++) {
        a.a[i] = Poly::Z::Sub(0, a.a[i]);
    }

    a.ToCoeff();
    b.ToCoeff();

    for (size_t i = 0; i < G; i++) {
        uint64_t t = gadget[i];
        Poly a0(true);
        Poly b0(true);
        for (size_t l = 0; l < Poly::N; l++) {
            a0.a[l] = a.a[l] / t % B;
            b0.a[l] = b.a[l] / t % B;
        }
        a0.ToNTT();
        b0.ToNTT();
        ra = ra + b0 * ctGSW.second[i][0] + a0 * ctGSW.first[i][0];
        rb = rb + b0 * ctGSW.second[i][1] + a0 * ctGSW.first[i][1];
    }
    return {ra, rb};
}

template <typename Poly, uint64_t B>
std::vector<typename SchemeImpl<Poly, B>::RGSWCiphertext> SchemeImpl<Poly, B>::BootstrappingKeyGen(std::vector<int64_t> z) {
    std::vector<RGSWCiphertext> result;
    for (size_t i = 0; i < z.size(); i++) {
        Poly m(true);
        m.a[(z[i] + Poly::O) % Poly::O] = 1;
        m.ToNTT();
        auto ct = RGSWEncrypt(m, sk);
        result.push_back(ct);
    }
    return result;
}

template <typename Poly, uint64_t B>
typename SchemeImpl<Poly, B>::RLWECiphertext SchemeImpl<Poly, B>::Process(const std::vector<RGSWCiphertext> &BK, std::vector<int64_t> a, int64_t b, uint64_t q_plain) {
    Poly ca(false);
    Poly cb(true);

    if (b < 0) {
        b = -b;
        b = b % Poly::O;
        b = Poly::O - b;
    }
    b = b % Poly::O;
    cb.a[b] = Q / q_plain;

    cb.ToNTT();
    RLWECiphertext ct{ca, cb};

    uint64_t t = 1;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] < 0) {
            a[i] = -a[i];
            a[i] = a[i] % Poly::O;
            a[i] = Poly::O - a[i];
        }
        a[i] = a[i] % Poly::O;
        a[i] = Poly::O - a[i];
        if (a[i] != Poly::O) {
            t = Zp<Poly::O>::Mul(t, Zp<Poly::O>::Pow(a[i], Poly::O - 2));
            if (t != 1) {
                ct = GaloisConjugate(ct, t);
                ct = KeySwitch(ct, ksk_galois[t]);
            }
            ct = ExtMult(ct, BK[i]);
            t = a[i];
        }
    }
    if (t != 1) {
        ct = GaloisConjugate(ct, t);
        ct = KeySwitch(ct, ksk_galois[t]);
    }
    return ct;
}

#endif // RLWE_IMPL_H
