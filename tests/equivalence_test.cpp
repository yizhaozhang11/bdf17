#include <algorithm>
#include <cstdint>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "expcrt.hpp"
#include "fun_extract.hpp"

#include "equivalence_ref.hpp"

namespace {

using bdf17::equiv_ref::CoeffTensorRef;
using bdf17::equiv_ref::Convolution1DRef;
using bdf17::equiv_ref::Convolution2DRef;
using bdf17::equiv_ref::BuildPaperBootstrapFunctionRef;
using bdf17::equiv_ref::CrtFoldIndex;
using bdf17::equiv_ref::CrtTraceTransportedRef;
using bdf17::equiv_ref::DecodeMonomialPhaseRef;
using bdf17::equiv_ref::FirstMismatch;
using bdf17::equiv_ref::FoldCrtRef;
using bdf17::equiv_ref::FoldPaperRef;
using bdf17::equiv_ref::GaloisCtRef;
using bdf17::equiv_ref::GaloisPairRef;
using bdf17::equiv_ref::GaloisPRef;
using bdf17::equiv_ref::InvMod;
using bdf17::equiv_ref::MakeFoldedMonomial;
using bdf17::equiv_ref::MakeMonomialP;
using bdf17::equiv_ref::MakeMonomialQ;
using bdf17::equiv_ref::MakePairMonomial;
using bdf17::equiv_ref::MakeNoiselessRlweCtRef;
using bdf17::equiv_ref::MulMod;
using bdf17::equiv_ref::PaperExpCrtRef;
using bdf17::equiv_ref::PaperFoldIndex;
using bdf17::equiv_ref::PaperLutPolyRef;
using bdf17::equiv_ref::PaperTraceCoeffRef;
using bdf17::equiv_ref::PhaseRef;
using bdf17::equiv_ref::RlweCiphertextRef;
using bdf17::equiv_ref::RoundNearestIndexRef;
using bdf17::equiv_ref::ScaleVectorModRef;
using bdf17::equiv_ref::SubMod;
using bdf17::equiv_ref::ToyEqParams;
using bdf17::equiv_ref::TransportPaperFoldedLutToCurrentPairBasisRef;
using bdf17::equiv_ref::TracePtoZRef;
using bdf17::equiv_ref::UnitMonomialRef;
using bdf17::equiv_ref::kAlpha;
using bdf17::equiv_ref::kAlphaInv;
using bdf17::equiv_ref::kBeta;
using bdf17::equiv_ref::kMod;
using bdf17::equiv_ref::kP;
using bdf17::equiv_ref::kPQ;
using bdf17::equiv_ref::kQ;

constexpr size_t kRandomCases = 16;

template <typename PolyT>
std::vector<uint64_t> ToCoeffVector(const PolyT &poly)
requires requires { typename PolyT::Domain; typename PolyT::TransformType; }
{
    std::vector<uint64_t> out(PolyT::N, 0);
    if constexpr (std::is_same_v<typename PolyT::Domain, CoeffTag>) {
        for (size_t i = 0; i < PolyT::N; ++i) {
            out[i] = poly[i];
        }
        return out;
    } else {
        CanonicalNttPlan<typename PolyT::TransformType> plan;
        auto coeff = plan.inverse(poly);
        for (size_t i = 0; i < PolyT::N; ++i) {
            out[i] = coeff[i];
        }
        return out;
    }
}

std::vector<uint64_t> RandomCoeffVector(size_t n, uint64_t seed) {
    std::mt19937_64 engine(seed);
    std::uniform_int_distribution<uint64_t> dist(0, kMod - 1);

    std::vector<uint64_t> out(n, 0);
    for (size_t i = 0; i < n; ++i) {
        out[i] = dist(engine);
    }
    return out;
}

void ExpectVecEqWithFirstMismatch(const std::vector<uint64_t> &actual, const std::vector<uint64_t> &expected, const std::string &context) {
    const auto mismatch = FirstMismatch(actual, expected);
    if (!mismatch.has_value()) {
        return;
    }

    ADD_FAILURE() << context
                  << " first_mismatch_index=" << mismatch->index
                  << " got=" << mismatch->lhs
                  << " expected=" << mismatch->rhs
                  << " actual_size=" << actual.size()
                  << " expected_size=" << expected.size();
}

uint64_t CurrentSemanticScalar(const std::vector<uint64_t> &lut_pair_coeff, const std::vector<uint64_t> &pair_monomial) {
    auto product_pair = Convolution2DRef(lut_pair_coeff, pair_monomial);
    auto product_poly = ToyEqParams::SchemePQ::Coeff::FromUnsigned(product_pair);
    auto traced = bdf17::TracePQtoP<ToyEqParams>(product_poly);
    return bdf17::TracePtoZ<ToyEqParams>(traced);
}

uint64_t PaperSemanticScalar(const std::vector<uint64_t> &lut_paper_folded, size_t folded_monomial_index) {
    const auto paper_monomial = MakeFoldedMonomial(folded_monomial_index);
    auto product_paper = Convolution1DRef(lut_paper_folded, paper_monomial);
    auto traced_paper = PaperTraceCoeffRef(product_paper);
    auto traced_transported = GaloisPRef(traced_paper, kAlphaInv);
    return TracePtoZRef(traced_transported);
}

uint64_t PaperScalarFromFoldedLut(const std::vector<uint64_t> &paper_lut_folded, size_t folded_message_index) {
    const auto paper_monomial = MakeFoldedMonomial(folded_message_index);
    const auto product = Convolution1DRef(paper_lut_folded, paper_monomial);
    const auto traced = PaperTraceCoeffRef(product);
    return TracePtoZRef(traced);
}

std::vector<uint64_t> ToU64(const std::vector<size_t> &in) {
    std::vector<uint64_t> out(in.size(), 0);
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = static_cast<uint64_t>(in[i]);
    }
    return out;
}

std::vector<uint64_t> BuildDeterministicSecret(size_t n, uint64_t seed) {
    std::vector<uint64_t> out(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const uint64_t selector = (seed + i) % 3;
        if (selector == 0) {
            out[i] = 1;
        } else if (selector == 1) {
            out[i] = kMod - 1;
        } else {
            out[i] = 0;
        }
    }
    return out;
}

struct CurrentExpCrtPhaseResult {
    std::vector<uint64_t> phase_pair;
    std::vector<uint64_t> phase_folded;
};

CurrentExpCrtPhaseResult CurrentExpCrtPhaseNoiselessRef(
    const RlweCiphertextRef &cp,
    const RlweCiphertextRef &cq,
    const std::vector<uint64_t> &sp,
    const std::vector<uint64_t> &sq) {
    typename ToyEqParams::SchemePt::Plan plan_p;
    typename ToyEqParams::SchemeQt::Plan plan_q;

    auto p_a = plan_p.forward(ToyEqParams::SchemePt::Coeff::FromUnsigned(cp.a));
    auto p_b = plan_p.forward(ToyEqParams::SchemePt::Coeff::FromUnsigned(cp.b));
    auto q_a = plan_q.forward(ToyEqParams::SchemeQt::Coeff::FromUnsigned(cq.a));
    auto q_b = plan_q.forward(ToyEqParams::SchemeQt::Coeff::FromUnsigned(cq.b));

    typename ToyEqParams::SchemePt::RLWECiphertext ct_p{p_a, p_b};
    typename ToyEqParams::SchemeQt::RLWECiphertext ct_q{q_a, q_b};
    auto tensor_ct = bdf17::TensorCt<ToyEqParams>(ct_p, ct_q);

    if (tensor_ct.size() != 4) {
        throw std::runtime_error("CurrentExpCrtPhaseNoiselessRef expected 4-component tensor ciphertext");
    }
    const auto a0 = ToCoeffVector(tensor_ct[0]);
    const auto a1 = ToCoeffVector(tensor_ct[1]);
    const auto a2 = ToCoeffVector(tensor_ct[2]);
    const auto b = ToCoeffVector(tensor_ct[3]);

    auto s0 = CoeffTensorRef(sp, sq);
    for (size_t i = 0; i < s0.size(); ++i) {
        s0[i] = SubMod(0, s0[i], kMod);
    }
    const auto s1 = CoeffTensorRef(sp, UnitMonomialRef(kQ));
    const auto s2 = CoeffTensorRef(UnitMonomialRef(kP), sq);

    CurrentExpCrtPhaseResult out;
    out.phase_pair = b;
    const auto t0 = Convolution2DRef(a0, s0);
    const auto t1 = Convolution2DRef(a1, s1);
    const auto t2 = Convolution2DRef(a2, s2);
    for (size_t i = 0; i < kPQ; ++i) {
        out.phase_pair[i] = SubMod(out.phase_pair[i], t0[i], kMod);
        out.phase_pair[i] = SubMod(out.phase_pair[i], t1[i], kMod);
        out.phase_pair[i] = SubMod(out.phase_pair[i], t2[i], kMod);
    }
    out.phase_folded = FoldCrtRef(out.phase_pair);
    return out;
}

std::string LutName(size_t lut_id) {
    return lut_id == 0 ? "parity" : "custom";
}

} // namespace

TEST(Equivalence, PaperBootstrapFunctionRefBasicSanity) {
    const std::vector<size_t> parity = bdf17::BuildParityLut<ToyEqParams>();
    const std::vector<size_t> custom{0, 1, 1, 0, 1, 0, 0, 1};

    const auto parity_f = BuildPaperBootstrapFunctionRef(ToU64(parity), ToyEqParams::kPlainModulus, kPQ);
    const auto custom_f = BuildPaperBootstrapFunctionRef(ToU64(custom), ToyEqParams::kPlainModulus, kPQ);

    EXPECT_EQ(parity_f.size(), kPQ);
    EXPECT_EQ(custom_f.size(), kPQ);

    for (size_t m = 0; m < kPQ; ++m) {
        EXPECT_LT(parity_f[m], ToyEqParams::kPlainModulus) << "m=" << m;
        EXPECT_LT(custom_f[m], ToyEqParams::kPlainModulus) << "m=" << m;
        EXPECT_TRUE(parity_f[m] == 0 || parity_f[m] == 1) << "m=" << m;

        const size_t idx = RoundNearestIndexRef(ToyEqParams::kPlainModulus, m, kPQ);
        EXPECT_EQ(custom_f[m], custom[idx]) << "m=" << m << " idx=" << idx;
    }
}

TEST(Equivalence, FoldIndexIdentityExhaustive) {
    std::vector<bool> seen_crt(kPQ, false);
    for (size_t u = 0; u < kP; ++u) {
        for (size_t v = 0; v < kQ; ++v) {
            const size_t lhs = CrtFoldIndex(u, v);
            const size_t rhs = PaperFoldIndex((kAlpha * u) % kP, (kBeta * v) % kQ);
            EXPECT_EQ(lhs, rhs) << "(u,v)=(" << u << "," << v << ")";
            ASSERT_LT(lhs, kPQ);
            EXPECT_FALSE(seen_crt[lhs]) << "CrtFoldIndex is not a permutation at (u,v)=(" << u << "," << v << "), index=" << lhs;
            seen_crt[lhs] = true;
        }
    }

    std::vector<bool> seen_paper(kPQ, false);
    for (size_t a = 0; a < kP; ++a) {
        for (size_t b = 0; b < kQ; ++b) {
            const size_t idx = PaperFoldIndex(a, b);
            ASSERT_LT(idx, kPQ);
            EXPECT_FALSE(seen_paper[idx]) << "PaperFoldIndex is not a permutation at (a,b)=(" << a << "," << b << "), index=" << idx;
            seen_paper[idx] = true;
        }
    }

    EXPECT_TRUE(std::all_of(seen_crt.begin(), seen_crt.end(), [](bool hit) { return hit; }));
    EXPECT_TRUE(std::all_of(seen_paper.begin(), seen_paper.end(), [](bool hit) { return hit; }));
}

TEST(Equivalence, TensorNttMatchesCoeffTensorOnMonomials) {
    ToyEqParams::SchemePt::Plan plan_p;
    ToyEqParams::SchemeQt::Plan plan_q;

    for (size_t a = 0; a < kP; ++a) {
        for (size_t b = 0; b < kQ; ++b) {
            auto lhs_coeff = MakeMonomialP(a);
            auto rhs_coeff = MakeMonomialQ(b);

            auto lhs_ntt = plan_p.forward(ToyEqParams::SchemePt::Coeff::FromUnsigned(lhs_coeff));
            auto rhs_ntt = plan_q.forward(ToyEqParams::SchemeQt::Coeff::FromUnsigned(rhs_coeff));

            auto tensor_ntt = bdf17::Tensor(lhs_ntt, rhs_ntt);
            const auto actual = ToCoeffVector(tensor_ntt);
            const auto expected = CoeffTensorRef(lhs_coeff, rhs_coeff);

            std::ostringstream oss;
            oss << "(a,b)=(" << a << "," << b << ")";
            ExpectVecEqWithFirstMismatch(actual, expected, oss.str());
        }
    }
}

TEST(Equivalence, TensorNttMatchesCoeffTensorOnRandomInputs) {
    ToyEqParams::SchemePt::Plan plan_p;
    ToyEqParams::SchemeQt::Plan plan_q;

    for (size_t case_id = 0; case_id < kRandomCases; ++case_id) {
        auto lhs_coeff = RandomCoeffVector(kP, 0xA110000ULL + case_id);
        auto rhs_coeff = RandomCoeffVector(kQ, 0xB220000ULL + case_id);

        auto lhs_ntt = plan_p.forward(ToyEqParams::SchemePt::Coeff::FromUnsigned(lhs_coeff));
        auto rhs_ntt = plan_q.forward(ToyEqParams::SchemeQt::Coeff::FromUnsigned(rhs_coeff));

        auto tensor_ntt = bdf17::Tensor(lhs_ntt, rhs_ntt);
        const auto actual = ToCoeffVector(tensor_ntt);
        const auto expected = CoeffTensorRef(lhs_coeff, rhs_coeff);

        std::ostringstream oss;
        oss << "case_id=" << case_id;
        ExpectVecEqWithFirstMismatch(actual, expected, oss.str());
    }
}

TEST(Equivalence, CurrentLutSemanticsMatchesIndependentPaperFunction) {
    const std::vector<std::vector<size_t>> plain_luts = {
        bdf17::BuildParityLut<ToyEqParams>(),
        {0, 1, 1, 0, 1, 0, 0, 1},
    };

    for (size_t lut_id = 0; lut_id < plain_luts.size(); ++lut_id) {
        const auto f_paper = BuildPaperBootstrapFunctionRef(ToU64(plain_luts[lut_id]), ToyEqParams::kPlainModulus, kPQ);
        const auto samples_current = bdf17::BuildTensorLutSamples<ToyEqParams>(plain_luts[lut_id]);
        auto lut_current_poly = bdf17::ConstructLutPoly<ToyEqParams>(samples_current);
        const auto lut_current_coeff = ToCoeffVector(lut_current_poly);

        for (size_t m = 0; m < kPQ; ++m) {
            const size_t u = m % kP;
            const size_t v = m % kQ;
            const auto pair_m = MakePairMonomial(u, v);
            const uint64_t got = CurrentSemanticScalar(lut_current_coeff, pair_m);
            const uint64_t expected = f_paper[m];
            EXPECT_EQ(got, expected)
                << "lut=" << LutName(lut_id)
                << " m=" << m
                << " (u,v)=(" << u << "," << v << ")"
                << " expected=" << expected
                << " got=" << got;
        }
    }
}

TEST(Equivalence, PaperLutAndCurrentLutAgreeSemanticallyOnAllMessages) {
    const std::vector<std::vector<size_t>> plain_luts = {
        bdf17::BuildParityLut<ToyEqParams>(),
        {0, 1, 1, 0, 1, 0, 0, 1},
    };

    for (size_t lut_id = 0; lut_id < plain_luts.size(); ++lut_id) {
        const auto f_paper = BuildPaperBootstrapFunctionRef(ToU64(plain_luts[lut_id]), ToyEqParams::kPlainModulus, kPQ);
        const auto paper_lut = PaperLutPolyRef(f_paper);

        const auto samples_current = bdf17::BuildTensorLutSamples<ToyEqParams>(plain_luts[lut_id]);
        auto lut_current_poly = bdf17::ConstructLutPoly<ToyEqParams>(samples_current);
        const auto lut_current_coeff = ToCoeffVector(lut_current_poly);

        for (size_t m = 0; m < kPQ; ++m) {
            const size_t u = m % kP;
            const size_t v = m % kQ;
            const uint64_t current_scalar = CurrentSemanticScalar(lut_current_coeff, MakePairMonomial(u, v));
            const uint64_t paper_scalar = PaperScalarFromFoldedLut(paper_lut, m);
            const uint64_t expected = f_paper[m];
            EXPECT_EQ(current_scalar, expected)
                << "lut=" << LutName(lut_id) << " m=" << m << " current expected mismatch";
            EXPECT_EQ(paper_scalar, expected)
                << "lut=" << LutName(lut_id) << " m=" << m << " paper expected mismatch";
            EXPECT_EQ(current_scalar, paper_scalar)
                << "lut=" << LutName(lut_id) << " m=" << m << " current-paper mismatch";
        }
    }
}

TEST(Equivalence, FoldCrtEqualsFoldPaperAfterTwistsExhaustive) {
    for (size_t u = 0; u < kP; ++u) {
        for (size_t v = 0; v < kQ; ++v) {
            const auto monomial = MakePairMonomial(u, v);
            const auto fold_crt = FoldCrtRef(monomial);
            const auto fold_paper = FoldPaperRef(GaloisPairRef(monomial, kAlpha, kBeta));

            std::ostringstream oss;
            oss << "(u,v)=(" << u << "," << v << ")";
            ExpectVecEqWithFirstMismatch(fold_crt, fold_paper, oss.str());
        }
    }
}

TEST(Equivalence, FoldCrtEqualsFoldPaperAfterTwistsRandom) {
    for (size_t case_id = 0; case_id < kRandomCases; ++case_id) {
        const auto pair_coeff = RandomCoeffVector(kPQ, 0xC330000ULL + case_id);
        const auto fold_crt = FoldCrtRef(pair_coeff);
        const auto fold_paper = FoldPaperRef(GaloisPairRef(pair_coeff, kAlpha, kBeta));

        std::ostringstream oss;
        oss << "case_id=" << case_id;
        ExpectVecEqWithFirstMismatch(fold_crt, fold_paper, oss.str());
    }
}

TEST(Equivalence, ExpCrtCiphertextPhaseMatchesPaperReferenceNoiseless) {
    const uint64_t expcrt_scale = (ToyEqParams::SchemePQ::Q - ToyEqParams::kPlainModulus) % kMod;
    const uint64_t delta = 1;
    const uint64_t expected_phase_unit = MulMod(expcrt_scale, MulMod(delta, delta, kMod), kMod);

    const auto sp = BuildDeterministicSecret(kP, 0x51);
    const auto sq = BuildDeterministicSecret(kQ, 0xA2);

    for (size_t m_p = 0; m_p < kP; ++m_p) {
        for (size_t m_q = 0; m_q < kQ; ++m_q) {
            const auto cp = MakeNoiselessRlweCtRef(
                sp,
                MakeMonomialP(m_p),
                delta,
                static_cast<uint64_t>(0x110000 + m_p * kQ + m_q));
            const auto cq = MakeNoiselessRlweCtRef(
                sq,
                MakeMonomialQ(m_q),
                delta,
                static_cast<uint64_t>(0x220000 + m_p * kQ + m_q));

            const auto current_phase = CurrentExpCrtPhaseNoiselessRef(cp, cq, sp, sq);
            const auto paper_ref = PaperExpCrtRef(cp, cq, sp, sq, expcrt_scale);
            const auto paper_phase = PhaseRef(paper_ref.ct, paper_ref.secret);

            const size_t expected = CrtFoldIndex(m_p, m_q);

            size_t got_current = 0;
            size_t got_paper = 0;
            try {
                got_current = DecodeMonomialPhaseRef(current_phase.phase_folded, expected_phase_unit);
            } catch (const std::exception &e) {
                ADD_FAILURE() << "(m_p,m_q)=(" << m_p << "," << m_q << ") current decode error: " << e.what();
                continue;
            }
            try {
                got_paper = DecodeMonomialPhaseRef(paper_phase, expected_phase_unit);
            } catch (const std::exception &e) {
                ADD_FAILURE() << "(m_p,m_q)=(" << m_p << "," << m_q << ") paper decode error: " << e.what();
                continue;
            }

            EXPECT_EQ(got_current, expected)
                << "(m_p,m_q)=(" << m_p << "," << m_q << ") expected=" << expected << " got_current=" << got_current;
            EXPECT_EQ(got_paper, expected)
                << "(m_p,m_q)=(" << m_p << "," << m_q << ") expected=" << expected << " got_paper=" << got_paper;

            std::ostringstream oss;
            oss << "(m_p,m_q)=(" << m_p << "," << m_q << ")";
            ExpectVecEqWithFirstMismatch(current_phase.phase_folded, paper_phase, oss.str());
        }
    }
}

TEST(Equivalence, ExpCrtPlusF0ExtractionMatchesPaperReferenceNoiseless) {
    const uint64_t expcrt_scale = (ToyEqParams::SchemePQ::Q - ToyEqParams::kPlainModulus) % kMod;
    const uint64_t delta = 1;
    const uint64_t expected_phase_unit = MulMod(expcrt_scale, MulMod(delta, delta, kMod), kMod);
    const uint64_t phase_unit_inv = InvMod(expected_phase_unit, kMod);

    std::vector<uint64_t> f0(kPQ, 0);
    f0[0] = 1;
    const auto paper_f0_lut = PaperLutPolyRef(f0);
    const auto current_f0_lut_pair = TransportPaperFoldedLutToCurrentPairBasisRef(paper_f0_lut);

    const auto sp = BuildDeterministicSecret(kP, 0x71);
    const auto sq = BuildDeterministicSecret(kQ, 0xB3);

    for (size_t m_p = 0; m_p < kP; ++m_p) {
        for (size_t m_q = 0; m_q < kQ; ++m_q) {
            const auto cp = MakeNoiselessRlweCtRef(
                sp,
                MakeMonomialP(m_p),
                delta,
                static_cast<uint64_t>(0x330000 + m_p * kQ + m_q));
            const auto cq = MakeNoiselessRlweCtRef(
                sq,
                MakeMonomialQ(m_q),
                delta,
                static_cast<uint64_t>(0x440000 + m_p * kQ + m_q));

            const auto current_phase = CurrentExpCrtPhaseNoiselessRef(cp, cq, sp, sq);
            const auto paper_ref = PaperExpCrtRef(cp, cq, sp, sq, expcrt_scale);
            const auto paper_phase = PhaseRef(paper_ref.ct, paper_ref.secret);

            const auto current_phase_monomial_pair = ScaleVectorModRef(current_phase.phase_pair, phase_unit_inv);
            const auto paper_phase_monomial = ScaleVectorModRef(paper_phase, phase_unit_inv);

            const auto current_product_pair = Convolution2DRef(current_f0_lut_pair, current_phase_monomial_pair);
            const auto current_trace = bdf17::TracePQtoP<ToyEqParams>(ToyEqParams::SchemePQ::Coeff::FromUnsigned(current_product_pair));
            const uint64_t current_scalar = bdf17::TracePtoZ<ToyEqParams>(current_trace);

            const auto paper_product = Convolution1DRef(paper_f0_lut, paper_phase_monomial);
            const auto paper_trace = PaperTraceCoeffRef(paper_product);
            const uint64_t paper_scalar = TracePtoZRef(paper_trace);

            const size_t expected_idx = CrtFoldIndex(m_p, m_q);
            const uint64_t expected = expected_idx == 0 ? 1 : 0;
            EXPECT_EQ(current_scalar, expected)
                << "(m_p,m_q)=(" << m_p << "," << m_q << ") expected=" << expected << " got_current=" << current_scalar;
            EXPECT_EQ(paper_scalar, expected)
                << "(m_p,m_q)=(" << m_p << "," << m_q << ") expected=" << expected << " got_paper=" << paper_scalar;
        }
    }
}

TEST(Equivalence, TraceCoeffMatchesTransportedPaperTraceExhaustive) {
    for (size_t u = 0; u < kP; ++u) {
        for (size_t v = 0; v < kQ; ++v) {
            const auto pair_coeff = MakePairMonomial(u, v);
            auto pair_poly = ToyEqParams::SchemePQ::Coeff::FromUnsigned(pair_coeff);

            const auto actual = ToCoeffVector(bdf17::TracePQtoP<ToyEqParams>(pair_poly));
            const auto expected = CrtTraceTransportedRef(pair_coeff);

            std::ostringstream oss;
            oss << "(u,v)=(" << u << "," << v << ")";
            ExpectVecEqWithFirstMismatch(actual, expected, oss.str());
        }
    }
}

TEST(Equivalence, TraceNttMatchesTransportedPaperTraceRandom) {
    for (size_t case_id = 0; case_id < kRandomCases; ++case_id) {
        const auto pair_coeff = RandomCoeffVector(kPQ, 0xD440000ULL + case_id);
        ToyEqParams::SchemePQ::Plan plan_pq;
        auto pair_poly = plan_pq.forward(ToyEqParams::SchemePQ::Coeff::FromUnsigned(pair_coeff));

        const auto actual = ToCoeffVector(bdf17::TracePQtoP<ToyEqParams>(pair_poly));
        const auto expected = CrtTraceTransportedRef(pair_coeff);

        std::ostringstream oss;
        oss << "case_id=" << case_id;
        ExpectVecEqWithFirstMismatch(actual, expected, oss.str());
    }
}

TEST(Equivalence, LutAndTraceSemanticsAgreeOnAllMessages) {
    const std::vector<std::vector<size_t>> plain_luts = {
        bdf17::BuildParityLut<ToyEqParams>(),
        {0, 1, 1, 0, 1, 0, 0, 1},
    };

    for (size_t lut_id = 0; lut_id < plain_luts.size(); ++lut_id) {
        const auto samples = bdf17::BuildTensorLutSamples<ToyEqParams>(plain_luts[lut_id]);
        auto lut_pair_poly = bdf17::ConstructLutPoly<ToyEqParams>(samples);
        const auto lut_pair_coeff = ToCoeffVector(lut_pair_poly);
        const auto lut_paper = FoldPaperRef(GaloisPairRef(lut_pair_coeff, kAlpha, kBeta));

        for (size_t m = 0; m < kPQ; ++m) {
            const size_t u = m % kP;
            const size_t v = m % kQ;
            const auto pair_m = MakePairMonomial(u, v);
            const size_t folded_index = CrtFoldIndex(u, v);

            const uint64_t current_scalar = CurrentSemanticScalar(lut_pair_coeff, pair_m);
            const uint64_t paper_scalar = PaperSemanticScalar(lut_paper, folded_index);
            EXPECT_EQ(current_scalar, paper_scalar)
                << "lut_id=" << lut_id << " m=" << m << " (u,v)=(" << u << "," << v << ") folded_index=" << folded_index;
        }
    }
}

TEST(Equivalence, EndToEndNoiselessSemanticAgreement) {
    const std::vector<size_t> plain_lut{0, 1, 1, 0, 1, 0, 0, 1};
    const auto samples = bdf17::BuildTensorLutSamples<ToyEqParams>(plain_lut);

    auto lut_pair_coeff = bdf17::ConstructLutPoly<ToyEqParams>(samples);
    ToyEqParams::SchemePQ::Plan plan_pq;
    auto lut_pair_ntt = plan_pq.forward(lut_pair_coeff);

    const auto lut_pair_coeff_vec = ToCoeffVector(lut_pair_coeff);
    const auto lut_paper = FoldPaperRef(GaloisPairRef(lut_pair_coeff_vec, kAlpha, kBeta));

    ToyEqParams::SchemePt::Plan plan_p;
    ToyEqParams::SchemeQt::Plan plan_q;

    const std::vector<size_t> messages{0, 1, 2, 7, 13, 42, 90};
    for (size_t m : messages) {
        const size_t u = m % kP;
        const size_t v = m % kQ;

        auto monomial_p = plan_p.forward(ToyEqParams::SchemePt::Coeff::FromUnsigned(MakeMonomialP(u)));
        auto monomial_q = plan_q.forward(ToyEqParams::SchemeQt::Coeff::FromUnsigned(MakeMonomialQ(v)));

        auto tensor_state_ntt = bdf17::Tensor(monomial_p, monomial_q);
        auto multiplied_ntt = lut_pair_ntt * tensor_state_ntt;
        const auto traced_ntt = bdf17::TracePQtoP<ToyEqParams>(multiplied_ntt);
        const uint64_t current_scalar = bdf17::TracePtoZ<ToyEqParams>(traced_ntt);

        const uint64_t paper_scalar = PaperSemanticScalar(lut_paper, CrtFoldIndex(u, v));
        EXPECT_EQ(current_scalar, paper_scalar) << "m=" << m << " (u,v)=(" << u << "," << v << ")";
    }
}
