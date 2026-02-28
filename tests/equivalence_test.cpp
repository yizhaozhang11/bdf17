#include <algorithm>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "expcrt.hpp"
#include "fun_extract.hpp"

#include "equivalence_ref.hpp"

namespace {

using bdf17::equiv_ref::CoeffTensorRef;
using bdf17::equiv_ref::Convolution1DRef;
using bdf17::equiv_ref::Convolution2DRef;
using bdf17::equiv_ref::CrtFoldIndex;
using bdf17::equiv_ref::CrtTraceTransportedRef;
using bdf17::equiv_ref::FirstMismatch;
using bdf17::equiv_ref::FoldCrtRef;
using bdf17::equiv_ref::FoldPaperRef;
using bdf17::equiv_ref::GaloisPairRef;
using bdf17::equiv_ref::GaloisPRef;
using bdf17::equiv_ref::MakeFoldedMonomial;
using bdf17::equiv_ref::MakeMonomialP;
using bdf17::equiv_ref::MakeMonomialQ;
using bdf17::equiv_ref::MakePairMonomial;
using bdf17::equiv_ref::PaperFoldIndex;
using bdf17::equiv_ref::PaperTraceCoeffRef;
using bdf17::equiv_ref::ToyEqParams;
using bdf17::equiv_ref::TracePtoZRef;
using bdf17::equiv_ref::kAlpha;
using bdf17::equiv_ref::kAlphaInv;
using bdf17::equiv_ref::kBeta;
using bdf17::equiv_ref::kMod;
using bdf17::equiv_ref::kP;
using bdf17::equiv_ref::kPQ;
using bdf17::equiv_ref::kQ;

constexpr size_t kRandomCases = 16;

template <typename PolyT>
std::vector<uint64_t> ToCoeffVector(PolyT poly) {
    poly.ToCoeff();
    std::vector<uint64_t> out(PolyT::N, 0);
    for (size_t i = 0; i < PolyT::N; ++i) {
        out[i] = poly.a[i];
    }
    return out;
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
    auto product_poly = ToyEqParams::PolyPQ::FromCoeff(product_pair);
    auto traced = bdf17::TracePQtoP<ToyEqParams>(product_poly);
    return bdf17::TracePtoZ(traced);
}

uint64_t PaperSemanticScalar(const std::vector<uint64_t> &lut_paper_folded, size_t folded_monomial_index) {
    const auto paper_monomial = MakeFoldedMonomial(folded_monomial_index);
    auto product_paper = Convolution1DRef(lut_paper_folded, paper_monomial);
    auto traced_paper = PaperTraceCoeffRef(product_paper);
    auto traced_transported = GaloisPRef(traced_paper, kAlphaInv);
    return TracePtoZRef(traced_transported);
}

} // namespace

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
    for (size_t a = 0; a < kP; ++a) {
        for (size_t b = 0; b < kQ; ++b) {
            auto lhs_coeff = MakeMonomialP(a);
            auto rhs_coeff = MakeMonomialQ(b);

            auto lhs_ntt = ToyEqParams::PolyPt::FromCoeff(lhs_coeff);
            auto rhs_ntt = ToyEqParams::PolyQt::FromCoeff(rhs_coeff);
            lhs_ntt.ToNTT();
            rhs_ntt.ToNTT();

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
    for (size_t case_id = 0; case_id < kRandomCases; ++case_id) {
        auto lhs_coeff = RandomCoeffVector(kP, 0xA110000ULL + case_id);
        auto rhs_coeff = RandomCoeffVector(kQ, 0xB220000ULL + case_id);

        auto lhs_ntt = ToyEqParams::PolyPt::FromCoeff(lhs_coeff);
        auto rhs_ntt = ToyEqParams::PolyQt::FromCoeff(rhs_coeff);
        lhs_ntt.ToNTT();
        rhs_ntt.ToNTT();

        auto tensor_ntt = bdf17::Tensor(lhs_ntt, rhs_ntt);
        const auto actual = ToCoeffVector(tensor_ntt);
        const auto expected = CoeffTensorRef(lhs_coeff, rhs_coeff);

        std::ostringstream oss;
        oss << "case_id=" << case_id;
        ExpectVecEqWithFirstMismatch(actual, expected, oss.str());
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

TEST(Equivalence, TraceCoeffMatchesTransportedPaperTraceExhaustive) {
    for (size_t u = 0; u < kP; ++u) {
        for (size_t v = 0; v < kQ; ++v) {
            const auto pair_coeff = MakePairMonomial(u, v);
            auto pair_poly = ToyEqParams::PolyPQ::FromCoeff(pair_coeff);

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
        auto pair_poly = ToyEqParams::PolyPQ::FromCoeff(pair_coeff);
        pair_poly.ToNTT();

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
    auto lut_pair_ntt = lut_pair_coeff;
    lut_pair_ntt.ToNTT();

    const auto lut_pair_coeff_vec = ToCoeffVector(lut_pair_coeff);
    const auto lut_paper = FoldPaperRef(GaloisPairRef(lut_pair_coeff_vec, kAlpha, kBeta));

    const std::vector<size_t> messages{0, 1, 2, 7, 13, 42, 90};
    for (size_t m : messages) {
        const size_t u = m % kP;
        const size_t v = m % kQ;

        auto monomial_p = ToyEqParams::PolyPt::FromCoeff(MakeMonomialP(u));
        auto monomial_q = ToyEqParams::PolyQt::FromCoeff(MakeMonomialQ(v));
        monomial_p.ToNTT();
        monomial_q.ToNTT();

        auto tensor_state_ntt = bdf17::Tensor(monomial_p, monomial_q);
        auto multiplied_ntt = lut_pair_ntt * tensor_state_ntt;
        const auto traced_ntt = bdf17::TracePQtoP<ToyEqParams>(multiplied_ntt);
        const uint64_t current_scalar = bdf17::TracePtoZ(traced_ntt);

        const uint64_t paper_scalar = PaperSemanticScalar(lut_paper, CrtFoldIndex(u, v));
        EXPECT_EQ(current_scalar, paper_scalar) << "m=" << m << " (u,v)=(" << u << "," << v << ")";
    }
}
