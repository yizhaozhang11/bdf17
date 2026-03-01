#include <gtest/gtest.h>

#include "params.hpp"

TEST(RlweShape, Ct2AdaptersRoundTripAndValidateShape) {
    using Scheme = bdf17::ToyEquivalenceProfile::SchemePt;
    using Eval = Scheme::Eval;

    Eval a;
    Eval b;
    a[0] = 1;
    b[0] = 2;

    Scheme::RLWECiphertext legacy{a, b};
    const auto ct2 = Scheme::ToRlweCt2(legacy);
    EXPECT_EQ(ct2.a, a);
    EXPECT_EQ(ct2.b, b);

    const auto back = Scheme::FromRlweCt2(ct2);
    ASSERT_EQ(back.size(), 2u);
    EXPECT_EQ(back[0], a);
    EXPECT_EQ(back[1], b);

    Scheme::RLWECiphertext wrong{a};
    EXPECT_THROW((void)Scheme::ToRlweCt2(wrong), std::runtime_error);
}

TEST(RlweShape, TensorCt4AdaptersRoundTripAndValidateShape) {
    using Scheme = bdf17::ToyEquivalenceProfile::SchemePQ;
    using Eval = Scheme::Eval;

    Eval c0;
    Eval c1;
    Eval c2;
    Eval c3;
    c0[0] = 3;
    c1[0] = 4;
    c2[0] = 5;
    c3[0] = 6;

    Scheme::TensorCt4 tensor{c0, c1, c2, c3};
    const auto legacy = Scheme::FromTensorCt4(tensor);
    ASSERT_EQ(legacy.size(), 4u);

    const auto back = Scheme::ToTensorCt4(legacy);
    EXPECT_EQ(back.c0, c0);
    EXPECT_EQ(back.c1, c1);
    EXPECT_EQ(back.c2, c2);
    EXPECT_EQ(back.c3, c3);

    Scheme::RLWECiphertext wrong{c0, c1, c2};
    EXPECT_THROW((void)Scheme::ToTensorCt4(wrong), std::runtime_error);
}
