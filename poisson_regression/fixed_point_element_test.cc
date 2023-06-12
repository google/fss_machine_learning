/*
 * Copyright 2020 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "poisson_regression/fixed_point_element.h"

#include "poisson_regression/fixed_point_element_factory.h"
#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"

namespace private_join_and_compute {
namespace {

using ::testing::HasSubstr;
using testing::StatusIs;

const FixedPointElementFactory::Params kSampleParams
    = {10, 32, 1024, 4194304, 4294967296};  // {10, 32, 2^10, 2^22, 2^32}

const uint64_t kTwoPower31 = 2147483648;

class FixedPointElementTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Create a sample 32-bit factory.
    ASSERT_OK_AND_ASSIGN(
        FixedPointElementFactory temp32,
        FixedPointElementFactory::Create(
            kSampleParams.num_fractional_bits,
            kSampleParams.num_ring_bits));
    self_32bit_factory_ = absl::make_unique<FixedPointElementFactory>(
        std::move(temp32));
  }

 protected:
  std::unique_ptr<FixedPointElementFactory> self_32bit_factory_;
};

TEST_F(FixedPointElementTest, TestPerfectExportToDouble) {
  std::vector<double> double_values = {-1.5, 0, 1.5, 2.75};
  for (size_t i = 0; i < double_values.size(); i++) {
    ASSERT_OK_AND_ASSIGN(
        FixedPointElement fpe,
        self_32bit_factory_->CreateFixedPointElementFromDouble(
            double_values[i]));
    EXPECT_EQ(fpe.ExportToDouble(), double_values[i]);
  }
}

TEST_F(FixedPointElementTest, TestClosestExportToDouble) {
  // double values that are not completely representable in the ring.
  std::vector<double> imperfect_doubles = {-2.44, -1.3, 0.7, 2.1};
  for (size_t i = 0; i < imperfect_doubles.size(); i++) {
    ASSERT_OK_AND_ASSIGN(
        FixedPointElement fpe,
        self_32bit_factory_->CreateFixedPointElementFromDouble(
            imperfect_doubles[i]));
    double closest_representation;
    if (imperfect_doubles[i] >= 0) {
      closest_representation =
        floor(imperfect_doubles[i] * kSampleParams.fractional_multiplier)
        / kSampleParams.fractional_multiplier;
    } else {
      closest_representation =
        ceil(imperfect_doubles[i] * kSampleParams.fractional_multiplier)
        / kSampleParams.fractional_multiplier;
    }
    EXPECT_EQ(fpe.ExportToDouble(), closest_representation);
  }
}

TEST_F(FixedPointElementTest, TestModAddWrapAround) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->ImportFixedPointElementFromUint64(
                           kTwoPower31));
  ASSERT_OK_AND_ASSIGN(FixedPointElement result, fpe + fpe);
  EXPECT_EQ(result.ExportToUint64(), 0);
}

TEST_F(FixedPointElementTest,
       TestModAddEqualsRealAdditionUptoFixedPointRepresentation) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->CreateFixedPointElementFromDouble(
                           1.5));
  ASSERT_OK_AND_ASSIGN(FixedPointElement result, fpe + fpe);

  // result should have value (1.5 + 1.5) * 1024 = 3072.
  EXPECT_EQ(result.ExportToUint64(), 3072);
}

TEST_F(FixedPointElementTest, TestModAddInverseDoublesResultsInZero) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement positive_fpe,
                       self_32bit_factory_->CreateFixedPointElementFromDouble(
                           1.5));
  ASSERT_OK_AND_ASSIGN(FixedPointElement negative_fpe,
                       self_32bit_factory_->CreateFixedPointElementFromDouble(
                           -1.5));
  ASSERT_OK_AND_ASSIGN(FixedPointElement result, positive_fpe + negative_fpe);

  // result should have value (1.5 + -1.5) * 1024 = 0.
  EXPECT_EQ(result.ExportToUint64(), 0);
}

TEST_F(FixedPointElementTest, TestModSub) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->ImportFixedPointElementFromUint64(
                           kTwoPower31));
  ASSERT_OK_AND_ASSIGN(FixedPointElement result, fpe - fpe);
  EXPECT_EQ(result.ExportToUint64(), 0);
}

TEST_F(FixedPointElementTest, TestModMulWrapAround) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->ImportFixedPointElementFromUint64(
                           kTwoPower31 - 1));
  ASSERT_OK_AND_ASSIGN(FixedPointElement result, fpe * fpe);
  EXPECT_EQ(result.ExportToUint64(), 1);
}

TEST_F(FixedPointElementTest, TestNegateAndDoubleNegate) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->ImportFixedPointElementFromUint64(
                           kTwoPower31 - 1));
  ASSERT_OK_AND_ASSIGN(FixedPointElement negation, -fpe);
  ASSERT_OK_AND_ASSIGN(FixedPointElement double_negation, -negation);
  EXPECT_EQ(negation.ExportToUint64(), kTwoPower31 + 1);
  EXPECT_EQ(double_negation.ExportToUint64(), fpe.ExportToUint64());
}

TEST_F(FixedPointElementTest, TestNegateZeroEqualsZero) {
  ASSERT_OK_AND_ASSIGN(
      FixedPointElement fpe,
      self_32bit_factory_->ImportFixedPointElementFromUint64(0));
  ASSERT_OK_AND_ASSIGN(FixedPointElement negated_fpe, fpe.Negate());
  EXPECT_EQ(negated_fpe.ExportToUint64(), 0);
}

TEST_F(FixedPointElementTest, TestLargeTruncMul) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->ImportFixedPointElementFromUint64(
                           kTwoPower31 - 1));
  ASSERT_OK_AND_ASSIGN(FixedPointElement result, fpe.TruncMul(fpe));
  // result should be floor(((2^31 - 1) * (2^31 - 1)) / 2^10) = 4290772992.
  EXPECT_EQ(result.ExportToUint64(), 4290772992);
}

TEST_F(FixedPointElementTest, TestPowerOfTwoDivide) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->ImportFixedPointElementFromUint64(
                           kTwoPower31 - 1));
  ASSERT_OK_AND_ASSIGN(FixedPointElement result, fpe.PowerOfTwoDiv(10));
  // result should be floor((2^31 - 1) / 2^10) = 2097151.
  EXPECT_EQ(result.ExportToUint64(), 2097151);
}

TEST_F(FixedPointElementTest, TestDivideByTwoPowerResultsInZero) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->ImportFixedPointElementFromUint64(
                           kTwoPower31 - 1));
  ASSERT_OK_AND_ASSIGN(FixedPointElement result, fpe.PowerOfTwoDiv(32));
  // result should be floor((2^31 - 1) / 2^32) = 0.
  EXPECT_EQ(result.ExportToUint64(), 0);
}

TEST_F(FixedPointElementTest, TestInvalidPowerOfTwoDivide) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->ImportFixedPointElementFromUint64(
                           kTwoPower31 - 1));
  EXPECT_THAT(fpe.PowerOfTwoDiv(kSampleParams.num_ring_bits + 1),
              StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid exponent exp")));
}

TEST_F(FixedPointElementTest, TestPowerOfTwoDivFP) {
  // Test 1: PowerOfTwoDivFP on positive fixed point number.
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->ImportFixedPointElementFromUint64(
                           kTwoPower31 - 1));
  ASSERT_OK_AND_ASSIGN(auto result, fpe.PowerOfTwoDivFP(10));
  // result should be (floor((2^31 - 1)/2^{10}) = 2097151.
  EXPECT_EQ(result.ExportToUint64(), 2097151);

  // Test 2: PowerOfTwoDivFP on negative fixed point number.
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe_negate, fpe.Negate());
  ASSERT_OK_AND_ASSIGN(result, fpe_negate.PowerOfTwoDivFP(10));
  // result should be (2^{32} - floor((2^31 - 1)/2^{10}) = 2^{32} - 2097151.
  EXPECT_EQ(result.ExportToUint64(), 4292870145);

  // Test 3: PowerOfTwoDivFP on zero.
  ASSERT_OK_AND_ASSIGN(FixedPointElement zero, fpe + fpe_negate);
  ASSERT_OK_AND_ASSIGN(result, zero.PowerOfTwoDivFP(10));
  // result should be (floor(0/2^{10}) = 0.
  EXPECT_EQ(result.ExportToUint64(), 0);
}

TEST_F(FixedPointElementTest, TestSmallFractionalTrucMultResultsInZero) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->CreateFixedPointElementFromDouble(
                           0.01));
  // fpe is not equal to zero.
  EXPECT_NE(fpe.ExportToUint64(), 0);

  // but both fpe.TruncMul(fpe) and fpe.TruncMulFP(fpe) should be zero.
  ASSERT_OK_AND_ASSIGN(FixedPointElement result, fpe.TruncMul(fpe));
  ASSERT_OK_AND_ASSIGN(FixedPointElement result_trunc_fp, fpe.TruncMulFP(fpe));
  EXPECT_EQ(result.ExportToUint64(), 0);
  EXPECT_EQ(result_trunc_fp.ExportToUint64(), 0);
}

TEST_F(FixedPointElementTest, TestSmallNegFractionalTrucMulFPResultsInZero) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->CreateFixedPointElementFromDouble(
                           -0.01));
  // fpe is not equal to zero.
  EXPECT_NE(fpe.ExportToUint64(), 0);

  // but fpe.TruncMulFP(fpe) should be zero.
  // note that fpe.TruncMul(fpe) will not be zero.
  ASSERT_OK_AND_ASSIGN(FixedPointElement result_trunc_fp, fpe.TruncMulFP(fpe));
  EXPECT_EQ(result_trunc_fp.ExportToUint64(), 0);
}

TEST_F(FixedPointElementTest, TestTruncMulFPApproximatesRealNumberMult) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->CreateFixedPointElementFromDouble(
                           11.71));

  // Compare computing fpe*fpe in the ring vs in real numbers.
  // Compute result in the ring.
  ASSERT_OK_AND_ASSIGN(FixedPointElement trunc_mult_fpe, fpe.TruncMulFP(fpe));

  // Compute result in real numbers.
  double closest_representable_double
      = 1.0 * static_cast<int64_t>(11.71 * kSampleParams.fractional_multiplier)
      / kSampleParams.fractional_multiplier;
  ASSERT_OK_AND_ASSIGN(FixedPointElement real_mult_fpe,
                       self_32bit_factory_->CreateFixedPointElementFromDouble(
                           closest_representable_double *
                           closest_representable_double));

  uint64_t trunc_mult_uint_result = trunc_mult_fpe.ExportToUint64();
  uint64_t real_mult_uint_result = real_mult_fpe.ExportToUint64();
  uint64_t difference = trunc_mult_uint_result - real_mult_uint_result;

  // Difference between the two outputs should be -1, 0, or 1.
  // i.e., difference + 1 should be in (0,1,2).
  EXPECT_LE(difference + 1, 2);
}

TEST_F(FixedPointElementTest, TestTruncMultFPTwoNegativesResultsInPositive) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_32bit_factory_->CreateFixedPointElementFromDouble(
                           -1.5));
  ASSERT_OK_AND_ASSIGN(FixedPointElement result, fpe.TruncMulFP(fpe));

  // result should have value (-1.5 * -1.5) * 1024 = 2304.
  EXPECT_EQ(result.ExportToUint64(), 2304);
}

TEST_F(FixedPointElementTest, TestTruncMulFPOnePositiveAndOneNegative) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement positive_fpe,
                       self_32bit_factory_->CreateFixedPointElementFromDouble(
                           1.5));
  ASSERT_OK_AND_ASSIGN(FixedPointElement negative_fpe,
                       self_32bit_factory_->CreateFixedPointElementFromDouble(
                           -1.5));
  ASSERT_OK_AND_ASSIGN(FixedPointElement result,
                       positive_fpe.TruncMulFP(negative_fpe));

  // result should have value (1.5 * -1.5) * 1024 = -2304 = 4294964992
  EXPECT_EQ(result.ExportToUint64(), 4294964992);
}

TEST_F(FixedPointElementTest,
       TestTruncMulAndTruncMulFPAgreeOnTwoPositiveMult) {
  auto positive_multiplicands = std::vector<double>{1, 1.4, 2.1, 4.3};

  for (size_t i = 0; i < positive_multiplicands.size(); i++) {
    for (size_t j = 0; j < positive_multiplicands.size(); j++) {
      ASSERT_OK_AND_ASSIGN(FixedPointElement fpe_1,
                       self_32bit_factory_->CreateFixedPointElementFromDouble(
                           positive_multiplicands[i]));
      ASSERT_OK_AND_ASSIGN(FixedPointElement fpe_2,
                 self_32bit_factory_->CreateFixedPointElementFromDouble(
                     positive_multiplicands[j]));
      ASSERT_OK_AND_ASSIGN(FixedPointElement result, fpe_1.TruncMul(fpe_2));
      ASSERT_OK_AND_ASSIGN(FixedPointElement result_mult_fp,
                           fpe_1.TruncMulFP(fpe_2));

      // TruncMul and TruncMulFP should give the same answer when both FPE are
      // positive.
      EXPECT_EQ(result.ExportToUint64(), result_mult_fp.ExportToUint64());
    }
  }
}

}  // namespace
}  // namespace private_join_and_compute
