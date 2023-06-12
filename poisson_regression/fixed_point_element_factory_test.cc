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

#include "poisson_regression/fixed_point_element_factory.h"

#include "poisson_regression/fixed_point_element.h"
#include "private_join_and_compute/util/status.inc"
#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"

namespace private_join_and_compute{
namespace {

using ::testing::HasSubstr;
using testing::StatusIs;

const int64_t kTooLargeRingBits = 65;
const int64_t kNegativeRingBits = -1;
const int64_t kNegativeFractionalBits = -1;

const FixedPointElementFactory::Params kSampleParams
    = {10, 32, 1024, 4194304, 4294967296};  // {10, 32, 2^10, 2^22, 2^32}

const uint64_t kTwoPower32 = 4294967296;  // 2^{32}
const double kSampleFixedPointDouble = 1.5;
const uint64_t kSampleUintVal = 1536;  // represents 1.5 in ring {10, 32}
const double kSampleNegativeDouble = -1.5;
const uint64_t kSampleNegativeUintVal = 4294965760;  // 2^{32} - 1536

class FixedPointElementFactoryCreateTest : public ::testing::Test {
 public:
  FixedPointElementFactory::Params ReturnParams(
      FixedPointElementFactory& factory) {
    return factory.GetParams();
  }
};

TEST_F(FixedPointElementFactoryCreateTest, TestInvalidRingSize) {
  EXPECT_THAT(
      FixedPointElementFactory::Create(10, kTooLargeRingBits),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("Invalid num_ring_bits")));
  EXPECT_THAT(
      FixedPointElementFactory::Create(10, kNegativeRingBits),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("Invalid num_ring_bits")));
}

TEST_F(FixedPointElementFactoryCreateTest, TestInvalidFractionalBits) {
  EXPECT_THAT(
      FixedPointElementFactory::Create(
          kSampleParams.num_ring_bits + 1, kSampleParams.num_ring_bits),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("Invalid num_fractional_bits")));
  EXPECT_THAT(
      FixedPointElementFactory::Create(
          kNegativeFractionalBits, kSampleParams.num_ring_bits),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("Invalid num_fractional_bits")));
}

TEST_F(FixedPointElementFactoryCreateTest, TestValidFactoryCreation) {
  ASSERT_OK_AND_ASSIGN(FixedPointElementFactory fpe_factory,
                       FixedPointElementFactory::Create(
                           kSampleParams.num_fractional_bits,
                           kSampleParams.num_ring_bits));
  FixedPointElementFactory::Params params = ReturnParams(fpe_factory);

  // Compare Parameters
  EXPECT_EQ(params.num_fractional_bits, kSampleParams.num_fractional_bits);
  EXPECT_EQ(params.num_ring_bits, kSampleParams.num_ring_bits);
  EXPECT_EQ(params.fractional_multiplier, kSampleParams.fractional_multiplier);
  EXPECT_EQ(params.integer_ring_modulus, kSampleParams.integer_ring_modulus);
  EXPECT_EQ(params.primary_ring_modulus, kSampleParams.primary_ring_modulus);
}

TEST_F(FixedPointElementFactoryCreateTest, TestIsUint64InRing) {
  ASSERT_OK_AND_ASSIGN(FixedPointElementFactory fpe_factory,
                       FixedPointElementFactory::Create(
                           kSampleParams.num_fractional_bits,
                           kSampleParams.num_ring_bits));
  EXPECT_FALSE(fpe_factory.IsUint64InRing(kTwoPower32));
  EXPECT_TRUE(fpe_factory.IsUint64InRing(kTwoPower32 - 1));
}

class FixedPointElementFactoryCreateElementTest : public ::testing::Test {
 public:
  bool IsSameFactoryAs(FixedPointElementFactory& factory) {
    return self_fpe_factory_->IsSameFactoryAs(factory);
  }
  void SetUp() override {
    // Create a sample factory.
    ASSERT_OK_AND_ASSIGN(
        FixedPointElementFactory temp,
        FixedPointElementFactory::Create(
            kSampleParams.num_fractional_bits,
            kSampleParams.num_ring_bits));
    self_fpe_factory_ = absl::make_unique<FixedPointElementFactory>(
        std::move(temp));
  }
 protected:
  std::unique_ptr<FixedPointElementFactory> self_fpe_factory_;
};

TEST_F(FixedPointElementFactoryCreateElementTest,
       TestInvalidPositiveElementCreate) {
  // 2^{22} for 21 (signed) integer bits is invalid.
  EXPECT_THAT(self_fpe_factory_->CreateFixedPointElementFromInt(
                           kSampleParams.integer_ring_modulus),
              StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid value")));
}

TEST_F(FixedPointElementFactoryCreateElementTest,
       TestInvalidNegativeDoubleCreate) {
  // -2^{22} for 21 (signed) integer bits is invalid.
  EXPECT_THAT(self_fpe_factory_->CreateFixedPointElementFromDouble(
                           -1 * static_cast<double>(
                               kSampleParams.integer_ring_modulus)),
              StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid value")));
}

TEST_F(FixedPointElementFactoryCreateElementTest,
       TestInvalidImportElementFromUint64) {
  // 2^{32} unsigned value in 32-bit ring is invalid.
  EXPECT_THAT(self_fpe_factory_->ImportFixedPointElementFromUint64(kTwoPower32),
              StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
                      HasSubstr("Invalid value")));
}

TEST_F(FixedPointElementFactoryCreateElementTest, TestValidUint64Import) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_fpe_factory_->ImportFixedPointElementFromUint64(
                       kSampleUintVal));
  EXPECT_EQ(fpe.ExportToUint64(), kSampleUintVal);
}

TEST_F(FixedPointElementFactoryCreateElementTest,
       TestValidPositiveCreateFromDouble) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_fpe_factory_->CreateFixedPointElementFromDouble(
                       kSampleFixedPointDouble));
  // fpe should represent 1.5 in ring with params {10, 32}
  // which is equal to 1536 = kSampleUintVal.
  EXPECT_EQ(fpe.ExportToUint64(), kSampleUintVal);
}

TEST_F(FixedPointElementFactoryCreateElementTest,
       TestValidNegativeCreateFromDouble) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_fpe_factory_->CreateFixedPointElementFromDouble(
                       kSampleNegativeDouble));
  // fpe should represent -1.5 in ring with params {10, 32}
  // which is equal to 2^{32} - 1536 = 4294965760 = kSampleNegativeUintVal.
  EXPECT_EQ(fpe.ExportToUint64(), kSampleNegativeUintVal);
}

TEST_F(FixedPointElementFactoryCreateElementTest,
       TestCreateFromSmallDoubleResultsInZero) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_fpe_factory_->CreateFixedPointElementFromDouble(
                           0.0001));
  // 0.0001 < 1/1024 and so should be represented as 0.
  EXPECT_EQ(fpe.ExportToUint64(), 0);
}

TEST_F(FixedPointElementFactoryCreateElementTest,
       TestValidPositiveCreateFromInt) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_fpe_factory_->CreateFixedPointElementFromInt(1));
  // fpe should represent 1 in ring with params {10, 32}
  // which is equal to 1024.
  EXPECT_EQ(fpe.ExportToUint64(), 1024);
}

TEST_F(FixedPointElementFactoryCreateElementTest,
       TestValidNegativeCreateFromInt) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       self_fpe_factory_->CreateFixedPointElementFromInt(-1));
  // fpe should represent -1 in ring with params {10, 32}
  // which is equal to 2^{32} - 1024 = 4294966272.
  EXPECT_EQ(fpe.ExportToUint64(), 4294966272);
}

TEST_F(FixedPointElementFactoryCreateElementTest, TestSameFactory) {
  ASSERT_OK_AND_ASSIGN(FixedPointElementFactory other_fpe_factory,
                       FixedPointElementFactory::Create(
                           kSampleParams.num_fractional_bits,
                           kSampleParams.num_ring_bits));
  EXPECT_TRUE(IsSameFactoryAs(other_fpe_factory));
}

}  // namespace
}  // namespace private_join_and_compute
