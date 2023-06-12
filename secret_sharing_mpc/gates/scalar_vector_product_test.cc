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

#include "secret_sharing_mpc/gates/scalar_vector_product.h"

#include <cstdint>
#include <utility>

#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/prng/basic_rng.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace private_join_and_compute {
namespace {
using ::testing::HasSubstr;
using testing::StatusIs;
using ::testing::Test;

const size_t kNumFractionalBits = 10;
const size_t kNumRingBits = 32;
const uint64_t kRingModulus = (1ULL << 32);

// {10, 32, 2^10, 2^22, 2^32}
const FixedPointElementFactory::Params kSampleParams = {
    kNumFractionalBits, kNumRingBits, (1ULL << kNumFractionalBits),
    (1ULL << (kNumRingBits - kNumFractionalBits)), (1ULL << kNumRingBits)};

class ScalarVectorProductTest : public Test {
 protected:
  void SetUp() override {
    // Create a sample 63-bit factory.
    ASSERT_OK_AND_ASSIGN(
        FixedPointElementFactory temp32,
        FixedPointElementFactory::Create(kSampleParams.num_fractional_bits,
                                         kSampleParams.num_ring_bits));
    fp_factory_ =
        absl::make_unique<FixedPointElementFactory>(std::move(temp32));
  }
  StatusOr<std::unique_ptr<BasicRng>> MakePrng() {
    auto random_seed = BasicRng::GenerateSeed();
    if (!random_seed.ok()) {
      return InternalError("Random seed generation fails.");
    }
    return BasicRng::Create(random_seed.value());
  }
  std::unique_ptr<FixedPointElementFactory> fp_factory_;
};

// In this file, test that secure scalar vector product works correctly.
// I.e. for PUBLIC scalar a and SECRET-SHARED vector B = [b0 b1 ... bn],
// we get Z = [ab0 ab1 ... abn]
// Note that this operation is done locally by each player.
// Hence, there is no test pertaining to preprocessing.
// Due to truncation, each element of the result can have error of at most:
// 2^-kNumFractionalBits

// Test that passing empty vector to scalar vector product will result in error
TEST_F(ScalarVectorProductTest, SecureScalarVectorProductEmptyVectorFails) {
  uint64_t scalar_a = 0;
  std::vector<uint64_t> empty_vector_b;
  auto res_one = ScalarVectorProductPartyZero(scalar_a, empty_vector_b,
                                              fp_factory_, kRingModulus);
  auto res_two = ScalarVectorProductPartyOne(scalar_a, empty_vector_b,
                                             fp_factory_, kRingModulus);
  EXPECT_THAT(
      res_one,
      StatusIs(
          StatusCode::kInvalidArgument,
          HasSubstr("ScalarVectorProduct: input vector must not be empty.")));
  EXPECT_THAT(
      res_two,
      StatusIs(
          StatusCode::kInvalidArgument,
          HasSubstr("ScalarVectorProduct: input vector must not be empty.")));
}

// End-to-end test for secure scalar vector product.
// scalar a is publicly known and is positive
// Each party has secret share of vector [B] (also matrix flattened into vector)
// The output is a share of vector [aB]
// Multiply 2.25 * (-2.75, 0, 2.75)
// Tests + * -, + * 0, and + * +
TEST_F(ScalarVectorProductTest,
       SecureScalarVectorProductPositiveScalarSucceeds) {
  double scalar_a = 2.25;

  // fpe_share = 2.75
  // Sum of these two numbers (2147483648, 2147486464) equals 2816 (mod 2^32)
  // 2816 is a fixed point representation of 2.75 (2816 = 2.75 * 2^10)
  ASSERT_OK_AND_ASSIGN(
      auto fpe_share_0,
      fp_factory_->ImportFixedPointElementFromUint64(2147483648));
  ASSERT_OK_AND_ASSIGN(
      auto fpe_share_1,
      fp_factory_->ImportFixedPointElementFromUint64(2147486464));
  // fpe_negate = -2.75
  ASSERT_OK_AND_ASSIGN(auto negative_fpe_share_0, fpe_share_0.Negate());
  ASSERT_OK_AND_ASSIGN(auto negative_fpe_share_1, fpe_share_1.Negate());
  // zero = 0
  ASSERT_OK_AND_ASSIGN(
      auto zero_share_0,
      fp_factory_->ImportFixedPointElementFromUint64(33554432));
  ASSERT_OK_AND_ASSIGN(
      auto zero_share_1,
      fp_factory_->ImportFixedPointElementFromUint64(4261412864));

  // Initialize input vector B = [-2.75, 0, 2.75].
  std::vector<FixedPointElement> fpe_b_share_0{negative_fpe_share_0,
                                               zero_share_0, fpe_share_0};
  std::vector<FixedPointElement> fpe_b_share_1{negative_fpe_share_1,
                                               zero_share_1, fpe_share_1};
  std::vector<uint64_t> vector_b_share_0;
  std::vector<uint64_t> vector_b_share_1;
  for (size_t idx = 0; idx < fpe_b_share_0.size(); idx++) {
    vector_b_share_0.push_back(fpe_b_share_0[idx].ExportToUint64());
    vector_b_share_1.push_back(fpe_b_share_1[idx].ExportToUint64());
  }

  // Expected output (aB) in the clear.
  ASSERT_OK_AND_ASSIGN(auto out_one,
                       fp_factory_->CreateFixedPointElementFromDouble(-6.1875));
  ASSERT_OK_AND_ASSIGN(auto out_two,
                       fp_factory_->CreateFixedPointElementFromInt(0));
  ASSERT_OK_AND_ASSIGN(auto out_three, out_one.Negate());
  std::vector<FixedPointElement> expected_ab{out_one, out_two, out_three};

  // Share of party 0
  ASSERT_OK_AND_ASSIGN(auto res_share_0,
                       ScalarVectorProductPartyZero(scalar_a, vector_b_share_0,
                                                    fp_factory_, kRingModulus));
  // Share of party 1
  ASSERT_OK_AND_ASSIGN(auto res_share_1,
                       ScalarVectorProductPartyOne(scalar_a, vector_b_share_1,
                                                   fp_factory_, kRingModulus));
  // Reconstruct the output
  ASSERT_OK_AND_ASSIGN(auto res,
                       BatchedModAdd(res_share_0, res_share_1, kRingModulus));

  // Check output
  for (size_t idx = 0; idx < expected_ab.size(); idx++) {
    // Error can be at most 2^-10, in ring this is 2^-10 * 2^10 = 1
    EXPECT_NEAR(res[idx], expected_ab[idx].ExportToUint64(), 1);
  }
}

// End-to-end test for secure scalar vector product.
// scalar a is publicly known and is negative
// Each party has secret share of vector [B] (also matrix flattened into vector)
// The output is a share of vector [aB]
// Multiply -2.25 * (-2.75, 0, 2.75)
// Tests - * -, - * 0, and - * +
TEST_F(ScalarVectorProductTest,
       SecureScalarVectorProductNegativeScalarSucceeds) {
  double scalar_a = -2.25;

  // fpe_share = 2.75
  // Sum of these two numbers (2147483648, 2147486464) equals 2816 (mod 2^32)
  // 2816 is a fixed point representation of 2.75 (2816 = 2.75 * 2^10)
  ASSERT_OK_AND_ASSIGN(
      auto fpe_share_0,
      fp_factory_->ImportFixedPointElementFromUint64(2147483648));
  ASSERT_OK_AND_ASSIGN(
      auto fpe_share_1,
      fp_factory_->ImportFixedPointElementFromUint64(2147486464));
  // fpe_negate = -2.75
  ASSERT_OK_AND_ASSIGN(auto negative_fpe_share_0, fpe_share_0.Negate());
  ASSERT_OK_AND_ASSIGN(auto negative_fpe_share_1, fpe_share_1.Negate());
  // zero = 0
  ASSERT_OK_AND_ASSIGN(
      auto zero_share_0,
      fp_factory_->ImportFixedPointElementFromUint64(33554432));
  ASSERT_OK_AND_ASSIGN(
      auto zero_share_1,
      fp_factory_->ImportFixedPointElementFromUint64(4261412864));

  // Initialize input vector B = [-2.75, 0, 2.75].
  std::vector<FixedPointElement> fpe_b_share_0{negative_fpe_share_0,
                                               zero_share_0, fpe_share_0};
  std::vector<FixedPointElement> fpe_b_share_1{negative_fpe_share_1,
                                               zero_share_1, fpe_share_1};
  std::vector<uint64_t> vector_b_share_0;
  std::vector<uint64_t> vector_b_share_1;
  for (size_t idx = 0; idx < fpe_b_share_0.size(); idx++) {
    vector_b_share_0.push_back(fpe_b_share_0[idx].ExportToUint64());
    vector_b_share_1.push_back(fpe_b_share_1[idx].ExportToUint64());
  }

  // Expected output (aB) in the clear.
  ASSERT_OK_AND_ASSIGN(auto out_one,
                       fp_factory_->CreateFixedPointElementFromDouble(6.1875));
  ASSERT_OK_AND_ASSIGN(auto out_two,
                       fp_factory_->CreateFixedPointElementFromInt(0));
  ASSERT_OK_AND_ASSIGN(auto out_three, out_one.Negate());
  std::vector<FixedPointElement> expected_ab{out_one, out_two, out_three};

  // Share of party 0
  ASSERT_OK_AND_ASSIGN(auto res_share_0,
                       ScalarVectorProductPartyZero(scalar_a, vector_b_share_0,
                                                    fp_factory_, kRingModulus));
  // Share of party 1
  ASSERT_OK_AND_ASSIGN(auto res_share_1,
                       ScalarVectorProductPartyOne(scalar_a, vector_b_share_1,
                                                   fp_factory_, kRingModulus));
  // Reconstruct the output
  ASSERT_OK_AND_ASSIGN(auto res,
                       BatchedModAdd(res_share_0, res_share_1, kRingModulus));

  // Check output
  for (size_t idx = 0; idx < expected_ab.size(); idx++) {
    // Error can be at most 2^-10, in ring this is 2^-10 * 2^10 = 1
    EXPECT_NEAR(res[idx], expected_ab[idx].ExportToUint64(), 1);
  }
}

}  // namespace
}  // namespace private_join_and_compute
