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

#include "secret_sharing_mpc/gates/hadamard_product.h"

#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace private_join_and_compute {
namespace {
using ::testing::Test;

const size_t kNumFractionalBits = 10;
const size_t kNumRingBits = 63;
const uint64_t kRingModulus = (1ULL << 63);

// {10, 63, 2^10, 2^53, 2^63}
const FixedPointElementFactory::Params kSampleParams
    = {kNumFractionalBits, kNumRingBits, (1ULL << kNumFractionalBits),
       (1ULL << (kNumRingBits - kNumFractionalBits)), (1ULL << kNumRingBits)};

class HadamardProductTest : public Test {
 protected:
  void SetUp() override {
    // Create a sample 63-bit factory.
    ASSERT_OK_AND_ASSIGN(
        FixedPointElementFactory temp63,
        FixedPointElementFactory::Create(
            kSampleParams.num_fractional_bits,
            kSampleParams.num_ring_bits));
    fp_factory_ = absl::make_unique<FixedPointElementFactory>(
        std::move(temp63));
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

// Test that secure hadamard product works correctly.

// End-to-end test for secure batched hadamard product.
// Each party has input: share of vectors [X], [Y] and beaver triples
// ([A], [B], [C]). The output is share of vector [X*Y] where * represents
// element-wise multiplication mod modulus.
// In this test, X = {0,   0,    0, 1.5, -1.5, 1.5,  1.5, -1.5, -1.5}*2^{10}
// and           Y = {0, 1.5, -1.5,   0,    0, 1.5, -1.5,  1.5, -1.5}*2^{10}.
// We test all the combinations:
// (0, 0), (0, +), (0, -), (+, 0), (-, 0), (+, +), (+, -), (-, +), (-, -).
TEST_F(HadamardProductTest, SecureHadamardProductSucceeds) {
  // fpe = 1.5 and represented as 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe,
                       fp_factory_->CreateFixedPointElementFromDouble(1.5));
  // fpe_negate = -1.5 and represented as kRingModulus - 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe_negate, fpe.Negate());
  // zero = 0 and represented as 0*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto zero,
                       fp_factory_->CreateFixedPointElementFromDouble(0));
  // Initialize input vectors in the clear.
  std::vector<FixedPointElement> fpe_x{zero, zero, zero, fpe, fpe_negate, fpe,
                                       fpe, fpe_negate, fpe_negate};
  std::vector<FixedPointElement> fpe_y{zero, fpe, fpe_negate, zero, zero, fpe,
                                       fpe_negate, fpe, fpe_negate};
  EXPECT_EQ(fpe_x.size(), fpe_y.size());

  size_t length = fpe_x.size();
  // Representation of X and Y in ring Z_kRingModulus.
  // X = {0,   0,    0, 1.5, -1.5, 1.5,  1.5, -1.5, -1.5}*2^{10}
  // Y = {0, 1.5, -1.5,   0,    0, 1.5, -1.5,  1.5, -1.5}*2^{10}
  std::vector<uint64_t> vector_x;
  std::vector<uint64_t> vector_y;
  for (size_t idx = 0; idx < length; idx++) {
    vector_x.push_back(fpe_x[idx].ExportToUint64());
    vector_y.push_back(fpe_y[idx].ExportToUint64());
  }
  // Expected output xy in the clear.
  ASSERT_OK_AND_ASSIGN(auto fpe_squared,
                       fp_factory_->CreateFixedPointElementFromDouble(2.25));
  ASSERT_OK_AND_ASSIGN(auto fpe_squared_negate, fpe_squared.Negate());
  std::vector<FixedPointElement> expected_xy{zero,
                                             zero,
                                             zero,
                                             zero,
                                             zero,
                                             fpe_squared,
                                             fpe_squared_negate,
                                             fpe_squared_negate,
                                             fpe_squared};

  // Batched multiplication on share of X and Y.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, MakePrng());

  // Generate Beaver triple vector for P0 and P1.
  ASSERT_OK_AND_ASSIGN(auto beaver_vector_shares,
                       SampleBeaverTripleVector(length, kRingModulus));
  auto beaver_vector_share_0 = beaver_vector_shares.first;
  auto beaver_vector_share_1 = beaver_vector_shares.second;

  // Generate random shares for vector x and y and distribute to P0 and P1.
  ASSERT_OK_AND_ASSIGN(auto share_x_0,
                       SampleVectorFromPrng(length, kRingModulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(auto share_x_1,
                       BatchedModSub(vector_x, share_x_0, kRingModulus));
  ASSERT_OK_AND_ASSIGN(auto share_y_0,
                       SampleVectorFromPrng(length, kRingModulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(auto share_y_1,
                       BatchedModSub(vector_y, share_y_0, kRingModulus));

  // Each party generates its batched multiplication message.
  ASSERT_OK_AND_ASSIGN(
      auto p0_return,
      GenerateHadamardProductMessage(
          share_x_0, share_y_0, beaver_vector_share_0, kRingModulus));
  auto state0 = p0_return.first;
  auto batched_mult_msg_0 = p0_return.second;
  ASSERT_OK_AND_ASSIGN(
      auto p1_return,
      GenerateHadamardProductMessage(
          share_x_1, share_y_1, beaver_vector_share_1, kRingModulus));
  auto state1 = p1_return.first;
  auto batched_mult_msg_1 = p1_return.second;
  // Each party computes its share of output.
  ASSERT_OK_AND_ASSIGN(auto share_xy_0,
                       HadamardProductPartyZero(
                           state0, beaver_vector_share_0, batched_mult_msg_1,
                           kNumFractionalBits, kRingModulus));
  ASSERT_OK_AND_ASSIGN(
      auto share_xy_1,
      HadamardProductPartyOne(state1, beaver_vector_share_1, batched_mult_msg_0,
                              kNumFractionalBits, kRingModulus));

  // NOTE: This is to test batched multiplication for integers.
  // Reconstruct output and verify the correctness.
  ASSERT_OK_AND_ASSIGN(auto computed_xy,
                       BatchedModAdd(share_xy_0, share_xy_1, kRingModulus));
  for (size_t idx = 0; idx < length; idx++) {
    // The error is at most 2^-lf (for each output element)
    // (2^-10 in our case which is 2^10 * 2^-10 = 1 in the ring)
    EXPECT_NEAR(computed_xy[idx], expected_xy[idx].ExportToUint64(), 1);
  }
}

// This test checks whether Hadamard Product works for X * Y when:
// X is a secret-sharing of a positive integer NOT in our fixed-point ring
// representation i.e. X_0 and X_1 are not multiplied by 2^num_fractional bits
// The shares of [X] are converted into fixed point ring after being
// secret-shared Y is a secret-sharing in the fixed point ring
// This is one of the settings in logistic regression.
TEST_F(HadamardProductTest,
       SecureHadamardProductMultiplicandNotFixedPointRingSucceeds) {
  // fpe = 1.5 and represented as 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe_one,
                       fp_factory_->CreateFixedPointElementFromInt(1));
  ASSERT_OK_AND_ASSIGN(auto fpe_five,
                       fp_factory_->CreateFixedPointElementFromInt(5));
  // zero = 0 and represented as 0*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto zero,
                       fp_factory_->CreateFixedPointElementFromDouble(0));
  // Initialize input vectors in the clear.
  std::vector<FixedPointElement> fpe_y{zero, fpe_one, fpe_five};

  size_t length = fpe_y.size();
  // Representation of X and Y in ring Z_kRingModulus.
  // X = {0,   0,    0, 1.5, -1.5, 1.5,  1.5, -1.5, -1.5}*2^{10}
  // Y = {0, 1.5, -1.5,   0,    0, 1.5, -1.5,  1.5, -1.5}*2^{10}
  std::vector<uint64_t> vector_x = {0, 1, 5};
  EXPECT_EQ(vector_x.size(), fpe_y.size());
  std::vector<uint64_t> vector_y;
  for (size_t idx = 0; idx < length; idx++) {
    vector_y.push_back(fpe_y[idx].ExportToUint64());
  }
  EXPECT_EQ(vector_x.size(), vector_y.size());
  // Expected output xy in the clear.
  ASSERT_OK_AND_ASSIGN(auto fpe_twentyfive,
                       fp_factory_->CreateFixedPointElementFromInt(25));
  std::vector<FixedPointElement> expected_xy{zero, fpe_one, fpe_twentyfive};

  // Batched multiplication on share of X and Y.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, MakePrng());

  // Generate Beaver triple vector for P0 and P1.
  ASSERT_OK_AND_ASSIGN(auto beaver_vector_shares,
                       SampleBeaverTripleVector(length, kRingModulus));
  auto beaver_vector_share_0 = beaver_vector_shares.first;
  auto beaver_vector_share_1 = beaver_vector_shares.second;

  // Generate random shares for vector x and y and distribute to P0 and P1.
  ASSERT_OK_AND_ASSIGN(auto share_x_0,
                       SampleVectorFromPrng(length, kRingModulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(auto share_x_1,
                       BatchedModSub(vector_x, share_x_0, kRingModulus));
  // Multiply shares of X by 2^lf
  std::vector<uint64_t> vector_ones = {(1ULL << kNumFractionalBits),
                                       (1ULL << kNumFractionalBits),
                                       (1ULL << kNumFractionalBits)};
  ASSERT_OK_AND_ASSIGN(share_x_0,
                       BatchedModMul(share_x_0, vector_ones, kRingModulus));
  ASSERT_OK_AND_ASSIGN(share_x_1,
                       BatchedModMul(share_x_1, vector_ones, kRingModulus));

  ASSERT_OK_AND_ASSIGN(auto share_y_0,
                       SampleVectorFromPrng(length, kRingModulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(auto share_y_1,
                       BatchedModSub(vector_y, share_y_0, kRingModulus));

  // Each party generates its batched multiplication message.
  ASSERT_OK_AND_ASSIGN(
      auto p0_return,
      GenerateHadamardProductMessage(share_x_0, share_y_0,
                                     beaver_vector_share_0, kRingModulus));
  auto state0 = p0_return.first;
  auto batched_mult_msg_0 = p0_return.second;
  ASSERT_OK_AND_ASSIGN(
      auto p1_return,
      GenerateHadamardProductMessage(share_x_1, share_y_1,
                                     beaver_vector_share_1, kRingModulus));
  auto state1 = p1_return.first;
  auto batched_mult_msg_1 = p1_return.second;
  // Each party computes its share of output.
  ASSERT_OK_AND_ASSIGN(auto share_xy_0,
                       HadamardProductPartyZero(
                           state0, beaver_vector_share_0, batched_mult_msg_1,
                           kNumFractionalBits, kRingModulus));
  ASSERT_OK_AND_ASSIGN(
      auto share_xy_1,
      HadamardProductPartyOne(state1, beaver_vector_share_1, batched_mult_msg_0,
                              kNumFractionalBits, kRingModulus));

  // NOTE: This is to test batched multiplication for integers.
  // Reconstruct output and verify the correctness.
  ASSERT_OK_AND_ASSIGN(auto computed_xy,
                       BatchedModAdd(share_xy_0, share_xy_1, kRingModulus));
  for (size_t idx = 0; idx < length; idx++) {
    // The error is at most 2^-lf (for each output element)
    // (2^-10 in our case which is 2^10 * 2^-10 = 1 in the ring)
    EXPECT_NEAR(computed_xy[idx], expected_xy[idx].ExportToUint64(), 1);
  }
}

}  // namespace
}  // namespace private_join_and_compute

