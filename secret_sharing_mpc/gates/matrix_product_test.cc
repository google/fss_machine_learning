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

#include "secret_sharing_mpc/gates/matrix_product.h"

#include <utility>

#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status_testing.inc"
#include "secret_sharing_mpc/arithmetic/fixed_point_arithmetic.h"
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

class MatrixProductTest : public Test {
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

// Test that secure matrix product works correctly.
// For matrices X, Y, we get Z = X * Y

// End-to-end test for secure matrix multiplication gate.
// Each party has input: share of vectors [X], [Y] and beaver triples
// ([A], [B], [C]). The output is share of vector [X*Y] where * represents
// matrix product mod modulus.
TEST_F(MatrixProductTest, SecureMatrixProductSucceeds) {
  // fpe = 1.5 and represented as 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(
      auto fpe, fp_factory_->CreateFixedPointElementFromDouble(1.5));
  // fpe_negate = -1.5 and represented as kRingModulus - 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe_negate, fpe.Negate());
  // zero = 0 and represented as 0*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto zero,
                       fp_factory_->CreateFixedPointElementFromInt(0));
  size_t dim1 = 2;
  size_t dim2 = 3;
  size_t dim3 = 2;
  // Initialize input vectors in the clear.
  // X = [0    1.5  -1.5].
  //     [-1.5   0   1.5].
  std::vector<FixedPointElement> fpe_x{zero, fpe, fpe_negate, fpe_negate,
                                       zero, fpe};
  // Y = [ 1.5  -1.5].
  //     [-1.5     0].
  //     [ 1.5   1.5].
  std::vector<FixedPointElement> fpe_y{fpe, fpe_negate, fpe_negate, zero,
                                       fpe, fpe};

  // Make input matrices X and Y from fixed point numbers.
  // X = {0, 1.5, -1.5, -1.5, 0, 1.5}*2^{10}
  // Y = {1.5, -1.5, -1.5, 0, 1.5, 1.5}*2^{10}
  std::vector<uint64_t> matrix_x;
  std::vector<uint64_t> matrix_y;
  for (size_t idx = 0; idx < fpe_x.size(); idx++) {
    matrix_x.push_back(fpe_x[idx].ExportToUint64());
  }
  for (size_t idx = 0; idx < fpe_y.size(); idx++) {
    matrix_y.push_back(fpe_y[idx].ExportToUint64());
  }

  // Matrix multiplication on share of X, Y.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, MakePrng());
  ASSERT_OK_AND_ASSIGN(auto beaver_matrix_shares,
                       SampleBeaverTripleMatrix(
                           dim1, dim2, dim3, kRingModulus));
  auto beaver_matrix_share_0 = beaver_matrix_shares.first;
  auto beaver_matrix_share_1 = beaver_matrix_shares.second;

  // Generate random shares for matrix x and y and distribute them to P0 and P1.
  ASSERT_OK_AND_ASSIGN(
      auto share_x_0,
      SampleVectorFromPrng(dim1 * dim2, kRingModulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(auto share_x_1,
                       BatchedModSub(matrix_x, share_x_0, kRingModulus));
  ASSERT_OK_AND_ASSIGN(
      auto share_y_0,
      SampleVectorFromPrng(dim2 * dim3, kRingModulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(auto share_y_1,
                       BatchedModSub(matrix_y, share_y_0, kRingModulus));

  // Each party generates its matrix multiplication message.
  ASSERT_OK_AND_ASSIGN(
      auto p0_return,
      GenerateMatrixProductMessage(
          share_x_0, share_y_0, beaver_matrix_share_0, kRingModulus));
  ASSERT_OK_AND_ASSIGN(
      auto p1_return,
      GenerateMatrixProductMessage(
          share_x_1, share_y_1, beaver_matrix_share_1, kRingModulus));
  auto state0 = p0_return.first;
  auto matrix_mult_msg_0 = p0_return.second;
  auto state1 = p1_return.first;
  auto matrix_mult_msg_1 = p1_return.second;

  // Each party computes its share of output from its input, Beaver triple
  // matrix, and other party's message.
  ASSERT_OK_AND_ASSIGN(
      auto share_xy_0,
      MatrixProductPartyZero(state0, beaver_matrix_share_0, matrix_mult_msg_1,
                             kNumFractionalBits, kRingModulus));
  ASSERT_OK_AND_ASSIGN(
      auto share_xy_1,
      MatrixProductPartyOne(state1, beaver_matrix_share_1, matrix_mult_msg_0,
                            kNumFractionalBits, kRingModulus));

  // Reconstruct output and verify the correctness.
  // NOTE: This is to test matrix multiplication for integers.
  ASSERT_OK_AND_ASSIGN(auto computed_xy,
                       BatchedModAdd(share_xy_0, share_xy_1, kRingModulus));

  // Test with hand-computed values for certainty
  ASSERT_OK_AND_ASSIGN(auto out_one,
                       fp_factory_->CreateFixedPointElementFromDouble(-4.5));
  ASSERT_OK_AND_ASSIGN(auto out_two,
                       fp_factory_->CreateFixedPointElementFromDouble(-2.25));
  ASSERT_OK_AND_ASSIGN(auto out_three,
                       fp_factory_->CreateFixedPointElementFromInt(0));
  ASSERT_OK_AND_ASSIGN(auto out_four,
                       fp_factory_->CreateFixedPointElementFromDouble(4.5));
  std::vector<FixedPointElement> expected_xy{out_one, out_two, out_three,
                                             out_four};
  for (size_t idx = 0; idx < dim1 * dim3; idx++) {
    // The error is at most 2^-lf
    // (2^-10 in our case which is 2^10 * 2^-10 = 1 in the ring)
    EXPECT_NEAR(computed_xy[idx], expected_xy[idx].ExportToUint64(), 1);
  }
}

}  // namespace
}  // namespace private_join_and_compute
