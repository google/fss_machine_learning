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

#include "secret_sharing_mpc/arithmetic/fixed_point_arithmetic.h"

#include <utility>

#include "poisson_regression/prng/basic_rng.h"
#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace private_join_and_compute {
namespace {

using ::testing::HasSubstr;
using testing::StatusIs;
using ::testing::Test;

const size_t kNumFractionalBits = 10;
const size_t kNumRingBits = 63;

// {10, 63, 2^10, 2^53, 2^63}
const FixedPointElementFactory::Params kSampleParams = {
    kNumFractionalBits, kNumRingBits, (1ULL << kNumFractionalBits),
    (1ULL << (kNumRingBits - kNumFractionalBits)), (1ULL << kNumRingBits)};

class FixedPointArithmeticTest : public Test {
 protected:
  void SetUp() override {
    // Create a sample 63-bit factory.
    ASSERT_OK_AND_ASSIGN(
        FixedPointElementFactory temp63,
        FixedPointElementFactory::Create(kSampleParams.num_fractional_bits,
                                         kSampleParams.num_ring_bits));
    fp_factory_ =
        absl::make_unique<FixedPointElementFactory>(std::move(temp63));
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
// For matrices A, B, we get C = A * B

TEST_F(FixedPointArithmeticTest, TruncMatrixMulFPEmptyVectorFails) {
  std::vector<FixedPointElement> empty_vector;
  std::vector<FixedPointElement> small_vector{
      fp_factory_->CreateFixedPointElementFromDouble(0.0).value(),
      fp_factory_->CreateFixedPointElementFromDouble(1.0).value()};
  auto product =
      TruncMatrixMulFP(empty_vector, small_vector, 0, 2, 1, fp_factory_);
  EXPECT_THAT(
      product,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("TruncMatrixMulFP: input must not be empty.")));
}

TEST_F(FixedPointArithmeticTest, TruncMatrixMulFPWrongDimensionsFails) {
  ASSERT_OK_AND_ASSIGN(auto zero,
                       fp_factory_->CreateFixedPointElementFromDouble(0.0));
  ASSERT_OK_AND_ASSIGN(auto one,
                       fp_factory_->CreateFixedPointElementFromDouble(1.0));
  std::vector<FixedPointElement> small_vector_one{zero, one};
  std::vector<FixedPointElement> small_vector_two{one, zero};
  // dimension mismatch vector one
  auto product_one = TruncMatrixMulFP(small_vector_one, small_vector_two, 2, 2,
                                      1, fp_factory_);
  // dimension mismatch vector two
  auto product_two = TruncMatrixMulFP(small_vector_one, small_vector_two, 1, 2,
                                      2, fp_factory_);
  EXPECT_THAT(
      product_one,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("TruncMatrixMulFP: invalid matrix dimension.")));
  EXPECT_THAT(
      product_two,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("TruncMatrixMulFP: invalid matrix dimension.")));
}

// End-to-end test for fixed-point matrix product.
// Input: A, B fixed-point matrices
// Output: A * B where * represents matrix product mod modulus.
// Tests all combinations: ++/+-/-+/--, multiplication by 0, 0 * 0
TEST_F(FixedPointArithmeticTest, TruncMatrixMulFPSucceeds) {
  // fpe = 1.5 and represented as 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe,
                       fp_factory_->CreateFixedPointElementFromDouble(1.5));
  // fpe_negate = -1.5 and represented as kRingModulus - 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe_negate, fpe.Negate());
  // zero = 0 and represented as 0*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto zero,
                       fp_factory_->CreateFixedPointElementFromDouble(0));
  size_t dim1 = 2;
  size_t dim2 = 3;
  size_t dim3 = 2;
  // Initialize input vectors in the clear.
  // A = [0    1.5  -1.5].
  //     [-1.5   0   1.5].
  std::vector<FixedPointElement> fpe_a{zero,       fpe,  fpe_negate,
                                       fpe_negate, zero, fpe};
  // B = [ 1.5  -1.5].
  //     [-1.5     0].
  //     [ 1.5   1.5].
  std::vector<FixedPointElement> fpe_b{fpe,  fpe_negate, fpe_negate,
                                       zero, fpe,        fpe};

  // Expected output (X*Y) in the clear.
  ASSERT_OK_AND_ASSIGN(
      std::vector<FixedPointElement> fpe_ab,
      TruncMatrixMulFP(fpe_a, fpe_b, dim1, dim2, dim3, fp_factory_));

  // Test with hand-computed values
  std::vector<double> ab{-4.5, -2.25, 0, 4.5};
  for (size_t idx = 0; idx < dim1 * dim3; idx++) {
    // The error is at most 2^-lf per each of dim2 multiplications
    // (dim2 * 2^-10 in our case)
    EXPECT_NEAR(fpe_ab[idx].ExportToDouble(), ab[idx],
                dim2 * (1.0 / (1ULL << kNumFractionalBits)));
  }
}

}  // namespace
}  // namespace private_join_and_compute
