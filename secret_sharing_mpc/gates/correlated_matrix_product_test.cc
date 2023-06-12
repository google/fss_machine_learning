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

#include "secret_sharing_mpc/gates/correlated_matrix_product.h"

#include <tuple>
#include <utility>

#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status_testing.h"
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
const uint64_t kRingModulus = (1ULL << 63);

// {10, 63, 2^10, 2^53, 2^63}
const FixedPointElementFactory::Params kSampleParams = {
    kNumFractionalBits, kNumRingBits, (1ULL << kNumFractionalBits),
    (1ULL << (kNumRingBits - kNumFractionalBits)), (1ULL << kNumRingBits)};

class CorrelatedMatrixProductTest : public Test {
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

TEST_F(CorrelatedMatrixProductTest,
       SampleBeaverTripleMatrixAInvalidDimensionsFails) {
  size_t dim1 = 0;
  size_t dim2 = 1;
  auto vector_a = SampleBeaverTripleMatrixA(dim1, dim2, kRingModulus);
  EXPECT_THAT(vector_a,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("SampleBeaverTripleMatrixA: length of vector "
                                 "must be a positive integer")));
}

TEST_F(CorrelatedMatrixProductTest, SampleBeaverTripleMatrixASucceeds) {
  size_t dim1 = 2;
  size_t dim2 = 3;
  ASSERT_OK_AND_ASSIGN(auto vector_a,
                       SampleBeaverTripleMatrixA(dim1, dim2, kRingModulus));
  // Check dimensions of the shares of A are appropriate
  ASSERT_EQ(std::get<2>(vector_a).size(), dim1 * dim2);
  ASSERT_EQ(std::get<0>(vector_a).size(), std::get<2>(vector_a).size());
  ASSERT_EQ(std::get<1>(vector_a).size(), std::get<2>(vector_a).size());

  // Check the shares of A add to A
  ASSERT_OK_AND_ASSIGN(auto reconstructed_a,
                       BatchedModAdd(std::get<0>(vector_a),
                                     std::get<1>(vector_a), kRingModulus));
  for (size_t idx = 0; idx < (dim1 * dim2); idx++) {
    EXPECT_EQ(reconstructed_a[idx], std::get<2>(vector_a)[idx]);
  }
}

TEST_F(CorrelatedMatrixProductTest,
       SampleBeaverTripleMatrixBandCInvalidDimensionsFails) {
  size_t dim1 = 2;
  size_t dim2 = 2;
  size_t dim3 = 0;
  ASSERT_OK_AND_ASSIGN(auto vector_a,
                       SampleBeaverTripleMatrixA(dim1, dim2, kRingModulus));
  auto beaver_shares =
      SampleBeaverTripleMatrixBandC(vector_a, dim1, dim2, dim3, kRingModulus);
  EXPECT_THAT(beaver_shares,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("SampleBeaverTripleMatrixBandC: length of "
                                 "vector must be a positive integer")));
}

TEST_F(CorrelatedMatrixProductTest,
       GenerateMatrixXminusAProductMessageInvalidDimensionsFails) {
  std::vector<uint64_t> empty_vector(0);
  std::vector<uint64_t> small_vector{1, 2, 3, 4};
  auto mult_state_and_message = GenerateMatrixXminusAProductMessage(
      empty_vector, small_vector, kRingModulus);
  EXPECT_THAT(mult_state_and_message,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("input must not be empty")));
}

TEST_F(CorrelatedMatrixProductTest,
       GenerateMatrixXminusAProductMessageNotEqualLengthSharesFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4};
  std::vector<uint64_t> small_vector_2{1, 2, 3, 4, 5, 6};
  auto mult_state_and_message = GenerateMatrixXminusAProductMessage(
      small_vector_2, small_vector_1, kRingModulus);
  EXPECT_THAT(
      mult_state_and_message,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("input and beaver shares have different size.")));
}

TEST_F(CorrelatedMatrixProductTest,
       GenerateMatrixYminusBProductMessageInvalidDimensionsFails) {
  std::vector<uint64_t> empty_vector(0);
  std::vector<uint64_t> small_vector{1, 2, 3, 4};
  auto mult_state_and_message = GenerateMatrixYminusBProductMessage(
      empty_vector, small_vector, small_vector, kRingModulus);
  EXPECT_THAT(mult_state_and_message,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("input must not be empty")));
}

TEST_F(CorrelatedMatrixProductTest,
       GenerateMatrixYminusBProductMessageNotEqualLengthSharesFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4};
  std::vector<uint64_t> small_vector_2{1, 2, 3, 4, 5, 6};
  auto mult_state_and_message = GenerateMatrixYminusBProductMessage(
      small_vector_2, small_vector_2, small_vector_1, kRingModulus);
  EXPECT_THAT(
      mult_state_and_message,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("input and beaver shares have different size.")));
}

TEST_F(CorrelatedMatrixProductTest,
       CorrelatedMatrixProductPartyZeroEmptyStateFails) {
  std::vector<uint64_t> empty_vector(0);
  std::vector<uint64_t> small_vector{1, 2, 3, 4};
  MatrixMultState state = {empty_vector, empty_vector};
  MatrixXminusAProductMessage other_party_x_minus_a_message;
  MatrixYminusBProductMessage other_party_y_minus_b_message;
  auto out_share = CorrelatedMatrixProductPartyZero(
      state, small_vector, small_vector, small_vector,
      other_party_x_minus_a_message, other_party_y_minus_b_message, 2, 2, 2,
      kNumFractionalBits, kRingModulus);
  EXPECT_THAT(out_share, StatusIs(StatusCode::kInvalidArgument,
                                  HasSubstr("input must not be empty")));
}

TEST_F(CorrelatedMatrixProductTest,
       CorrelatedMatrixProductPartyZeroNotEqualLengthSharesFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4};
  std::vector<uint64_t> small_vector_2{1, 2, 3, 4, 5, 6};
  MatrixMultState state = {small_vector_2, small_vector_2};
  MatrixXminusAProductMessage other_party_x_minus_a_message;
  MatrixYminusBProductMessage other_party_y_minus_b_message;
  ASSERT_OK_AND_ASSIGN(auto beaver_matrix, BeaverTripleMatrix<uint64_t>::Create(
                                               small_vector_1, small_vector_1,
                                               small_vector_1, 2, 2, 2));
  auto out_share = CorrelatedMatrixProductPartyZero(
      state, small_vector_1, small_vector_1, small_vector_1,
      other_party_x_minus_a_message, other_party_y_minus_b_message, 2, 2, 2,
      kNumFractionalBits, kRingModulus);
  EXPECT_THAT(out_share,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("input and beaver share dimension mismatch")));
}

TEST_F(CorrelatedMatrixProductTest,
       CorrelatedMatrixProductPartyOneEmptyStateFails) {
  std::vector<uint64_t> empty_vector(0);
  std::vector<uint64_t> small_vector{1, 2, 3, 4};
  MatrixMultState state = {empty_vector, empty_vector};
  MatrixXminusAProductMessage other_party_x_minus_a_message;
  MatrixYminusBProductMessage other_party_y_minus_b_message;
  auto out_share = CorrelatedMatrixProductPartyOne(
      state, small_vector, small_vector, small_vector,
      other_party_x_minus_a_message, other_party_y_minus_b_message, 2, 2, 2,
      kNumFractionalBits, kRingModulus);
  EXPECT_THAT(out_share, StatusIs(StatusCode::kInvalidArgument,
                                  HasSubstr("input must not be empty")));
}

TEST_F(CorrelatedMatrixProductTest,
       CorrelatedMatrixProductPartyOneNotEqualLengthSharesFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4};
  std::vector<uint64_t> small_vector_2{1, 2, 3, 4, 5, 6};
  MatrixMultState state = {small_vector_2, small_vector_2};
  MatrixXminusAProductMessage other_party_x_minus_a_message;
  MatrixYminusBProductMessage other_party_y_minus_b_message;
  auto out_share = CorrelatedMatrixProductPartyOne(
      state, small_vector_1, small_vector_1, small_vector_1,
      other_party_x_minus_a_message, other_party_y_minus_b_message, 2, 2, 2,
      kNumFractionalBits, kRingModulus);
  EXPECT_THAT(out_share,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("input and beaver share dimension mismatch")));
}

// End-to-end test for secure correlated matrix multiplication gate.
// Each party has input: share of vectors [X], [Y] and beaver triples
// ([A], [B], [C]). The output is share of vector [X*Y] where * represents
// matrix product mod modulus.
// The key is that the mask A is treated separately from the Beaver triple,
// and hence can result in communication savings if the same matrix X multiplies
// multiple Y_i's.
TEST_F(CorrelatedMatrixProductTest,
       SecureCorrelatedMatrixSingleProductSucceeds) {
  // fpe = 1.5 and represented as 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe,
                       fp_factory_->CreateFixedPointElementFromDouble(1.5));
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
  std::vector<FixedPointElement> fpe_x{zero,       fpe,  fpe_negate,
                                       fpe_negate, zero, fpe};
  // Y = [ 1.5  -1.5].
  //     [-1.5     0].
  //     [ 1.5   1.5].
  std::vector<FixedPointElement> fpe_y{fpe,  fpe_negate, fpe_negate,
                                       zero, fpe,        fpe};

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

  // Matrix multiplication on share of X (correlated), Y.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, MakePrng());
  ASSERT_OK_AND_ASSIGN(auto beaver_matrix_a_shares,
                       SampleBeaverTripleMatrixA(dim1, dim2, kRingModulus));
  ASSERT_OK_AND_ASSIGN(
      auto beaver_matrix_b_c_shares,
      SampleBeaverTripleMatrixBandC(beaver_matrix_a_shares, dim1, dim2, dim3,
                                    kRingModulus));
  auto beaver_matrix_b_c_share_p0 = beaver_matrix_b_c_shares.first;
  auto beaver_matrix_b_c_share_p1 = beaver_matrix_b_c_shares.second;

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
      auto p0_part0_return,
      GenerateMatrixXminusAProductMessage(
          share_x_0, std::get<0>(beaver_matrix_a_shares), kRingModulus));
  ASSERT_OK_AND_ASSIGN(auto p0_part1_return,
                       GenerateMatrixYminusBProductMessage(
                           share_y_0, p0_part0_return.first,
                           beaver_matrix_b_c_share_p0.first, kRingModulus));
  ASSERT_OK_AND_ASSIGN(
      auto p1_part0_return,
      GenerateMatrixXminusAProductMessage(
          share_x_1, std::get<1>(beaver_matrix_a_shares), kRingModulus));
  ASSERT_OK_AND_ASSIGN(auto p1_part1_return,
                       GenerateMatrixYminusBProductMessage(
                           share_y_1, p1_part0_return.first,
                           beaver_matrix_b_c_share_p1.first, kRingModulus));

  auto state0 = p0_part1_return.first;
  auto matrix_mult_msg0_p0 = p0_part0_return.second;
  auto matrix_mult_msg1_p0 = p0_part1_return.second;

  auto state1 = p1_part1_return.first;
  auto matrix_mult_msg0_p1 = p1_part0_return.second;
  auto matrix_mult_msg1_p1 = p1_part1_return.second;

  // Each party computes its share of output from its input, Beaver triple
  // matrix, and other party's message.
  ASSERT_OK_AND_ASSIGN(
      auto share_xy_0,
      CorrelatedMatrixProductPartyZero(
          state0, std::get<0>(beaver_matrix_a_shares),
          beaver_matrix_b_c_share_p0.first, beaver_matrix_b_c_share_p0.second,
          matrix_mult_msg0_p1, matrix_mult_msg1_p1, dim1, dim2, dim3,
          kNumFractionalBits, kRingModulus));
  ASSERT_OK_AND_ASSIGN(
      auto share_xy_1,
      CorrelatedMatrixProductPartyOne(
          state1, std::get<1>(beaver_matrix_a_shares),
          beaver_matrix_b_c_share_p1.first, beaver_matrix_b_c_share_p1.second,
          matrix_mult_msg0_p0, matrix_mult_msg1_p0, dim1, dim2, dim3,
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

// Compute X * Y0 and X * Y1
TEST_F(CorrelatedMatrixProductTest,
       SecureCorrelatedMatrixMultipleProductsSucceeds) {
  // fpe = 1.5 and represented as 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe,
                       fp_factory_->CreateFixedPointElementFromDouble(1.5));
  // fpe_negate = -1.5 and represented as kRingModulus - 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe_negate, fpe.Negate());
  // zero = 0 and represented as 0*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto zero,
                       fp_factory_->CreateFixedPointElementFromInt(0));
  size_t dim1 = 1;
  size_t dim2 = 3;
  size_t dim3 = 1;

  // Initialize input vectors in the clear.
  // X = [0    1.5  1.5].
  std::vector<FixedPointElement> fpe_x{zero, fpe, fpe};
  // Y0 = [ 1.5 ].
  //      [ 1.5 ].
  //      [ 1.5 ].
  std::vector<FixedPointElement> fpe_y_0{fpe, fpe, fpe};
  // Y1 =[ -1.5 ].
  //     [ -1.5 ].
  //     [ -1.5 ].
  std::vector<FixedPointElement> fpe_y_1{fpe_negate, fpe_negate, fpe_negate};

  // Make input matrices X and Y from fixed point numbers.
  // X = {0, 1.5, -1.5}*2^{10}
  // Y0 = {1.5, 1.5, 1.5}*2^{10}
  // Y1 = {-1.5, -1.5, -1.5}*2^{10}
  std::vector<uint64_t> matrix_x;
  std::vector<uint64_t> matrix_y0;
  std::vector<uint64_t> matrix_y1;
  for (size_t idx = 0; idx < fpe_x.size(); idx++) {
    matrix_x.push_back(fpe_x[idx].ExportToUint64());
  }
  for (size_t idx = 0; idx < fpe_y_0.size(); idx++) {
    matrix_y0.push_back(fpe_y_0[idx].ExportToUint64());
    matrix_y1.push_back(fpe_y_1[idx].ExportToUint64());
  }

  // Matrix multiplication on share of X (correlated) with Y0, Y1.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, MakePrng());

  // Sample A, mask for X, only once
  ASSERT_OK_AND_ASSIGN(auto a_shares,
                       SampleBeaverTripleMatrixA(dim1, dim2, kRingModulus));

  // Sample masks for Y0 and Y1
  ASSERT_OK_AND_ASSIGN(
      auto beaver_matrix_b_c_shares_y0,
      SampleBeaverTripleMatrixBandC(a_shares, dim1, dim2, dim3, kRingModulus));
  ASSERT_OK_AND_ASSIGN(
      auto beaver_matrix_b_c_shares_y1,
      SampleBeaverTripleMatrixBandC(a_shares, dim1, dim2, dim3, kRingModulus));

  auto beaver_matrix_b_c_share_y0_p0 = beaver_matrix_b_c_shares_y0.first;
  auto beaver_matrix_b_c_share_y0_p1 = beaver_matrix_b_c_shares_y0.second;
  auto beaver_matrix_b_c_share_y1_p0 = beaver_matrix_b_c_shares_y1.first;
  auto beaver_matrix_b_c_share_y1_p1 = beaver_matrix_b_c_shares_y1.second;

  // Generate random shares for matrix x and y0 and y1
  // and distribute them to P0 and P1.
  ASSERT_OK_AND_ASSIGN(
      auto share_x_0,
      SampleVectorFromPrng(dim1 * dim2, kRingModulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(auto share_x_1,
                       BatchedModSub(matrix_x, share_x_0, kRingModulus));

  ASSERT_OK_AND_ASSIGN(
      auto share_y0_0,
      SampleVectorFromPrng(dim2 * dim3, kRingModulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(auto share_y0_1,
                       BatchedModSub(matrix_y0, share_y0_0, kRingModulus));

  ASSERT_OK_AND_ASSIGN(
      auto share_y1_0,
      SampleVectorFromPrng(dim2 * dim3, kRingModulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(auto share_y1_1,
                       BatchedModSub(matrix_y1, share_y1_0, kRingModulus));

  // Each party generates its matrix multiplication pair of messages.

  // P_0 Messages

  // X-A is sent only once
  ASSERT_OK_AND_ASSIGN(auto p0_part0_return,
                       GenerateMatrixXminusAProductMessage(
                           share_x_0, std::get<0>(a_shares), kRingModulus));
  // Generate P0's Y0 - B0 and Y1 - B1
  ASSERT_OK_AND_ASSIGN(auto p0_prod0_part1_return,
                       GenerateMatrixYminusBProductMessage(
                           share_y0_0, p0_part0_return.first,
                           beaver_matrix_b_c_share_y0_p0.first, kRingModulus));
  ASSERT_OK_AND_ASSIGN(auto p0_prod1_part1_return,
                       GenerateMatrixYminusBProductMessage(
                           share_y1_0, p0_part0_return.first,
                           beaver_matrix_b_c_share_y1_p0.first, kRingModulus));

  // P_1 Messages

  // X-A is sent only once
  ASSERT_OK_AND_ASSIGN(auto p1_part0_return,
                       GenerateMatrixXminusAProductMessage(
                           share_x_1, std::get<1>(a_shares), kRingModulus));
  // Generate P1's Y0 - B0 and Y1 - B1
  ASSERT_OK_AND_ASSIGN(auto p1_prod0_part1_return,
                       GenerateMatrixYminusBProductMessage(
                           share_y0_1, p1_part0_return.first,
                           beaver_matrix_b_c_share_y0_p1.first, kRingModulus));
  ASSERT_OK_AND_ASSIGN(auto p1_prod1_part1_return,
                       GenerateMatrixYminusBProductMessage(
                           share_y1_1, p1_part0_return.first,
                           beaver_matrix_b_c_share_y1_p1.first, kRingModulus));

  auto state_prod0_p0 = p0_prod0_part1_return.first;
  auto state_prod1_p0 = p0_prod1_part1_return.first;
  auto matrix_mult_msg0_p0 = p0_part0_return.second;
  auto matrix_mult_prod0_msg1_p0 = p0_prod0_part1_return.second;
  auto matrix_mult_prod1_msg1_p0 = p0_prod1_part1_return.second;

  auto state_prod0_p1 = p1_prod0_part1_return.first;
  auto state_prod1_p1 = p1_prod1_part1_return.first;
  auto matrix_mult_msg0_p1 = p1_part0_return.second;
  auto matrix_mult_prod0_msg1_p1 = p1_prod0_part1_return.second;
  auto matrix_mult_prod1_msg1_p1 = p1_prod1_part1_return.second;

  // Each party computes its share of output from its input, Beaver triple
  // matrix, and other party's message.

  // X * Y0
  ASSERT_OK_AND_ASSIGN(auto share_xy0_0,
                       CorrelatedMatrixProductPartyZero(
                           state_prod0_p0, std::get<0>(a_shares),
                           beaver_matrix_b_c_share_y0_p0.first,
                           beaver_matrix_b_c_share_y0_p0.second,
                           matrix_mult_msg0_p1, matrix_mult_prod0_msg1_p1, dim1,
                           dim2, dim3, kNumFractionalBits, kRingModulus));
  ASSERT_OK_AND_ASSIGN(auto share_xy0_1,
                       CorrelatedMatrixProductPartyOne(
                           state_prod0_p1, std::get<1>(a_shares),
                           beaver_matrix_b_c_share_y0_p1.first,
                           beaver_matrix_b_c_share_y0_p1.second,
                           matrix_mult_msg0_p0, matrix_mult_prod0_msg1_p0, dim1,
                           dim2, dim3, kNumFractionalBits, kRingModulus));
  // X * Y1
  ASSERT_OK_AND_ASSIGN(auto share_xy1_0,
                       CorrelatedMatrixProductPartyZero(
                           state_prod1_p0, std::get<0>(a_shares),
                           beaver_matrix_b_c_share_y1_p0.first,
                           beaver_matrix_b_c_share_y1_p0.second,
                           matrix_mult_msg0_p1, matrix_mult_prod1_msg1_p1, dim1,
                           dim2, dim3, kNumFractionalBits, kRingModulus));
  ASSERT_OK_AND_ASSIGN(auto share_xy1_1,
                       CorrelatedMatrixProductPartyOne(
                           state_prod1_p1, std::get<1>(a_shares),
                           beaver_matrix_b_c_share_y1_p1.first,
                           beaver_matrix_b_c_share_y1_p1.second,
                           matrix_mult_msg0_p0, matrix_mult_prod1_msg1_p0, dim1,
                           dim2, dim3, kNumFractionalBits, kRingModulus));

  // Reconstruct output and verify the correctness.
  // NOTE: This is to test matrix multiplication for integers.
  ASSERT_OK_AND_ASSIGN(auto computed_xy0,
                       BatchedModAdd(share_xy0_0, share_xy0_1, kRingModulus));
  ASSERT_OK_AND_ASSIGN(auto computed_xy1,
                       BatchedModAdd(share_xy1_0, share_xy1_1, kRingModulus));

  // Test with hand-computed values for certainty
  ASSERT_OK_AND_ASSIGN(auto out_one,
                       fp_factory_->CreateFixedPointElementFromDouble(4.5));
  ASSERT_OK_AND_ASSIGN(auto out_two, out_one.Negate());
  // The error is at most 2^-lf
  // (2^-10 in our case which is 2^10 * 2^-10 = 1 in the ring)
  EXPECT_NEAR(computed_xy0[0], out_one.ExportToUint64(), 1);
  EXPECT_NEAR(computed_xy1[0], out_two.ExportToUint64(), 1);
}

}  // namespace
}  // namespace private_join_and_compute
