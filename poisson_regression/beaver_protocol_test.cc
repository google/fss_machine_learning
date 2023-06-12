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

#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
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

// kPrimeModulus = 2^{63} - 25;
const uint64_t kPrimeModulus = (1ULL << 63) - 25;

const size_t kNumFractionalBits = 10;
const size_t kNumRingBits = 63;
const uint64_t kRingModulus = (1ULL << 63);

// {10, 63, 2^10, 2^53, 2^63}
const FixedPointElementFactory::Params kSampleParams
    = {kNumFractionalBits, kNumRingBits, (1ULL << kNumFractionalBits),
       (1ULL << (kNumRingBits - kNumFractionalBits)), (1ULL << kNumRingBits)};

class BeaverProtocolTest : public Test {
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

TEST_F(BeaverProtocolTest, MultMessageEmptyInputFails) {
  std::vector<uint64_t> empty_vector(0);
  std::vector<uint64_t> small_vector{1, 2, 3, 4, 5};
  ASSERT_OK_AND_ASSIGN(auto beaver_vector,
                       BeaverTripleVector<uint64_t>::Create(
                           small_vector, small_vector, small_vector));
  auto mult_state_and_message = GenerateBatchedMultiplicationGateMessage(
      empty_vector, empty_vector, beaver_vector, kPrimeModulus);
  EXPECT_THAT(
      mult_state_and_message,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("GenerateBatchedMultiplicationGateMessage: input "
                         "must not be empty.")));
}

TEST_F(BeaverProtocolTest, MultMessageNotEqualLengthSharesFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  std::vector<uint64_t> small_vector_2{1, 2, 3, 4, 5, 6};
  ASSERT_OK_AND_ASSIGN(auto beaver_vector,
                       BeaverTripleVector<uint64_t>::Create(
                           small_vector_1, small_vector_1, small_vector_1));
  auto mult_state_and_message = GenerateBatchedMultiplicationGateMessage(
      small_vector_1, small_vector_2, beaver_vector, kPrimeModulus);
  EXPECT_THAT(
      mult_state_and_message,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("GenerateBatchedMultiplicationGateMessage: shares "
                         "must have the same length.")));
}

TEST_F(BeaverProtocolTest, MultMessageInputAndBeaverLengthMismatchFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  std::vector<uint64_t> small_vector_2{1, 2, 3, 4, 5, 6};
  ASSERT_OK_AND_ASSIGN(auto beaver_vector,
                       BeaverTripleVector<uint64_t>::Create(
                           small_vector_1, small_vector_1, small_vector_1));
  auto mult_state_and_message = GenerateBatchedMultiplicationGateMessage(
      small_vector_2, small_vector_2, beaver_vector, kPrimeModulus);
  EXPECT_THAT(
      mult_state_and_message,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("GenerateBatchedMultiplicationGateMessage: "
                         "input and beaver shares have different size.")));
}

TEST_F(BeaverProtocolTest, MultMessageSmallInputSucceeds) {
  size_t length = 10;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, this->MakePrng());
  ASSERT_OK_AND_ASSIGN(auto share_x,
                       SampleVectorFromPrng(length, kPrimeModulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(auto share_y,
                       SampleVectorFromPrng(length, kPrimeModulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(auto beaver_vector_shares,
                       internal::SampleBeaverVectorShareWithPrng(
                           length, kPrimeModulus));
  auto beaver_vector_share = beaver_vector_shares.first;
  ASSERT_OK_AND_ASSIGN(
      auto mult_state_and_message,
      GenerateBatchedMultiplicationGateMessage(
          share_x, share_y, beaver_vector_share, kPrimeModulus));
  auto state = mult_state_and_message.first;
  auto mult_message = mult_state_and_message.second;
  ASSERT_OK_AND_ASSIGN(auto x_minus_a,
                       BatchedModSub(
                           share_x, beaver_vector_share.GetA(), kPrimeModulus));
  ASSERT_OK_AND_ASSIGN(auto y_minus_b,
                       BatchedModSub(
                           share_y, beaver_vector_share.GetB(), kPrimeModulus));
  EXPECT_EQ(x_minus_a.size(),
            mult_message.vector_x_minus_vector_a_shares_size());
  EXPECT_EQ(y_minus_b.size(),
            mult_message.vector_y_minus_vector_b_shares_size());
  for (size_t idx = 0; idx < length; idx++) {
    EXPECT_EQ(x_minus_a[idx], mult_message.vector_x_minus_vector_a_shares(idx));
    EXPECT_EQ(y_minus_b[idx], mult_message.vector_y_minus_vector_b_shares(idx));
  }
}


TEST_F(BeaverProtocolTest, MatrixMultMessageEmptyInputFails) {
  std::vector<uint64_t> empty_vector(0);
  std::vector<uint64_t> small_vector{1};
  ASSERT_OK_AND_ASSIGN(auto beaver_matrix,
                       BeaverTripleMatrix<uint64_t>::Create(
                           small_vector, small_vector, small_vector, 1, 1, 1));
  auto matrix_mult_state_and_message = GenerateMatrixMultiplicationGateMessage(
      empty_vector, empty_vector, beaver_matrix, kPrimeModulus);
  EXPECT_THAT(
      matrix_mult_state_and_message,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("GenerateMatrixMultiplicationGateMessage: input "
                         "must not be empty.")));
}

TEST_F(BeaverProtocolTest, MatrixMultMessageInputAndBeaverLengthMismatchFails) {
  std::vector<uint64_t> share_x{1, 2, 3, 4};
  std::vector<uint64_t> share_y{1, 2, 3, 4};
  std::vector<uint64_t> small_vector{1};
  ASSERT_OK_AND_ASSIGN(auto beaver_matrix,
                       BeaverTripleMatrix<uint64_t>::Create(
                           small_vector, small_vector, small_vector, 1, 1, 1));
  auto matrix_mult_state_and_message = GenerateMatrixMultiplicationGateMessage(
      share_x, share_y, beaver_matrix, kPrimeModulus);
  EXPECT_THAT(
      matrix_mult_state_and_message,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("GenerateMatrixMultiplicationGateMessage: "
                         "input and beaver shares have different size.")));
}

TEST_F(BeaverProtocolTest, MatrixMultMessageSmallInputSucceeds) {
  size_t dim1 = 2;
  size_t dim2 = 3;
  size_t dim3 = 4;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, this->MakePrng());
  ASSERT_OK_AND_ASSIGN(auto share_x,
                       SampleVectorFromPrng(dim1*dim2, kPrimeModulus,
                                            prng.get()));
  ASSERT_OK_AND_ASSIGN(auto share_y,
                       SampleVectorFromPrng(dim2*dim3, kPrimeModulus,
                                            prng.get()));
  ASSERT_OK_AND_ASSIGN(auto beaver_matrix_shares,
                       internal::SampleBeaverMatrixShareWithPrng(
                           dim1, dim2, dim3, kPrimeModulus));
  auto beaver_matrix_share = beaver_matrix_shares.first;
  ASSERT_OK_AND_ASSIGN(
      auto matrix_mult_state_and_message,
      GenerateMatrixMultiplicationGateMessage(
          share_x, share_y, beaver_matrix_share, kPrimeModulus));
  auto state = matrix_mult_state_and_message.first;
  auto matrix_mult_message = matrix_mult_state_and_message.second;
  ASSERT_OK_AND_ASSIGN(auto x_minus_a,
                       BatchedModSub(share_x, beaver_matrix_share.GetA(),
                                     kPrimeModulus));
  ASSERT_OK_AND_ASSIGN(auto y_minus_b,
                       BatchedModSub(share_y, beaver_matrix_share.GetB(),
                                     kPrimeModulus));
  EXPECT_EQ(x_minus_a.size(),
            matrix_mult_message.matrix_x_minus_matrix_a_shares_size());
  EXPECT_EQ(y_minus_b.size(),
            matrix_mult_message.matrix_y_minus_matrix_b_shares_size());
  for (size_t idx = 0; idx < dim1*dim2; idx++) {
    EXPECT_EQ(x_minus_a[idx],
              matrix_mult_message.matrix_x_minus_matrix_a_shares(idx));
  }
  for (size_t idx = 0; idx < dim2*dim3; idx++) {
    EXPECT_EQ(y_minus_b[idx],
              matrix_mult_message.matrix_y_minus_matrix_b_shares(idx));
  }
}

TEST_F(BeaverProtocolTest, TruncateShareEmptyInputFails) {
  std::vector<uint64_t> empty_share(0);
  auto truncated_share_0 = TruncateSharePartyZero(
      empty_share, kNumFractionalBits, kPrimeModulus);
  auto truncated_share_1 = TruncateSharePartyOne(
      empty_share, kNumFractionalBits, kPrimeModulus);
  EXPECT_THAT(truncated_share_0,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("TruncateSharePartyZero: "
                                 "input must not be empty.")));
  EXPECT_THAT(truncated_share_1,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("TruncateSharePartyOne: "
                                 "input must not be empty.")));
}

// Test the truncation function when the share are "good share". If we have a
// share of positive value input, then share_0 + share_1 = input + modulus. if
// input is negative, then share_0 + share_1 = input.

// Test 1: Truncate (1.5*2^{10})*(1.5*2^{10}) --> 2.25*2^{10}.
TEST_F(BeaverProtocolTest, TruncateShareOfPositiveNumberSucceeds) {
  // fpe = 1.5 and represented as 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe,
                       fp_factory_->CreateFixedPointElementFromDouble(1.5));
  // Get a random value to generate share_0 and share_1.
  ASSERT_OK_AND_ASSIGN(auto random_value,
                       fp_factory_->CreateFixedPointElementFromDouble(
                           12342.34234));
  // input = fpe*fpe represented as 2.25*2^{20}.
  ASSERT_OK_AND_ASSIGN(auto input, fpe.ModMul(fpe));

  // When multiply two fixed point number, we need to perform truncation.
  // The truncated value of input is 2.25*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto truncated_input, fpe.TruncMulFP(fpe));

  // Generate share for input (2.25*2^{20}) and give each to a party.
  // The share is generated such that:
  // input_share_0 + input_share_1 = input + kRingModulus.
  ASSERT_OK_AND_ASSIGN(auto input_share_0, input.ModAdd(random_value));
  ASSERT_OK_AND_ASSIGN(auto input_share_1, random_value.Negate());

  std::vector<uint64_t> share_0{input_share_0.ExportToUint64()};
  std::vector<uint64_t> share_1{input_share_1.ExportToUint64()};

  // Simulate the truncation.
  ASSERT_OK_AND_ASSIGN(auto truncated_share_0,
                       TruncateSharePartyZero(share_0,
                                              1ULL << kNumFractionalBits,
                                              kRingModulus));
  ASSERT_OK_AND_ASSIGN(auto truncated_share_1,
                       TruncateSharePartyOne(share_1,
                                             1ULL << kNumFractionalBits,
                                             kRingModulus));
  auto computed_truncated_input = ModAdd(
      truncated_share_0[0], truncated_share_1[0], kRingModulus);
  // The difference between the computed value and the ground truth.
  auto error = abs(static_cast<double>(computed_truncated_input) -
                   static_cast<double>(truncated_input.ExportToUint64()));
  EXPECT_LE(error, 1);
}

// Test 2: Truncate (1.5*2^{10})*(-1.5*2^{10}) --> -2.25*2^{10}.
TEST_F(BeaverProtocolTest, TruncateShareOfNegativeNumberSucceeds) {
  // fpe = 1.5 and represented as 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe,
                       fp_factory_->CreateFixedPointElementFromDouble(1.5));
  // fpe_negate = -1.5 and represented as kRingModulus - 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe_negate, fpe.Negate());
  ASSERT_OK_AND_ASSIGN(auto random_value,
                       fp_factory_->CreateFixedPointElementFromDouble(
                           12342.34234));
  // input = fpe*fpe_negate represented as kRingModulus - 2.25*2^{20}.
  ASSERT_OK_AND_ASSIGN(auto input, fpe.ModMul(fpe_negate));
  // The truncated value of input is kRingModulus - 2.25*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto truncated_input, fpe.TruncMulFP(fpe_negate));

  // Generate share for input (kModulus - 2.25*2^{20}) and give each to a party.
  // The share is generated such that: input_share_0 + input_share_1 = input.
  ASSERT_OK_AND_ASSIGN(auto input_share_0, input.ModSub(random_value));
  std::vector<uint64_t> share_0{input_share_0.ExportToUint64()};
  std::vector<uint64_t> share_1{random_value.ExportToUint64()};

  // Simulate the truncation.
  ASSERT_OK_AND_ASSIGN(auto truncated_share_0,
                       TruncateSharePartyZero(share_0,
                                              1ULL << kNumFractionalBits,
                                              kRingModulus));
  ASSERT_OK_AND_ASSIGN(auto truncated_share_1,
                       TruncateSharePartyOne(share_1,
                                             1ULL << kNumFractionalBits,
                                             kRingModulus));
  auto computed_truncated_input = ModAdd(
      truncated_share_0[0], truncated_share_1[0], kRingModulus);
  auto error = abs(static_cast<double>(computed_truncated_input) -
                   static_cast<double>(truncated_input.ExportToUint64()));
  EXPECT_LE(error, 1);
}

// Truncate (1.5*2^{10})*(1.5*2^{10}) fails when the random shares the parties
// hold is share_0 + share_1 = input (without wraparound in the modulus)
// for a positive value input. When the shares are sampled uniformly at random,
// this happens with negligible probability.
TEST_F(BeaverProtocolTest, TruncateShareOfPositiveNumberFails) {
  // fpe = 1.5 and represented as 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe,
                       fp_factory_->CreateFixedPointElementFromDouble(1.5));
  // Get a random value to generate share_0 and share_1.
  ASSERT_OK_AND_ASSIGN(auto random_value,
                       fp_factory_->CreateFixedPointElementFromDouble(0.5));
  // input = fpe*fpe represented as 2.25*2^{20}.
  ASSERT_OK_AND_ASSIGN(auto input, fpe.ModMul(fpe));

  // When multiply two fixed point number, we need to perform truncation.
  // The truncated value of input is 2.25*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto truncated_input, fpe.TruncMulFP(fpe));

  // Generate share for input (2.25*2^{20}) and give each to a party.
  // The share is generated such that:
  // input_share_0 + input_share_1 = input
  ASSERT_OK_AND_ASSIGN(auto input_share_0, input.ModSub(random_value));

  std::vector<uint64_t> share_0{input_share_0.ExportToUint64()};
  std::vector<uint64_t> share_1{random_value.ExportToUint64()};

  // Simulate the truncation.
  ASSERT_OK_AND_ASSIGN(auto truncated_share_0,
                       TruncateSharePartyZero(share_0,
                                              1ULL << kNumFractionalBits,
                                              kRingModulus));
  ASSERT_OK_AND_ASSIGN(auto truncated_share_1,
                       TruncateSharePartyOne(share_1,
                                             1ULL << kNumFractionalBits,
                                             kRingModulus));
  auto computed_truncated_input = ModAdd(
      truncated_share_0[0], truncated_share_1[0], kRingModulus);
  // The difference between the computed value and the ground truth.
  auto error = abs(static_cast<double>(computed_truncated_input) -
                   static_cast<double>(truncated_input.ExportToUint64()));
  EXPECT_GT(error, 1);
}

// Truncate (1.5*2^{10})*(-1.5*2^{10}) fails when the random shares the parties
// hold is share_0 + share_1 = input + modulus for a negative value input.
// When the shares are sampled uniformly at random, this happens with negligible
// probability.
TEST_F(BeaverProtocolTest, TruncateShareOfNegativeNumberFails) {
  // fpe = 1.5 and represented as 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe,
                       fp_factory_->CreateFixedPointElementFromDouble(1.5));
  // fpe_negate = -1.5 and represented as kRingModulus - 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe_negate, fpe.Negate());
  ASSERT_OK_AND_ASSIGN(auto random_value,
                       fp_factory_->CreateFixedPointElementFromDouble(0.5));
  // input = fpe*fpe_negate represented as kRingModulus - 2.25*2^{20}.
  ASSERT_OK_AND_ASSIGN(auto input, fpe.ModMul(fpe_negate));
  ASSERT_OK_AND_ASSIGN(auto truncated_fpe_square, fpe.TruncMulFP(fpe));
  // The truncated value of input is kRingModulus - 2.25*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto truncated_input, truncated_fpe_square.Negate());

  // Generate share for input (kModulus - 2.25*2^{20}) and give each to a party.
  // The share is generated such that:
  // input_share_0 + input_share_1 = input + modulus.
  ASSERT_OK_AND_ASSIGN(auto input_share_0, input.ModAdd(random_value));
  ASSERT_OK_AND_ASSIGN(auto input_share_1, random_value.Negate());
  std::vector<uint64_t> share_0{input_share_0.ExportToUint64()};
  std::vector<uint64_t> share_1{input_share_1.ExportToUint64()};

  // Simulate the truncation.
  ASSERT_OK_AND_ASSIGN(auto truncated_share_0,
                       TruncateSharePartyZero(share_0,
                                              1ULL << kNumFractionalBits,
                                              kRingModulus));
  ASSERT_OK_AND_ASSIGN(auto truncated_share_1,
                       TruncateSharePartyOne(share_1,
                                             1ULL << kNumFractionalBits,
                                             kRingModulus));
  auto computed_truncated_input = ModAdd(
      truncated_share_0[0], truncated_share_1[0], kRingModulus);
  auto error = abs(static_cast<double>(computed_truncated_input) -
                   static_cast<double>(truncated_input.ExportToUint64()));
  EXPECT_GT(error, 1);
}

// End-to-end test for secure batched multiplication gate.
// Each party has input: share of vectors [X], [Y] and beaver triples
// ([A], [B], [C]). The output is share of vector [X*Y] where * represents
// component-wise multiplication mod modulus.
// In this test, X = {0,   0,    0, 1.5, -1.5, 1.5,  1.5, -1.5, -1.5}*2^{10}
// and           Y = {0, 1.5, -1.5,   0,    0, 1.5, -1.5,  1.5, -1.5}*2^{10}.
// We test all the combinations:
// (0, 0), (0, +), (0, -), (+, 0), (-, 0), (+, +), (+, -), (-, +), (-, -).
TEST_F(BeaverProtocolTest, SecureBatchedMultiplicationSucceeds) {
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
  ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> xy,
                       BatchedModMul(vector_x, vector_y, kRingModulus));

  // Batched multiplication on share of X and Y.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, MakePrng());

  // Generate Beaver triple vector for P0 and P1.
  ASSERT_OK_AND_ASSIGN(auto beaver_vecto_shares,
                       internal::SampleBeaverVectorShareWithPrng(
                           length, kRingModulus));
  auto beaver_vector_share_0 = beaver_vecto_shares.first;
  auto beaver_vector_share_1 = beaver_vecto_shares.second;

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
      GenerateBatchedMultiplicationGateMessage(
          share_x_0, share_y_0, beaver_vector_share_0, kRingModulus));
  auto state0 = p0_return.first;
  auto batched_mult_msg_0 = p0_return.second;
  ASSERT_OK_AND_ASSIGN(
      auto p1_return,
      GenerateBatchedMultiplicationGateMessage(
          share_x_1, share_y_1, beaver_vector_share_1, kRingModulus));
  auto state1 = p1_return.first;
  auto batched_mult_msg_1 = p1_return.second;
  // Each party computes its share of output.
  ASSERT_OK_AND_ASSIGN(
      auto share_xy_0,
      GenerateBatchedMultiplicationOutputPartyZero(
          state0, beaver_vector_share_0, batched_mult_msg_1, kRingModulus));
  ASSERT_OK_AND_ASSIGN(
      auto share_xy_1,
      GenerateBatchedMultiplicationOutputPartyOne(
          state1, beaver_vector_share_1, batched_mult_msg_0, kRingModulus));

  // NOTE: This is to test batched multiplication for integers.
  // Reconstruct output and verify the correctness.
  ASSERT_OK_AND_ASSIGN(auto computed_xy,
                       BatchedModAdd(share_xy_0, share_xy_1, kRingModulus));
  for (size_t idx = 0; idx < length; idx++) {
    EXPECT_EQ(computed_xy[idx], xy[idx]);
  }
}

// End-to-end test for secure matrix multiplication gate.
// Each party has input: share of vectors [X], [Y] and beaver triples
// ([A], [B], [C]). The output is share of vector [X*Y] where * represents
// matrix multiplication mod modulus.
TEST_F(BeaverProtocolTest, SecureMatrixMultiplicationSucceeds) {
  // fpe = 1.5 and represented as 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(
      auto fpe, fp_factory_->CreateFixedPointElementFromDouble(1.5));
  // fpe_negate = -1.5 and represented as kRingModulus - 1.5*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto fpe_negate, fpe.Negate());
  // zero = 0 and represented as 0*2^{10}.
  ASSERT_OK_AND_ASSIGN(
      auto zero, fp_factory_->CreateFixedPointElementFromDouble(0));
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

  // Expected output (X*Y) in the clear.
  ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> xy,
                       ModMatrixMul(matrix_x, matrix_y, dim1, dim2, dim3,
                                    kRingModulus));

  // Matrix multiplication on share of X, Y.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, MakePrng());
  ASSERT_OK_AND_ASSIGN(auto beaver_matrix_shares,
                       internal::SampleBeaverMatrixShareWithPrng(
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

  // Each party generate its matrix multiplication message.
  ASSERT_OK_AND_ASSIGN(
      auto p0_return,
      GenerateMatrixMultiplicationGateMessage(
          share_x_0, share_y_0, beaver_matrix_share_0, kRingModulus));
  ASSERT_OK_AND_ASSIGN(
      auto p1_return,
      GenerateMatrixMultiplicationGateMessage(
          share_x_1, share_y_1, beaver_matrix_share_1, kRingModulus));
  auto state0 = p0_return.first;
  auto matrix_mult_msg_0 = p0_return.second;
  auto state1 = p1_return.first;
  auto matrix_mult_msg_1 = p1_return.second;

  // Each party computes its share of output from its input, Beaver triple
  // matrix, and other party's message.
  ASSERT_OK_AND_ASSIGN(
      auto share_xy_0,
      GenerateMatrixMultiplicationOutputPartyZero(
          state0, beaver_matrix_share_0, matrix_mult_msg_1, kRingModulus));
  ASSERT_OK_AND_ASSIGN(
      auto share_xy_1,
      GenerateMatrixMultiplicationOutputPartyOne(
          state1, beaver_matrix_share_1, matrix_mult_msg_0, kRingModulus));

  // Reconstruct output and verify the correctness.
  // NOTE: This is to test matrix multiplication for integers.
  ASSERT_OK_AND_ASSIGN(auto computed_xy,
                       BatchedModAdd(share_xy_0, share_xy_1, kRingModulus));
  for (size_t idx = 0; idx < dim1*dim3; idx++) {
    EXPECT_EQ(computed_xy[idx], xy[idx]);
  }
}

TEST_F(BeaverProtocolTest, SampleEmptyShareOfZeroFails) {
  auto share_of_zero = internal::SampleShareOfZero(0, kPrimeModulus);
  EXPECT_THAT(share_of_zero,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("SampleShareOfZero: length must positive.")));
}

TEST_F(BeaverProtocolTest, SampleSmallShareOfZeroSucceeds) {
  size_t length = 10;
  ASSERT_OK_AND_ASSIGN(auto shares_of_zero,
                       internal::SampleShareOfZero(length, kPrimeModulus));
  auto shares_of_zero_0 = shares_of_zero.first;
  auto shares_of_zero_1 = shares_of_zero.second;
  ASSERT_OK_AND_ASSIGN(auto sum_of_shares,
                       BatchedModAdd(shares_of_zero_0, shares_of_zero_1,
                                     kPrimeModulus));
  for (size_t idx = 0; idx < length; idx++) {
    EXPECT_EQ(sum_of_shares[idx], 0);
  }
}


}  // namespace
}  // namespace private_join_and_compute
