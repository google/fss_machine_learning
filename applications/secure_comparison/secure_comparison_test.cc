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

#include "applications/secure_comparison/secure_comparison.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "private_join_and_compute/util/status_testing.inc"
#include "secret_sharing_mpc/gates/vector_addition.h"
#include "secret_sharing_mpc/gates/vector_subtraction.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace private_join_and_compute {
namespace secure_comparison {
namespace {

using ::testing::HasSubstr;
using testing::StatusIs;
using ::testing::Test;

// l_f in secure poisson regression paper
const size_t kNumFractionalBits = 20;
// l in secure poisson regression paper
const size_t kNumRingBits = 62;//3;//62;
const uint64_t kRingModulus = (1ULL << kNumRingBits);

// {20, 63, 2^10, 2^43, 2^63}
const FixedPointElementFactory::Params kSampleParams = {
    kNumFractionalBits, kNumRingBits, (1ULL << kNumFractionalBits),
    (1ULL << (kNumRingBits - kNumFractionalBits)), (1ULL << kNumRingBits)};

class ComparisonTest : public Test {
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


TEST_F(ComparisonTest, ComparisonSucceeds) {
  // zero = 0 and represented as 0 * 2^{10}.
  ASSERT_OK_AND_ASSIGN(FixedPointElement zero,
                       fp_factory_->CreateFixedPointElementFromInt(0));
  // two = 2 and represented as 2 * 2^{10}.
  ASSERT_OK_AND_ASSIGN(FixedPointElement two,
                       fp_factory_->CreateFixedPointElementFromInt(2));
  // four = 4 and represented as 4 * 2^{10}.
  ASSERT_OK_AND_ASSIGN(FixedPointElement four,
                       fp_factory_->CreateFixedPointElementFromInt(4));

  // three = 3 and represented as 3 * 2^{10}.
  ASSERT_OK_AND_ASSIGN(FixedPointElement three,
                       fp_factory_->CreateFixedPointElementFromInt(3));

  // Testing x > y
  // Initialize comparison inputs x, y in the clear in ring Z_kRingModulus.
  // x = {0} * 2^10
  // y = {3} * 2^10
  std::vector<uint64_t> vector_x { zero.ExportToUint64(), two.ExportToUint64(), three.ExportToUint64(), four.ExportToUint64() };
  std::vector<uint64_t> vector_y { three.ExportToUint64(), three.ExportToUint64(), three.ExportToUint64(), three.ExportToUint64() };
  std::vector<uint64_t> expected_result = {1, 1, 0, 0};
  // std::vector<uint64_t> expected_result = {0, 0, 0, 1};


  EXPECT_EQ(vector_x.size(), vector_y.size());

  size_t batch_size = vector_x.size();
  size_t block_length = 4;
  size_t num_splits = 16;

  EXPECT_EQ(block_length * num_splits, 64);

  //
  // Offline phase with a trusted dealer
  //

  ASSERT_OK_AND_ASSIGN(
      auto precomputed_values,
      internal::ComparisonPrecomputation(
          batch_size, block_length, num_splits));

  auto short_gates = std::move(std::get<0>(precomputed_values));
  auto precomputed_values_p0 = std::move(std::get<1>(precomputed_values));  // P_0 share
  auto precomputed_values_p1 = std::move(std::get<2>(precomputed_values));  // P_1 share

  //
  // Online phase
  //

  // Round 1

  ASSERT_OK_AND_ASSIGN(auto round_one_p0,
      ComparisonGenerateRoundOneMessage(
          vector_x,
          precomputed_values_p0,
          num_splits, block_length));

  ComparisonStateRoundOne state_round_one_p0 = round_one_p0.first;
  ComparisonMessageRoundOne message_round_one_p0 = round_one_p0.second;

  ASSERT_OK_AND_ASSIGN(auto round_one_p1,
                       ComparisonGenerateRoundOneMessage(
                           vector_y,
                           precomputed_values_p1,
                           num_splits, block_length));

  ComparisonStateRoundOne state_round_one_p1 = round_one_p1.first;
  ComparisonMessageRoundOne message_round_one_p1 = round_one_p1.second;

  // Use the round 1 message to compute the short equalities/comparisons
  // and put them into appropriate form to be able to invoke the 'combination step'
  ASSERT_OK_AND_ASSIGN(
      ComparisonShortComparisonEquality combination_input_p0,
      ComparisonComputeShortComparisonEqualityPartyZero(
          short_gates,
          state_round_one_p0,
          message_round_one_p1,
          precomputed_values_p0,
          // needed for now to unmask other party's input
          num_splits, block_length));

  ASSERT_OK_AND_ASSIGN(
      ComparisonShortComparisonEquality combination_input_p1,
      ComparisonComputeShortComparisonEqualityPartyOne(
          short_gates,
          state_round_one_p1,
          message_round_one_p0,
          precomputed_values_p1,
          // needed for now to unmask other party's input
          num_splits, block_length));

  // Now recursively combine the short comparisons/equalities into a single output with AND gates
  size_t num_rounds = log2(num_splits);
  for (size_t idx = 0; idx < num_rounds; idx++) {

    // Generate message for the Hadamard Product

    ASSERT_OK_AND_ASSIGN(
        auto round_state_message_p0,
        ComparisonGenerateNextRoundMessage(
            combination_input_p0,
            precomputed_values_p0));

    BatchedMultState round_state_p0 = round_state_message_p0.first;
    MultiplicationGateMessage round_message_p0 = round_state_message_p0.second;

    ASSERT_OK_AND_ASSIGN(
        auto round_state_message_p1,
        ComparisonGenerateNextRoundMessage(
            combination_input_p1,
            precomputed_values_p1));

    BatchedMultState round_state_p1 = round_state_message_p1.first;
    MultiplicationGateMessage round_message_p1 = round_state_message_p1.second;

    // Compute Hadamard Product and prepare inputs for next round
    // Yields output in the last round

    ASSERT_OK_AND_ASSIGN(
        combination_input_p0,
        ComparisonProcessNextRoundMessagePartyZero(
            combination_input_p0,
            round_state_p0,
            precomputed_values_p0,
            round_message_p1,
            num_splits / pow(2, idx + 1)));

    ASSERT_OK_AND_ASSIGN(
        combination_input_p1,
        ComparisonProcessNextRoundMessagePartyOne(
            combination_input_p1,
            round_state_p1,
            precomputed_values_p1,
            round_message_p0,
            num_splits / pow(2, idx + 1)));

  }

  // Result is in combination_input_p0/1
  // Retrieve the result

  EXPECT_EQ(combination_input_p0.even_comparison_even_equality_output_shares.size(), batch_size); // where result is
  ASSERT_TRUE(combination_input_p0.odd_comparison_appended_zeros_shares.empty());
  ASSERT_TRUE(combination_input_p0.odd_equality_twice_copied_shares.empty());

  EXPECT_EQ(combination_input_p1.even_comparison_even_equality_output_shares.size(), batch_size); // where result is
  ASSERT_TRUE(combination_input_p1.odd_comparison_appended_zeros_shares.empty());
  ASSERT_TRUE(combination_input_p1.odd_equality_twice_copied_shares.empty());

  std::vector<uint64_t> comparison_result_p0 = combination_input_p0.even_comparison_even_equality_output_shares;
  std::vector<uint64_t> comparison_result_p1 = combination_input_p1.even_comparison_even_equality_output_shares;

  // Verify the shares are either 0 or 1
  for (size_t idx = 0; idx < batch_size; idx++) {
    ASSERT_TRUE(comparison_result_p0[idx] == 1 || comparison_result_p0[idx] == 0);
    ASSERT_TRUE(comparison_result_p1[idx] == 1 || comparison_result_p1[idx] == 0);
  }

  // Add the shares mod 2
  ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> comparison_result,
      VectorAdd(comparison_result_p0, comparison_result_p1, 2));

  // Check the reconstructed result is correct
  for (size_t idx = 0; idx < batch_size; idx++) {
    // std::cout << vector_x[idx] << " " << vector_y[idx] << " " << comparison_result[idx] << " " << 
    ASSERT_TRUE(comparison_result[idx] == expected_result[idx]);
  }
}

}  // namespace
}  // namespace secure_comparison
}  // namespace private_join_and_compute