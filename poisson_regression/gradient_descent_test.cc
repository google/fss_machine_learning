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

#include "poisson_regression/gradient_descent.h"

#include <cstdint>

#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status.inc"
#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"

namespace private_join_and_compute {
namespace poisson_regression {
namespace {

using internal::SampleShareOfZero;
using ::testing::Test;

// Set the number of fractional bits to be small, so that the fail probability
// is negligible.
const int kNumFractionalBits = 10;
const int kNumRingBits = 63;
const int kExpBound = 7;
const uint64_t kPrimeQ = 9223372036854775783ULL;
const int kNumIteration = 100;

class GradientDescentTest : public Test {
 public:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        FixedPointElementFactory temp,
        FixedPointElementFactory::Create(kNumFractionalBits, kNumRingBits));
    fp_factory_ = absl::make_unique<FixedPointElementFactory>(std::move(temp));
  }

 protected:
  std::unique_ptr<FixedPointElementFactory> fp_factory_;
};

TEST_F(GradientDescentTest, ValidInputSucceeds) {
  // Testing using Somoza dataset. The first column is for the offset.
  std::vector<uint64_t> data_x = {
    1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
    1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
    1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
    1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
    1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1};

  // Exposure.
  const std::vector<double> delta_t = {
    278.4167, 538.8333, 794.375, 1550.75, 3006, 8743.5, 14270,
    403.2083, 786.0000, 1165.25, 2294.75, 4500.5, 13201.5, 19525,
    495.2500, 956.6667, 1381.375, 2604.5, 4618.5, 9814.5, 5802.5};

  // Outcome.
  const std::vector<uint64_t> label_y = {
    168, 48, 63, 89, 102, 81, 0,
    197, 48, 62, 81, 97, 103, 0,
    195, 55, 58, 85, 87, 70, 0};

  // Generate parameters for gradient descent.
  const GradientDescentParams gd_params = {
    .num_features = label_y.size(),
    .feature_length = data_x.size()/label_y.size(),
    .num_iterations = kNumIteration,
    .num_ring_bits = kNumRingBits,
    .num_fractional_bits = kNumFractionalBits,
    .alpha = static_cast<uint64_t>(0.0001*(1ULL << kNumFractionalBits)),
    .one_minus_beta =
    ((1ULL << kNumFractionalBits) -
     static_cast<uint64_t>(0.0001*(1ULL << kNumFractionalBits))),
    .modulus = (1ULL << kNumRingBits),
    .exponent_bound = kExpBound,
    .prime_q = kPrimeQ
  };

  const FixedPointElementFactory::Params fpe_params = {
    .num_fractional_bits = kNumFractionalBits,
    .num_ring_bits = kNumRingBits,
    .fractional_multiplier = 1ULL << kNumFractionalBits,
    .integer_ring_modulus = 1ULL << (kNumRingBits - kNumFractionalBits),
    .primary_ring_modulus = 1ULL << kNumRingBits
  };

  const ExponentiationParams exp_params = {
    .exponent_bound = kExpBound,
    .prime_q = kPrimeQ
  };

  // Secret share inputs.
  // Sharing data (share_x).
  ASSERT_OK_AND_ASSIGN(auto share_x,
                       internal::SampleShareOfZero(data_x.size(),
                                                   gd_params.modulus));
  ASSERT_OK_AND_ASSIGN(auto share_x_p0,
                       BatchedModAdd(share_x.first, data_x, gd_params.modulus));
  auto share_x_p1 = std::move(share_x.second);
  // Sharing delta_t.
  ASSERT_OK_AND_ASSIGN(auto share_delta,
                       internal::SampleShareOfZero(delta_t.size(),
                                                   gd_params.modulus));
  std::vector<uint64_t> share_delta_p0(delta_t.size());
  for (size_t idx = 0; idx < delta_t.size(); idx++) {
    ASSERT_OK_AND_ASSIGN(
        FixedPointElement fpe,
        fp_factory_->CreateFixedPointElementFromDouble(delta_t[idx]));
    share_delta_p0[idx] = ModAdd(fpe.ExportToUint64(), share_delta.first[idx],
                                 gd_params.modulus);
  }
  auto share_delta_p1 = std::move(share_delta.second);
  // Sharing share_y.
  ASSERT_OK_AND_ASSIGN(auto share_y,
                       internal::SampleShareOfZero(label_y.size(),
                                                   gd_params.modulus));
  ASSERT_OK_AND_ASSIGN(auto share_y_p0,
                       BatchedModAdd(share_y.first, label_y,
                                     gd_params.modulus));
  auto share_y_p1 = std::move(share_y.second);
  // Initialize share_theta to 0.
  std::vector<uint64_t> share_theta_p0(gd_params.feature_length, 0);
  std::vector<uint64_t> share_theta_p1(gd_params.feature_length, 0);

  // Initialize preprocessed shares in trusted setup.
  std::vector<BeaverTripleMatrix<uint64_t>> beaver_triple_matrix_round1_p0;
  std::vector<MultToAddShare> mult_to_add_shares_round2_p0;
  std::vector<BeaverTripleVector<uint64_t>> beaver_triple_vector_round3_p0;
  std::vector<BeaverTripleMatrix<uint64_t>> beaver_triple_matrix_round4_p0;
  std::vector<BeaverTripleMatrix<uint64_t>> beaver_triple_matrix_round1_p1;
  std::vector<MultToAddShare> mult_to_add_shares_round2_p1;
  std::vector<BeaverTripleVector<uint64_t>> beaver_triple_vector_round3_p1;
  std::vector<BeaverTripleMatrix<uint64_t>> beaver_triple_matrix_round4_p1;
  for (size_t idx = 0; idx < gd_params.num_iterations; idx++) {
    // X*Theta: X (#features x feature length), Theta (feature length x 1).
    ASSERT_OK_AND_ASSIGN(auto matrix_shares_1,
                         internal::SampleBeaverMatrixShareWithPrng(
                             gd_params.num_features,
                             gd_params.feature_length,
                             1,
                             gd_params.modulus));
    beaver_triple_matrix_round1_p0.push_back(matrix_shares_1.first);
    beaver_triple_matrix_round1_p1.push_back(matrix_shares_1.second);
    ASSERT_OK_AND_ASSIGN(auto mult_to_add_shares_2,
                         internal::SampleMultToAddSharesWithPrng(
                             gd_params.num_features,
                             gd_params.prime_q));
    mult_to_add_shares_round2_p0.push_back(mult_to_add_shares_2.first);
    mult_to_add_shares_round2_p1.push_back(mult_to_add_shares_2.second);
    ASSERT_OK_AND_ASSIGN(auto vector_shares_3,
                         internal::SampleBeaverVectorShareWithPrng(
                             gd_params.num_features,
                             gd_params.modulus));
    beaver_triple_vector_round3_p0.push_back(vector_shares_3.first);
    beaver_triple_vector_round3_p1.push_back(vector_shares_3.second);
    // S*X: S (1 x #features), X (#features x feature length).
    ASSERT_OK_AND_ASSIGN(auto matrix_shares_4,
                         internal::SampleBeaverMatrixShareWithPrng(
                             1,
                             gd_params.num_features,
                             gd_params.feature_length,
                             gd_params.modulus));
    beaver_triple_matrix_round4_p0.push_back(matrix_shares_4.first);
    beaver_triple_matrix_round4_p1.push_back(matrix_shares_4.second);
  }

  // Initialize ShareProvider with preprocessed shares.
  ASSERT_OK_AND_ASSIGN(auto share_provider_p0,
                       ShareProvider::Create(
      beaver_triple_matrix_round1_p0, mult_to_add_shares_round2_p0,
      beaver_triple_vector_round3_p0, beaver_triple_matrix_round4_p0,
      gd_params.num_features, gd_params.feature_length,
      gd_params.num_iterations));
  ASSERT_OK_AND_ASSIGN(auto share_provider_p1,
                       ShareProvider::Create(
      beaver_triple_matrix_round1_p1, mult_to_add_shares_round2_p1,
      beaver_triple_vector_round3_p1, beaver_triple_matrix_round4_p1,
      gd_params.num_features, gd_params.feature_length,
      gd_params.num_iterations));

  // Initialize GradientDescentPartyZero/One with input and preprocessed shares.
  ASSERT_OK_AND_ASSIGN(
      auto gd_party_zero,
      GradientDescentPartyZero::Init(
          share_x_p0, share_y_p0, share_delta_p0, share_theta_p0,
          absl::make_unique<ShareProvider>(share_provider_p0), fpe_params,
          exp_params, gd_params));
  ASSERT_OK_AND_ASSIGN(
      auto gd_party_one,
      GradientDescentPartyOne::Init(
          share_x_p1, share_y_p1, share_delta_p1, share_theta_p0,
          absl::make_unique<ShareProvider>(share_provider_p1), fpe_params,
          exp_params, gd_params));

  // Running gradient descent for poisson regression.
  for (size_t idx = 0; idx < gd_params.num_iterations; idx++) {
    // Round 1.
    ASSERT_OK_AND_ASSIGN(
        auto p0_return_round1,
        gd_party_zero.GenerateGradientDescentRoundOneMessage());
    ASSERT_OK_AND_ASSIGN(
        auto p1_return_round1,
        gd_party_one.GenerateGradientDescentRoundOneMessage());
    // Round 2.
    ASSERT_OK_AND_ASSIGN(
        auto p0_return_round2,
        gd_party_zero.GenerateGradientDescentRoundTwoMessage(
            p0_return_round1.first, p1_return_round1.second));
    ASSERT_OK_AND_ASSIGN(
        auto p1_return_round2,
        gd_party_one.GenerateGradientDescentRoundTwoMessage(
            p1_return_round1.first, p0_return_round1.second));
    // Round 3.
    ASSERT_OK_AND_ASSIGN(
        auto p0_return_round3,
        gd_party_zero.GenerateGradientDescentRoundThreeMessage(
            p0_return_round2.first, p1_return_round2.second));
    ASSERT_OK_AND_ASSIGN(
        auto p1_return_round3,
        gd_party_one.GenerateGradientDescentRoundThreeMessage(
            p1_return_round2.first, p0_return_round2.second));
    // Round 4.
    ASSERT_OK_AND_ASSIGN(
        auto p0_return_round4,
        gd_party_zero.GenerateGradientDescentRoundFourMessage(
            p0_return_round3.first, p1_return_round3.second));
    ASSERT_OK_AND_ASSIGN(
        auto p1_return_round4,
        gd_party_one.GenerateGradientDescentRoundFourMessage(
            p1_return_round3.first, p0_return_round3.second));
    // Compute gradient descent update.
    ASSERT_OK(gd_party_zero.ComputeGradientUpdate(
        p0_return_round4.first, p1_return_round4.second));
    ASSERT_OK(gd_party_one.ComputeGradientUpdate(
        p1_return_round4.first, p0_return_round4.second));
  }

  // Reconstruct theta, and check if the value of the parameters makes sense.
  // For Somoza dataset, theta[i] is always bounded (loosely) by (-5, 5).
  std::vector<uint64_t> updated_share_theta_p0 = gd_party_zero.GetTheta();
  std::vector<uint64_t> updated_share_theta_p1 = gd_party_one.GetTheta();
  ASSERT_OK_AND_ASSIGN(auto theta,
                       BatchedModMul(updated_share_theta_p0,
                                     updated_share_theta_p1,
                                     gd_params.modulus));
  // Get the absolute value of theta and check if it is in the good range.
  for (size_t idx = 0; idx < theta.size(); idx++) {
    if (theta[idx] > gd_params.modulus/2) {
      theta[idx] = gd_params.modulus - theta[idx];
    }
    theta[idx] = theta[idx] >> gd_params.num_fractional_bits;
    ASSERT_LT(theta[idx], 5);
  }
}

}  // namespace
}  // namespace poisson_regression
}  // namespace private_join_and_compute
