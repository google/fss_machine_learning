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

#include "applications/logistic_regression/gradient_descent_dp.h"
#include "applications/secure_sigmoid/secure_sigmoid.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status.inc"
#include "private_join_and_compute/util/status_testing.h"
#include "private_join_and_compute/util/status_testing.inc"
#include "secret_sharing_mpc/gates/correlated_matrix_product.h"
#include "secret_sharing_mpc/gates/matrix_product.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"

namespace private_join_and_compute {
namespace logistic_regression_dp {
namespace {

using internal::SampleShareOfZero;
using ::testing::Test;

const int kNumFractionalBits = 10;
const int kNumRingBits = 63;
const int kNumIteration = 5;

class GradientDescentDPTest : public Test {
 public:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        FixedPointElementFactory temp,
        FixedPointElementFactory::Create(kNumFractionalBits, kNumRingBits));
    fp_factory_ = absl::make_unique<FixedPointElementFactory>(std::move(temp));
  }
  StatusOr<std::unique_ptr<BasicRng>> MakePrng() {
    auto random_seed = BasicRng::GenerateSeed();
    if (!random_seed.ok()) {
      return InternalError("Random seed generation fails.");
    }
    return BasicRng::Create(random_seed.value());
  }

 protected:
  std::unique_ptr<FixedPointElementFactory> fp_factory_;
};


TEST_F(GradientDescentDPTest, GradientDescentDPOneIterationZeroNoiseSucceeds) {
  // Testing using a similar dataset to balloons dataset.
  // The first column is for the offset.
  std::vector<uint64_t> data_x = {
      1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
      1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0,
      1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
      1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1,
      1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0};

  // Outcome.
  const std::vector<int> label_y = {1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
                                    1, 1, 0, 0, 0, 1, 1, 0, 0, 0};

  // zero = 0 and represented as 0*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto zero,
                       fp_factory_->CreateFixedPointElementFromInt(0));
  // one = 1 and represented as 1*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto one,
                       fp_factory_->CreateFixedPointElementFromInt(1));

  std::vector<uint64_t> data_x_ring(data_x.size(), zero.ExportToUint64());
  for (size_t idx = 0; idx < data_x.size(); idx++) {
    if (data_x[idx] == 1) {
      data_x_ring[idx] = one.ExportToUint64();
    }
  }

  std::vector<uint64_t> label_y_ring(label_y.size(), zero.ExportToUint64());
  for (size_t idx = 0; idx < label_y.size(); idx++) {
    if (label_y[idx] == 1) {
      label_y_ring[idx] = one.ExportToUint64();
    }
  }

  // Generate parameters for gradient descent.
  const GradientDescentParams gd_params = {
      .num_examples = label_y.size(),  // 20
      .num_features = data_x.size() / label_y.size(),
      .num_iterations = 1,
      .num_ring_bits = kNumRingBits,
      .num_fractional_bits = kNumFractionalBits,
      .alpha = 18,  // Scalar multiplication will be 18/20 = 0.9
      .lambda = 0,
      .modulus = (1ULL << kNumRingBits)};

  const FixedPointElementFactory::Params fpe_params = {
      .num_fractional_bits = kNumFractionalBits,
      .num_ring_bits = kNumRingBits,
      .fractional_multiplier = 1ULL << kNumFractionalBits,
      .integer_ring_modulus = 1ULL << (kNumRingBits - kNumFractionalBits),
      .primary_ring_modulus = 1ULL << kNumRingBits};


  // Sigmoid setup

  const size_t kLogGroupSize = 63;
  const uint64_t kIntervalCount = 10;
  const uint64_t kTaylorPolynomialDegree = 10;

  const std::vector<double> sigmoid_spline_lower_bounds{
      0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

  const std::vector<double> sigmoid_spline_upper_bounds = {
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

  const std::vector<double> sigmoid_spline_slope = {
      0.24979187478940013, 0.24854809833537939, 0.24608519499181072,
      0.24245143300792976, 0.23771671089402596, 0.23196975023940808,
      0.2253146594237077, 0.2178670895944635, 0.20975021497391394,
      0.2010907600500101};

  const std::vector<double> sigmoid_spline_yIntercept = {0.5,
                                                         0.5001243776454021,
                                                         0.5006169583141158,
                                                         0.5017070869092801,
                                                         0.5036009757548416,
                                                         0.5064744560821506,
                                                         0.5104675105715708,
                                                         0.5156808094520418,
                                                         0.5221743091484814,
                                                         0.5299678185799949};

  const applications::SecureSplineParameters sigmoid_spline_params{
      kLogGroupSize,
      kIntervalCount,
      kNumFractionalBits,
      sigmoid_spline_lower_bounds,
      sigmoid_spline_upper_bounds,
      sigmoid_spline_slope,
      sigmoid_spline_yIntercept
  };

  const uint64_t kLargePrime = 9223372036854775783;  // 63 bit prime

  // TODO : The first param in kSampleLargeExpParams represents
  // the exponent bound. Set it up judiciously

  //            const ExponentiationParams kSampleLargeExpParams = {
  //                    2 * fixed_point_factory_->num_fractional_bits,
  //                    kLargePrime};

  const ExponentiationParams kSampleLargeExpParams = {
      13,
      kLargePrime};

  const applications::SecureSigmoidParameters sigmoid_params {
      kLogGroupSize,
      sigmoid_spline_params,
      kNumFractionalBits,
      kTaylorPolynomialDegree,
      kSampleLargeExpParams
  };

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<applications::SecureSigmoid> secure_sigmoid,
  applications::SecureSigmoid::Create(gd_params.num_examples, sigmoid_params));

  // Secret share inputs.

  // Sharing data (data_x).
  ASSERT_OK_AND_ASSIGN(
      auto share_x,
      internal::SampleShareOfZero(data_x_ring.size(), gd_params.modulus));
  ASSERT_OK_AND_ASSIGN(
      auto share_x_p0,
      BatchedModAdd(share_x.first, data_x_ring, gd_params.modulus));
  auto share_x_p1 = std::move(share_x.second);

  // Sharing original_labels (label_y).
  ASSERT_OK_AND_ASSIGN(
      auto share_y,
      internal::SampleShareOfZero(label_y_ring.size(), gd_params.modulus));
  ASSERT_OK_AND_ASSIGN(
      auto share_y_p0,
      BatchedModAdd(share_y.first, label_y_ring, gd_params.modulus));
  auto share_y_p1 = std::move(share_y.second);

  // Initialize theta to 0.
  std::vector<double> theta(gd_params.num_features, 0);

  // Initialize preprocessed shares in trusted setup.
  std::vector<applications::SigmoidPrecomputedValue> sigmoid_p0;
  std::vector<uint64_t> beaver_triple_matrix_a_p0;
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
  beaver_triple_matrix_b_c_p0;
  std::vector<applications::SigmoidPrecomputedValue> sigmoid_p1;
  std::vector<uint64_t> beaver_triple_matrix_a_p1;
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
  beaver_triple_matrix_b_c_p1;

  // Initialize DP noise also in trusted setup (set dp noise to 0 for now)
  std::vector<std::vector<uint64_t>> share_noise_p0 (gd_params.num_iterations,
                                                     std::vector<uint64_t>(gd_params.num_features, 0));
  std::vector<std::vector<uint64_t>> share_noise_p1 (gd_params.num_iterations,
                                                   std::vector<uint64_t>(gd_params.num_features, 0));

//  std::vector<uint64_t> noise (gd_params.num_features, 0);
//  ASSERT_OK_AND_ASSIGN(
//      auto share_noise,
//      internal::SampleShareOfZero(gd_params.num_features, gd_params.modulus));
//  ASSERT_OK_AND_ASSIGN(
//      auto share_one_noise_p0,
//      BatchedModAdd(share_noise.first, noise, gd_params.modulus));
//  auto share_one_noise_p1 = std::move(share_noise.second);
//  std::vector<std::vector<uint64_t>> share_noise_p0 (gd_params.num_iterations,
//                                                     share_one_noise_p0);
//  std::vector<std::vector<uint64_t>> share_noise_p1 (gd_params.num_iterations,
//                                                     share_one_noise_p1);

  for (unsigned int idx = 0; idx < gd_params.num_iterations; idx++) {
    ASSERT_OK_AND_ASSIGN(
        auto share_noise,
        internal::SampleShareOfZero(gd_params.num_features, gd_params.modulus)
    );
    share_noise_p0[idx] = std::move(share_noise.first);
    share_noise_p1[idx] = std::move(share_noise.second);
  }

  // sigmoid(u): u (#examples x 1)
  std::pair<applications::SigmoidPrecomputedValue, applications::SigmoidPrecomputedValue> preCompRes;
  ASSERT_OK_AND_ASSIGN(preCompRes,
    secure_sigmoid->PerformSigmoidPrecomputation());
  sigmoid_p0.push_back(std::move(preCompRes.first));
  sigmoid_p1.push_back(std::move(preCompRes.second));

  // X.transpose() * d: X.transpose() (#features * #examples), d (#examples x
  // 1).
  // Since only 1 iteration is run, no need to generate mask for X^T separately.
  ASSERT_OK_AND_ASSIGN(
      auto matrix_shares,
      SampleBeaverTripleMatrix(gd_params.num_features, gd_params.num_examples,
        1, gd_params.modulus));
  beaver_triple_matrix_a_p0 = std::move(matrix_shares.first.GetA());
  beaver_triple_matrix_b_c_p0.push_back(std::make_pair(
      matrix_shares.first.GetB(), matrix_shares.first.GetC()));
  beaver_triple_matrix_a_p1 = std::move(matrix_shares.second.GetA());
  beaver_triple_matrix_b_c_p1.push_back(std::make_pair(
      matrix_shares.second.GetB(), matrix_shares.second.GetC()));

  // Initialize LogRegDPShareProvider with preprocessed shares and noise.
  ASSERT_OK_AND_ASSIGN(
      auto logreg_share_provider_p0,
      LogRegDPShareProvider::Create(
      sigmoid_p0, beaver_triple_matrix_a_p0,
      beaver_triple_matrix_b_c_p0, share_noise_p0, gd_params.num_examples,
      gd_params.num_features, gd_params.num_iterations));
  ASSERT_OK_AND_ASSIGN(
      auto logreg_share_provider_p1,
      LogRegDPShareProvider::Create(
      sigmoid_p1, beaver_triple_matrix_a_p1,
      beaver_triple_matrix_b_c_p1, share_noise_p1, gd_params.num_examples,
      gd_params.num_features, gd_params.num_iterations));

  // Initialize GradientDescentPartyZero/One with input and preprocessed shares.
  ASSERT_OK_AND_ASSIGN(
      auto gd_party_zero,
      GradientDescentPartyZero::Init(
      share_x_p0, share_y_p0, theta,
      absl::make_unique<LogRegDPShareProvider>(logreg_share_provider_p0),
      fpe_params, gd_params));
  ASSERT_OK_AND_ASSIGN(
      auto gd_party_one,
      GradientDescentPartyOne::Init(
      share_x_p1, share_y_p1, theta,
      absl::make_unique<LogRegDPShareProvider>(logreg_share_provider_p1),
      fpe_params, gd_params));

  // Generate messages for masked X^T for the correlated product.

  ASSERT_OK_AND_ASSIGN(
      auto p0_return_masked_x_transpose,
      gd_party_zero.GenerateCorrelatedProductMessageForXTranspose());
  ASSERT_OK_AND_ASSIGN(
      auto p1_return_masked_x_transpose,
      gd_party_one.GenerateCorrelatedProductMessageForXTranspose());

  // Running gradient descent for logistic regression with dp.

  // Compute X * theta and Get Sigmoid Inputs
  ASSERT_OK_AND_ASSIGN(SigmoidInput p0_sigmoid_input,
    gd_party_zero.GenerateSigmoidInput());
  ASSERT_OK_AND_ASSIGN(SigmoidInput p1_sigmoid_input,
    gd_party_one.GenerateSigmoidInput());

  // Compute Sigmoid

  ASSERT_OK_AND_ASSIGN(applications::SigmoidPrecomputedValue gd_party_zero_share,
                       gd_party_zero.share_provider_->GetSigmoidPrecomputedValue());
  ASSERT_OK_AND_ASSIGN(applications::SigmoidPrecomputedValue gd_party_one_share,
                       gd_party_one.share_provider_->GetSigmoidPrecomputedValue());

  // GenerateSigmoidRoundOneMessage

  std::pair<applications::RoundOneSigmoidState, applications::RoundOneSigmoidMessage> round_one_res_party0;
  std::pair<applications::RoundOneSigmoidState, applications::RoundOneSigmoidMessage> round_one_res_party1;

  ASSERT_OK_AND_ASSIGN(
      round_one_res_party0,
      secure_sigmoid->GenerateSigmoidRoundOneMessage(0,
        gd_party_zero_share,
        p0_sigmoid_input.sigmoid_input));

  ASSERT_OK_AND_ASSIGN(
      round_one_res_party1,
      secure_sigmoid->GenerateSigmoidRoundOneMessage(1,
      gd_party_one_share,
      p1_sigmoid_input.sigmoid_input));

  // GenerateSigmoidRoundTwoMessage

  std::pair<applications::RoundTwoSigmoidState, applications::RoundTwoSigmoidMessage> round_two_res_party0,
      round_two_res_party1;

  ASSERT_OK_AND_ASSIGN(
      round_two_res_party0,
      secure_sigmoid->GenerateSigmoidRoundTwoMessage(0,
        gd_party_zero_share,
        round_one_res_party0.first,
        round_one_res_party1.second));

  ASSERT_OK_AND_ASSIGN(
      round_two_res_party1,
      secure_sigmoid->GenerateSigmoidRoundTwoMessage(1,
        gd_party_one_share,
        round_one_res_party1.first,
        round_one_res_party0.second));

  // GenerateSigmoidRoundThreeMessage

  std::pair<applications::RoundThreeSigmoidState, applications::RoundThreeSigmoidMessage> round_three_res_party0,
      round_three_res_party1;

  ASSERT_OK_AND_ASSIGN(
      round_three_res_party0,
      secure_sigmoid->GenerateSigmoidRoundThreeMessage(0,
        gd_party_zero_share,
        round_two_res_party0.first,
        round_two_res_party1.second));

  ASSERT_OK_AND_ASSIGN(
      round_three_res_party1,
      secure_sigmoid->GenerateSigmoidRoundThreeMessage(1,
        gd_party_one_share,
        round_two_res_party1.first,
        round_two_res_party0.second));

  // GenerateSigmoidRoundFourMessage

  std::pair<applications::RoundFourSigmoidState, applications::RoundFourSigmoidMessage> round_four_res_party0,
      round_four_res_party1;

  ASSERT_OK_AND_ASSIGN(
      round_four_res_party0,
      secure_sigmoid->GenerateSigmoidRoundFourMessage(0,
        gd_party_zero_share,
        round_three_res_party0.first,
        round_three_res_party1.second));

  ASSERT_OK_AND_ASSIGN(
      round_four_res_party1,
      secure_sigmoid->GenerateSigmoidRoundFourMessage(1,
        gd_party_one_share,
        round_three_res_party1.first,
        round_three_res_party0.second));

  // GenerateSigmoidResult

  std::vector<uint64_t> final_sigmoid_outputs_share_party0,
      final_sigmoid_outputs_share_party1;

  ASSERT_OK_AND_ASSIGN(
      final_sigmoid_outputs_share_party0,
      secure_sigmoid->GenerateSigmoidResult(0,
        gd_party_zero_share,
        round_four_res_party0.first,
        round_four_res_party1.second));

  ASSERT_OK_AND_ASSIGN(
      final_sigmoid_outputs_share_party1,
      secure_sigmoid->GenerateSigmoidResult(1,
        gd_party_one_share,
        round_four_res_party1.first,
        round_four_res_party0.second));

  SigmoidOutput p0_sigmoid_output = {
      .sigmoid_output = std::move(final_sigmoid_outputs_share_party0)
  };
  SigmoidOutput p1_sigmoid_output = {
      .sigmoid_output = std::move(final_sigmoid_outputs_share_party1)
  };

  // Insecure testing function

  //ASSERT_OK_AND_ASSIGN(auto sigmoid_outputs,
  //                     gd_party_zero.GenerateSigmoidOutputForTesting(
  //                         p0_sigmoid_input, p1_sigmoid_input));
  //SigmoidOutput p0_sigmoid_output = sigmoid_outputs.first;
  //SigmoidOutput p1_sigmoid_output = sigmoid_outputs.second;

  // X^transpose * d round
  ASSERT_OK_AND_ASSIGN(
      auto p0_return_x_transpose_d,
      gd_party_zero.GenerateXTransposeDMessage(
        p0_sigmoid_output, p0_return_masked_x_transpose.first));
  ASSERT_OK_AND_ASSIGN(
      auto p1_return_x_transpose_d,
      gd_party_one.GenerateXTransposeDMessage(
        p1_sigmoid_output, p1_return_masked_x_transpose.first));

  // Reconstruct gradient round
  ASSERT_OK_AND_ASSIGN(
      auto p0_return_reconstruct_gradient,
        gd_party_zero.GenerateReconstructGradientMessage(
        p0_return_x_transpose_d.first, p1_return_x_transpose_d.second,
        p1_return_masked_x_transpose.second));
  ASSERT_OK_AND_ASSIGN(
      auto p1_return_reconstruct_gradient,
      gd_party_one.GenerateReconstructGradientMessage(
      p1_return_x_transpose_d.first, p0_return_x_transpose_d.second,
      p0_return_masked_x_transpose.second));

  // Compute gradient descent update.
  ASSERT_OK(gd_party_zero.ComputeGradientUpdate(
    p0_return_reconstruct_gradient.first, p1_return_reconstruct_gradient.second));
  ASSERT_OK(gd_party_one.ComputeGradientUpdate(
    p1_return_reconstruct_gradient.first, p0_return_reconstruct_gradient.second));

  // Check if the value of the parameters makes sense.
  // For the derivative of the balloons dataset used in this test,
  // we ran an identical plaintext algorithm and hardcoded the theta outputs.
  // For 1 iteration and learning rate alpha = 18, we expect:
  // theta: -1.8,-0.9,-0.9,-3.6,1.8
  std::vector<double> approx_theta{-1.8, -0.9, -0.9, -3.6, 1.8};
  std::vector<double> theta_p0 = gd_party_zero.GetTheta();
  std::vector<double> theta_p1 = gd_party_one.GetTheta();
  for (size_t idx = 0; idx < theta.size(); idx++) {
    ASSERT_NEAR(theta_p0[idx], approx_theta[idx], 0.01);
    ASSERT_NEAR(theta_p1[idx], approx_theta[idx], 0.01);
  }
}

//
// MULTIPLE ITERATIONS (NO REGULARIZATION)
//

TEST_F(GradientDescentDPTest, GradientDescentDPFinalModelNoRegularizationSucceeds) {
  // Testing using a similar dataset to balloons dataset.
  // The first column is for the offset.
  std::vector<uint64_t> data_x = {
      1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
      1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0,
      1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
      1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1,
      1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0};

  // Outcome.
  const std::vector<int> label_y = {1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
                                    1, 1, 0, 0, 0, 1, 1, 0, 0, 0};

  // zero = 0 and represented as 0*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto zero,
                       fp_factory_->CreateFixedPointElementFromInt(0));
  // one = 1 and represented as 1*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto one,
                       fp_factory_->CreateFixedPointElementFromInt(1));

  std::vector<uint64_t> data_x_ring (data_x.size(), zero.ExportToUint64());
  for (size_t idx = 0; idx < data_x.size(); idx++) {
    if (data_x[idx] == 1) {
      data_x_ring[idx] = one.ExportToUint64();
    }
  }

  std::vector<uint64_t> label_y_ring (label_y.size(), zero.ExportToUint64());
  for (size_t idx = 0; idx < label_y.size(); idx++) {
    if (label_y[idx] == 1) {
      label_y_ring[idx] = one.ExportToUint64();
    }
  }

  // Generate parameters for gradient descent.
  const GradientDescentParams gd_params = {
      .num_examples = label_y.size(),  // 20
      .num_features = data_x.size() / label_y.size(),
      .num_iterations = kNumIteration,
      .num_ring_bits = kNumRingBits,
      .num_fractional_bits = kNumFractionalBits,
      .alpha = 18,  // Scalar multiplication will be 18/20 = 0.9
      .lambda = 0,
      .modulus = (1ULL << kNumRingBits)
  };

  const FixedPointElementFactory::Params fpe_params = {
      .num_fractional_bits = kNumFractionalBits,
      .num_ring_bits = kNumRingBits,
      .fractional_multiplier = 1ULL << kNumFractionalBits,
      .integer_ring_modulus = 1ULL << (kNumRingBits - kNumFractionalBits),
      .primary_ring_modulus = 1ULL << kNumRingBits
  };

  // Sigmoid setup

  const size_t kLogGroupSize = 63;
  const uint64_t kIntervalCount = 10;
  const uint64_t kTaylorPolynomialDegree = 10;

  const std::vector<double> sigmoid_spline_lower_bounds{
      0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

  const std::vector<double> sigmoid_spline_upper_bounds = {
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

  const std::vector<double> sigmoid_spline_slope = {
      0.24979187478940013, 0.24854809833537939, 0.24608519499181072,
      0.24245143300792976, 0.23771671089402596, 0.23196975023940808,
      0.2253146594237077, 0.2178670895944635, 0.20975021497391394,
      0.2010907600500101};

  const std::vector<double> sigmoid_spline_yIntercept = {0.5,
                                                         0.5001243776454021,
                                                         0.5006169583141158,
                                                         0.5017070869092801,
                                                         0.5036009757548416,
                                                         0.5064744560821506,
                                                         0.5104675105715708,
                                                         0.5156808094520418,
                                                         0.5221743091484814,
                                                         0.5299678185799949};

  const applications::SecureSplineParameters sigmoid_spline_params{
      kLogGroupSize,
      kIntervalCount,
      kNumFractionalBits,
      sigmoid_spline_lower_bounds,
      sigmoid_spline_upper_bounds,
      sigmoid_spline_slope,
      sigmoid_spline_yIntercept
  };

  const uint64_t kLargePrime = 9223372036854775783;  // 63 bit prime

  // TODO : The first param in kSampleLargeExpParams represents
  // the exponent bound. Set it up judiciously

  //            const ExponentiationParams kSampleLargeExpParams = {
  //                    2 * fixed_point_factory_->num_fractional_bits,
  //                    kLargePrime};

  const ExponentiationParams kSampleLargeExpParams = {
      13,
      kLargePrime};

  const applications::SecureSigmoidParameters sigmoid_params {
      kLogGroupSize,
      sigmoid_spline_params,
      kNumFractionalBits,
      kTaylorPolynomialDegree,
      kSampleLargeExpParams
  };

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<applications::SecureSigmoid> secure_sigmoid,
                       applications::SecureSigmoid::Create(gd_params.num_examples, sigmoid_params));

  // Secret share inputs.

  // Sharing data (data_x).
  ASSERT_OK_AND_ASSIGN(
      auto share_x,
      internal::SampleShareOfZero(data_x_ring.size(), gd_params.modulus));
  ASSERT_OK_AND_ASSIGN(
      auto share_x_p0,
      BatchedModAdd(share_x.first, data_x_ring, gd_params.modulus));
  auto share_x_p1 = std::move(share_x.second);

  // Sharing data (label_y).
  ASSERT_OK_AND_ASSIGN(
      auto share_y,
      internal::SampleShareOfZero(label_y_ring.size(), gd_params.modulus));
  ASSERT_OK_AND_ASSIGN(
      auto share_y_p0,
      BatchedModAdd(share_y.first, label_y_ring, gd_params.modulus));
  auto share_y_p1 = std::move(share_y.second);

  // Initialize theta to 0.
  std::vector<double> theta_p0(gd_params.num_features, 0);
  std::vector<double> theta_p1(gd_params.num_features, 0);

  // Initialize preprocessed shares and noise in trusted setup.
  std::vector<applications::SigmoidPrecomputedValue> sigmoid_p0;
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
  beaver_triple_matrix_b_c_p0;
  std::vector<applications::SigmoidPrecomputedValue> sigmoid_p1;
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
  beaver_triple_matrix_b_c_p1;

  // Initialize DP noise also in trusted setup (set dp noise to 0 for now)
  std::vector<std::vector<uint64_t>> share_noise_p0 (gd_params.num_iterations,
                                                     std::vector<uint64_t>(gd_params.num_features, 0));
  std::vector<std::vector<uint64_t>> share_noise_p1 (gd_params.num_iterations,
                                                     std::vector<uint64_t>(gd_params.num_features, 0));

  // maybe a better way to sample noise of 0:
//  std::vector<uint64_t> noise (gd_params.num_features, 0);
//  ASSERT_OK_AND_ASSIGN(
//      auto share_noise,
//      internal::SampleShareOfZero(gd_params.num_iterations, gd_params.modulus));
//  ASSERT_OK_AND_ASSIGN(
//      auto share_one_noise_p0,
//      BatchedModAdd(share_noise.first, noise, gd_params.modulus));
//  auto share_one_noise_p1 = std::move(share_noise.second);
//  std::vector<std::vector<uint64_t>> share_noise_p0 (gd_params.num_iterations,
//                                                     share_one_noise_p0);
//  std::vector<std::vector<uint64_t>> share_noise_p1 (gd_params.num_iterations,
//                                                     share_one_noise_p1);

  // Generates mask for X^T. These masks are part of the correlated
  // matrix product and do not change across iterations
  ASSERT_OK_AND_ASSIGN(
      auto vector_a_masks_x_transpose,
      SampleBeaverTripleMatrixA(gd_params.num_examples, gd_params.num_features,
      gd_params.modulus));

  // Create precomputed stuff for each iteration
  for (size_t idx = 0; idx < gd_params.num_iterations; idx++) {
        // sigmoid(u): u (#examples x 1)
        std::pair<applications::SigmoidPrecomputedValue, applications::SigmoidPrecomputedValue> preCompRes;
        ASSERT_OK_AND_ASSIGN(preCompRes,
                             secure_sigmoid->PerformSigmoidPrecomputation());
        sigmoid_p0.push_back(std::move(preCompRes.first));
        sigmoid_p1.push_back(std::move(preCompRes.second));

        // X.transpose() * d: X.transpose() (#features * #examples),
        // d (#examples x 1).
        ASSERT_OK_AND_ASSIGN(auto matrix_shares,
                             SampleBeaverTripleMatrixBandC(
                             vector_a_masks_x_transpose, gd_params.num_features,
        gd_params.num_examples, 1, gd_params.modulus));
        beaver_triple_matrix_b_c_p0.push_back(matrix_shares.first);
        beaver_triple_matrix_b_c_p1.push_back(matrix_shares.second);

        // DP noise
        ASSERT_OK_AND_ASSIGN(
            auto share_noise,
            internal::SampleShareOfZero(gd_params.num_features, gd_params.modulus)
        );
        share_noise_p0[idx] = std::move(share_noise.first);
        share_noise_p1[idx] = std::move(share_noise.second);
  }

  // Initialize LogRegDPShareProvider with preprocessed shares.
  ASSERT_OK_AND_ASSIGN(
      auto logreg_share_provider_p0,
      LogRegDPShareProvider::Create(
          sigmoid_p0, std::get<0>(vector_a_masks_x_transpose),
          beaver_triple_matrix_b_c_p0, share_noise_p0, gd_params.num_examples,
          gd_params.num_features, gd_params.num_iterations));
  ASSERT_OK_AND_ASSIGN(
      auto logreg_share_provider_p1,
      LogRegDPShareProvider::Create(
          sigmoid_p1, std::get<1>(vector_a_masks_x_transpose),
          beaver_triple_matrix_b_c_p1, share_noise_p1, gd_params.num_examples,
          gd_params.num_features, gd_params.num_iterations));

  // Initialize GradientDescentPartyZero/One with input and preprocessed shares.
  ASSERT_OK_AND_ASSIGN(
      auto gd_party_zero,
      GradientDescentPartyZero::Init(
          share_x_p0, share_y_p0, theta_p0,
          absl::make_unique<LogRegDPShareProvider>(logreg_share_provider_p0),
          fpe_params, gd_params));
  ASSERT_OK_AND_ASSIGN(
      auto gd_party_one,
      GradientDescentPartyOne::Init(
          share_x_p1, share_y_p1, theta_p1,
          absl::make_unique<LogRegDPShareProvider>(logreg_share_provider_p1),
          fpe_params, gd_params));

  // Generate messages for masked X^T for the correlated product.

  ASSERT_OK_AND_ASSIGN(
      auto p0_return_masked_x_transpose,
      gd_party_zero.GenerateCorrelatedProductMessageForXTranspose());
  ASSERT_OK_AND_ASSIGN(
      auto p1_return_masked_x_transpose,
      gd_party_one.GenerateCorrelatedProductMessageForXTranspose());

  // Running gradient descent for logistic regression.
  for (size_t iter = 0; iter < gd_params.num_iterations; iter++) {
    // Compute X * theta and Get Sigmoid Inputs
    ASSERT_OK_AND_ASSIGN(SigmoidInput p0_sigmoid_input,
                         gd_party_zero.GenerateSigmoidInput());
    ASSERT_OK_AND_ASSIGN(SigmoidInput p1_sigmoid_input,
                       gd_party_one.GenerateSigmoidInput());

    //SigmoidInput p0_sigmoid_input = {.sigmoid_input = std::move(std::vector<uint64_t>(gd_params.num_examples, 0))};
    //SigmoidInput p1_sigmoid_input = {.sigmoid_input = std::move(std::vector<uint64_t>(gd_params.num_examples, 0))};

    // Compute Sigmoid

    ASSERT_OK_AND_ASSIGN(applications::SigmoidPrecomputedValue gd_party_zero_share,
                         gd_party_zero.share_provider_->GetSigmoidPrecomputedValue());
    ASSERT_OK_AND_ASSIGN(applications::SigmoidPrecomputedValue gd_party_one_share,
                         gd_party_one.share_provider_->GetSigmoidPrecomputedValue());

    // GenerateSigmoidRoundOneMessage

    std::pair<applications::RoundOneSigmoidState, applications::RoundOneSigmoidMessage> round_one_res_party0;
    std::pair<applications::RoundOneSigmoidState, applications::RoundOneSigmoidMessage> round_one_res_party1;

    ASSERT_OK_AND_ASSIGN(
        round_one_res_party0,
        secure_sigmoid->GenerateSigmoidRoundOneMessage(0,
                                                       gd_party_zero_share,
                                                       p0_sigmoid_input.sigmoid_input));

    ASSERT_OK_AND_ASSIGN(
        round_one_res_party1,
        secure_sigmoid->GenerateSigmoidRoundOneMessage(1,
                                                       gd_party_one_share,
                                                       p1_sigmoid_input.sigmoid_input));

    // GenerateSigmoidRoundTwoMessage

    std::pair<applications::RoundTwoSigmoidState, applications::RoundTwoSigmoidMessage> round_two_res_party0,
        round_two_res_party1;

    ASSERT_OK_AND_ASSIGN(
        round_two_res_party0,
        secure_sigmoid->GenerateSigmoidRoundTwoMessage(0,
                                                       gd_party_zero_share,
                                                       round_one_res_party0.first,
                                                       round_one_res_party1.second));

    ASSERT_OK_AND_ASSIGN(
        round_two_res_party1,
        secure_sigmoid->GenerateSigmoidRoundTwoMessage(1,
                                                       gd_party_one_share,
                                                       round_one_res_party1.first,
                                                       round_one_res_party0.second));

    // GenerateSigmoidRoundThreeMessage

    std::pair<applications::RoundThreeSigmoidState, applications::RoundThreeSigmoidMessage> round_three_res_party0,
        round_three_res_party1;

    ASSERT_OK_AND_ASSIGN(
        round_three_res_party0,
        secure_sigmoid->GenerateSigmoidRoundThreeMessage(0,
                                                         gd_party_zero_share,
                                                         round_two_res_party0.first,
                                                         round_two_res_party1.second));

    ASSERT_OK_AND_ASSIGN(
        round_three_res_party1,
        secure_sigmoid->GenerateSigmoidRoundThreeMessage(1,
                                                         gd_party_one_share,
                                                         round_two_res_party1.first,
                                                         round_two_res_party0.second));

    // GenerateSigmoidRoundFourMessage

    std::pair<applications::RoundFourSigmoidState, applications::RoundFourSigmoidMessage> round_four_res_party0,
        round_four_res_party1;

    ASSERT_OK_AND_ASSIGN(
        round_four_res_party0,
        secure_sigmoid->GenerateSigmoidRoundFourMessage(0,
                                                        gd_party_zero_share,
                                                        round_three_res_party0.first,
                                                        round_three_res_party1.second));

    ASSERT_OK_AND_ASSIGN(
        round_four_res_party1,
        secure_sigmoid->GenerateSigmoidRoundFourMessage(1,
                                                        gd_party_one_share,
                                                        round_three_res_party1.first,
                                                        round_three_res_party0.second));

    // GenerateSigmoidResult

    std::vector<uint64_t> final_sigmoid_outputs_share_party0,
        final_sigmoid_outputs_share_party1;

    ASSERT_OK_AND_ASSIGN(
        final_sigmoid_outputs_share_party0,
        secure_sigmoid->GenerateSigmoidResult(0,
                                              gd_party_zero_share,
                                              round_four_res_party0.first,
                                              round_four_res_party1.second));

    ASSERT_OK_AND_ASSIGN(
        final_sigmoid_outputs_share_party1,
        secure_sigmoid->GenerateSigmoidResult(1,
                                              gd_party_one_share,
                                              round_four_res_party1.first,
                                              round_four_res_party0.second));

    SigmoidOutput p0_sigmoid_output = {
        .sigmoid_output = std::move(final_sigmoid_outputs_share_party0)
    };
    SigmoidOutput p1_sigmoid_output = {
        .sigmoid_output = std::move(final_sigmoid_outputs_share_party1)
    };

    // Insecure testing function

    //ASSERT_OK_AND_ASSIGN(auto sigmoid_outputs,
    //                     gd_party_zero.GenerateSigmoidOutputForTesting(
    //                         p0_sigmoid_input, p1_sigmoid_input));
    //SigmoidOutput p0_sigmoid_output = sigmoid_outputs.first;
    //SigmoidOutput p1_sigmoid_output = sigmoid_outputs.second;

    // X^transpose * d round
    ASSERT_OK_AND_ASSIGN(
        auto p0_return_x_transpose_d,
        gd_party_zero.GenerateXTransposeDMessage(
            p0_sigmoid_output, p0_return_masked_x_transpose.first));
    ASSERT_OK_AND_ASSIGN(
        auto p1_return_x_transpose_d,
        gd_party_one.GenerateXTransposeDMessage(
            p1_sigmoid_output, p1_return_masked_x_transpose.first));

    // Reconstruct gradient round
    ASSERT_OK_AND_ASSIGN(
        auto p0_return_reconstruct_gradient,
        gd_party_zero.GenerateReconstructGradientMessage(
            p0_return_x_transpose_d.first, p1_return_x_transpose_d.second,
            p1_return_masked_x_transpose.second));
    ASSERT_OK_AND_ASSIGN(
        auto p1_return_reconstruct_gradient,
        gd_party_one.GenerateReconstructGradientMessage(
            p1_return_x_transpose_d.first, p0_return_x_transpose_d.second,
            p0_return_masked_x_transpose.second));

    // Compute gradient descent update.
    ASSERT_OK(gd_party_zero.ComputeGradientUpdate(
        p0_return_reconstruct_gradient.first, p1_return_reconstruct_gradient.second));
    ASSERT_OK(gd_party_one.ComputeGradientUpdate(
        p1_return_reconstruct_gradient.first, p0_return_reconstruct_gradient.second));
  }

  // Check if the value of theta makes sense.
  // For the derivative of the balloons dataset used in this test,
  // we ran an identical plaintext algorithm and hardcoded the theta outputs.
  // For 5 iterations and learning rate alpha = 18, We expect:
  // 5 iters: -4.839549501873378,-2.4258426563571684,
  // -2.4258426563571676,-12.558532833003076,7.472449210787776

  std::vector<double> approx_theta{-4.839549501873378, -2.4258426563571684,
                                   -2.4258426563571676, -12.558532833003076,
                                   7.472449210787776};
  std::vector<double> final_theta_p0 = gd_party_zero.GetTheta();
  std::vector<double> final_theta_p1 = gd_party_one.GetTheta();

  for (size_t idx = 0; idx < gd_params.num_iterations; idx++) {
    ASSERT_NEAR(final_theta_p0[idx],
        approx_theta[idx], 0.5);
    ASSERT_NEAR(final_theta_p1[idx],
        approx_theta[idx], 0.5);
  }
}

}  // namespace
}  // namespace logistic_regression_dp
}  // namespace private_join_and_compute
