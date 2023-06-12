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

#include "applications/logistic_regression/gradient_descent.h"
#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
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
namespace logistic_regression {
namespace {

using internal::SampleShareOfZero;
using ::testing::Test;

const int kNumFractionalBits = 20;
const int kNumRingBits = 63;
const int kNumIteration = 5;

class GradientDescentNewMicTest : public Test {
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

std::vector<std::string> parseCSV(std::string name) {
  std::ifstream fl(name);
  std::vector<std::string> res;
  if (!fl.is_open()) {
    std::cerr << "Failed to open file" << std::endl;
    return res;
  }
  std::string ln;
  while (std::getline(fl, ln)) {
    std::stringstream ss(ln);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      res.push_back(cell);
    }
  }
  fl.close();
  return res;
}

//
// BALLOON DATASET
//

TEST_F(GradientDescentNewMicTest, GradientDescentNewMicFinalModelWithLambdaSucceeds) {
  // Testing using preprocessed titanic dataset
  std::vector<double> data_x;
  // Outcome.
  std::vector<int> label_y;

  // Read in data_x and label_y
  std::vector<std::string> X_string = parseCSV(
      "applications/logistic_regression/balloonX.csv");
  std::vector<std::string> y_string = parseCSV(
      "applications/logistic_regression/balloony.csv");

  for (const auto &elem: X_string) {
    data_x.push_back(std::stod(elem));
  }

  for (const auto &elem : y_string) {
    label_y.push_back(std::stoi(elem));
  }

  ASSERT_FALSE(data_x.empty());
  ASSERT_FALSE(label_y.empty());
  ASSERT_GE(data_x.size(), label_y.size());

  // Debug reading in
  std::cerr << "Num examples: " << label_y.size() << std::endl;
  std::cerr << "Num features: " << (data_x.size() / label_y.size()) << std::endl;

  // zero = 0 and represented as 0*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto zero,
                       fp_factory_->CreateFixedPointElementFromInt(0));
  // one = 1 and represented as 1*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto one,
                       fp_factory_->CreateFixedPointElementFromInt(1));

  std::vector<uint64_t> data_x_ring (data_x.size(), zero.ExportToUint64());
  for (size_t idx = 0; idx < data_x.size(); idx++) {
    ASSERT_OK_AND_ASSIGN(auto entry,
                         fp_factory_->CreateFixedPointElementFromDouble(data_x[idx]));
    data_x_ring[idx] = entry.ExportToUint64();
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
      .lambda = 0.1,
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

  size_t block_length = 16;
  size_t num_splits = 4;

  const applications::SecureSigmoidNewMicParameters sigmoid_params {
      kLogGroupSize,
      sigmoid_spline_params,
      kNumFractionalBits,
      kTaylorPolynomialDegree,
      kSampleLargeExpParams,
      block_length,
      num_splits
  };

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<applications::SecureSigmoidNewMic> secure_sigmoid,
                       applications::SecureSigmoidNewMic::Create(gd_params.num_examples, sigmoid_params));


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

  // Initialize share_theta to a valid share of 0.
  ASSERT_OK_AND_ASSIGN(
      auto share_theta,
      internal::SampleShareOfZero(gd_params.num_features, gd_params.modulus));
  std::vector<uint64_t> share_theta_p0 = std::move(share_theta.first);
  std::vector<uint64_t> share_theta_p1 = std::move(share_theta.second);

  // Initialize preprocessed shares in trusted setup.
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
      beaver_triple_matrix_b_c_round1_p0;
  std::vector<applications::SigmoidPrecomputedValueNewMic> sigmoid_p0;
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
      beaver_triple_matrix_b_c_round3_p0;
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
      beaver_triple_matrix_b_c_round1_p1;
  std::vector<applications::SigmoidPrecomputedValueNewMic> sigmoid_p1;
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
      beaver_triple_matrix_b_c_round3_p1;

  //  Create vectors with precomputed stuff for the sigmoid
  // for each iteration and then Create() SigmoidShareProvider and add to
  // sigmoid_round_2_p0 and sigmoid_round_2_p1

  // Generates mask for X and for X^T. These masks are part of the correlated
  // matrix product and do not change across iterations
  ASSERT_OK_AND_ASSIGN(
      auto vector_a_masks_x,
      SampleBeaverTripleMatrixA(gd_params.num_examples, gd_params.num_features,
                                gd_params.modulus));
  ASSERT_OK_AND_ASSIGN(
      auto vector_a_masks_x_transpose,
      SampleBeaverTripleMatrixA(gd_params.num_examples, gd_params.num_features,
                                gd_params.modulus));

  for (size_t idx = 0; idx < gd_params.num_iterations; idx++) {
    // u = X * Theta: X (#examples x #features), Theta (#features x 1).
    ASSERT_OK_AND_ASSIGN(auto matrix_shares_1,
                         SampleBeaverTripleMatrixBandC(
                             vector_a_masks_x, gd_params.num_examples,
                             gd_params.num_features, 1, gd_params.modulus));
    beaver_triple_matrix_b_c_round1_p0.push_back(matrix_shares_1.first);
    beaver_triple_matrix_b_c_round1_p1.push_back(matrix_shares_1.second);

    // sigmoid(u): u (#examples x 1)
    std::pair<applications::SigmoidPrecomputedValueNewMic, applications::SigmoidPrecomputedValueNewMic> preCompRes;
    ASSERT_OK_AND_ASSIGN(preCompRes,
                         secure_sigmoid->PerformSigmoidPrecomputation());
    sigmoid_p0.push_back(std::move(preCompRes.first));
    sigmoid_p1.push_back(std::move(preCompRes.second));


  // X.transpose() * d: X.transpose() (#features * #examples),
  // d (#examples x 1).
    ASSERT_OK_AND_ASSIGN(auto matrix_shares_3,
                         SampleBeaverTripleMatrixBandC(
                             vector_a_masks_x_transpose, gd_params.num_features,
                             gd_params.num_examples, 1, gd_params.modulus));
    beaver_triple_matrix_b_c_round3_p0.push_back(matrix_shares_3.first);
    beaver_triple_matrix_b_c_round3_p1.push_back(matrix_shares_3.second);
  }
  // Initialize LogRegShareProvider with preprocessed shares.
  ASSERT_OK_AND_ASSIGN(
      auto logreg_share_provider_p0,
      LogRegShareProvider::CreateNewMic(
          std::move(std::get<0>(vector_a_masks_x)), std::move(beaver_triple_matrix_b_c_round1_p0),
          std::move(sigmoid_p0), std::move(std::get<0>(vector_a_masks_x_transpose)),
          std::move(beaver_triple_matrix_b_c_round3_p0), gd_params.num_examples,
          gd_params.num_features, gd_params.num_iterations));
  ASSERT_OK_AND_ASSIGN(
      auto logreg_share_provider_p1,
      LogRegShareProvider::CreateNewMic(
          std::move(std::get<1>(vector_a_masks_x)), std::move(beaver_triple_matrix_b_c_round1_p1),
          std::move(sigmoid_p1), std::move(std::get<1>(vector_a_masks_x_transpose)),
          std::move(beaver_triple_matrix_b_c_round3_p1), gd_params.num_examples,
          gd_params.num_features, gd_params.num_iterations));

  // Initialize GradientDescentPartyZero/One with input and preprocessed shares.
  ASSERT_OK_AND_ASSIGN(
      auto gd_party_zero,
      GradientDescentPartyZero::Init(
          std::move(share_x_p0), std::move(share_y_p0), std::move(share_theta_p0),
          absl::make_unique<LogRegShareProvider>(std::move(logreg_share_provider_p0)),
          fpe_params, gd_params));
  ASSERT_OK_AND_ASSIGN(
      auto gd_party_one,
      GradientDescentPartyOne::Init(
          std::move(share_x_p1), std::move(share_y_p1), std::move(share_theta_p1),
          absl::make_unique<LogRegShareProvider>(std::move(logreg_share_provider_p1)),
          fpe_params, gd_params));

  // Generate messages for masked X and X^T for the correlated product.
  ASSERT_OK_AND_ASSIGN(auto p0_return_masked_x,
                       gd_party_zero.GenerateCorrelatedProductMessageForX());
  ASSERT_OK_AND_ASSIGN(auto p1_return_masked_x,
                       gd_party_one.GenerateCorrelatedProductMessageForX());

  ASSERT_OK_AND_ASSIGN(
      auto p0_return_masked_x_transpose,
      gd_party_zero.GenerateCorrelatedProductMessageForXTranspose());
  ASSERT_OK_AND_ASSIGN(
      auto p1_return_masked_x_transpose,
      gd_party_one.GenerateCorrelatedProductMessageForXTranspose());
  // Running gradient descent for logistic regression.
  for (size_t iter = 0; iter < gd_params.num_iterations; iter++) {
    // Round 1.
    ASSERT_OK_AND_ASSIGN(auto p0_return_round1,
                         gd_party_zero.GenerateGradientDescentRoundOneMessage(
                             p0_return_masked_x.first));
    ASSERT_OK_AND_ASSIGN(auto p1_return_round1,
                         gd_party_one.GenerateGradientDescentRoundOneMessage(
                             p1_return_masked_x.first));
    // Get Sigmoid Inputs
    ASSERT_OK_AND_ASSIGN(SigmoidInput p0_sigmoid_input,
                         gd_party_zero.GenerateSigmoidInput(
                             p0_return_round1.first, p1_return_masked_x.second,
                             p1_return_round1.second));
    ASSERT_OK_AND_ASSIGN(SigmoidInput p1_sigmoid_input,
                         gd_party_one.GenerateSigmoidInput(
                             p1_return_round1.first, p0_return_masked_x.second,
                             p0_return_round1.second));
    // Compute Sigmoid
    applications::SigmoidPrecomputedValueNewMic& gd_party_zero_share =
        gd_party_zero.share_provider_->GetSigmoidPrecomputedValueNewMic();
    applications::SigmoidPrecomputedValueNewMic& gd_party_one_share =
        gd_party_one.share_provider_->GetSigmoidPrecomputedValueNewMic();

    // GenerateSigmoidRoundOneMessage

    std::pair<applications::RoundOneSigmoidNewMicState, applications::RoundOneSigmoidNewMicMessage> round_one_res_party0;
    std::pair<applications::RoundOneSigmoidNewMicState, applications::RoundOneSigmoidNewMicMessage> round_one_res_party1;
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

    std::pair<applications::RoundTwoSigmoidNewMicState, applications::RoundTwoSigmoidNewMicMessage> round_two_res_party0,
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

    std::pair<applications::RoundThreeSigmoidNewMicState, applications::RoundThreeSigmoidNewMicMessage> round_three_res_party0,
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

    // GenerateSigmoidRoundThreeMessage

    std::pair<applications::RoundThreePointFiveSigmoidNewMicState, applications::RoundThreePointFiveSigmoidNewMicMessage> round_three_point_five_res_party0,
        round_three_point_five_res_party1;

    ASSERT_OK_AND_ASSIGN(
        round_three_point_five_res_party0,
        secure_sigmoid->GenerateSigmoidRoundThreePointFiveMessage(0,
                                                                  gd_party_zero_share,
                                                                  round_three_res_party0.first,
                                                                  round_three_res_party1.second));

    ASSERT_OK_AND_ASSIGN(
        round_three_point_five_res_party1,
        secure_sigmoid->GenerateSigmoidRoundThreePointFiveMessage(1,
                                                                  gd_party_one_share,
                                                                  round_three_res_party1.first,
                                                                  round_three_res_party0.second));


    // GenerateSigmoidRoundFourMessage

    std::pair<applications::RoundFourSigmoidNewMicState, applications::RoundFourSigmoidNewMicMessage> round_four_res_party0,
        round_four_res_party1;

    ASSERT_OK_AND_ASSIGN(
        round_four_res_party0,
        secure_sigmoid->GenerateSigmoidRoundFourMessage(0,
                                                        gd_party_zero_share,
                                                        round_three_point_five_res_party0.first,
                                                        round_three_point_five_res_party1.second));

    ASSERT_OK_AND_ASSIGN(
        round_four_res_party1,
        secure_sigmoid->GenerateSigmoidRoundFourMessage(1,
                                                        gd_party_one_share,
                                                        round_three_point_five_res_party1.first,
                                                        round_three_point_five_res_party0.second));
    // GenerateSigmoidRoundFiveMessage

    std::pair<applications::RoundFiveSigmoidNewMicState, applications::RoundFiveSigmoidNewMicMessage> round_five_res_party0,
        round_five_res_party1;

    ASSERT_OK_AND_ASSIGN(
        round_five_res_party0,
        secure_sigmoid->GenerateSigmoidRoundFiveMessage(0,
                                                        gd_party_zero_share,
                                                        round_four_res_party0.first,
                                                        round_four_res_party1.second));

    ASSERT_OK_AND_ASSIGN(
        round_five_res_party1,
        secure_sigmoid->GenerateSigmoidRoundFiveMessage(1,
                                                        gd_party_one_share,
                                                        round_four_res_party1.first,
                                                        round_four_res_party0.second));
    // GenerateSigmoidResult

    std::vector<uint64_t> final_sigmoid_outputs_share_party0,
        final_sigmoid_outputs_share_party1;

    ASSERT_OK_AND_ASSIGN(
        final_sigmoid_outputs_share_party0,
        secure_sigmoid->GenerateSigmoidResult(0,
                                              gd_party_zero_share,
                                              round_five_res_party0.first,
                                              round_five_res_party1.second));

    ASSERT_OK_AND_ASSIGN(
        final_sigmoid_outputs_share_party1,
        secure_sigmoid->GenerateSigmoidResult(1,
                                              gd_party_one_share,
                                              round_five_res_party1.first,
                                              round_five_res_party0.second));
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

    // Round 3.
    ASSERT_OK_AND_ASSIGN(
        auto p0_return_round3,
        gd_party_zero.GenerateGradientDescentRoundThreeMessage(
            p0_sigmoid_output, p0_return_masked_x_transpose.first));
    ASSERT_OK_AND_ASSIGN(
        auto p1_return_round3,
        gd_party_one.GenerateGradientDescentRoundThreeMessage(
            p1_sigmoid_output, p1_return_masked_x_transpose.first));

    // Compute gradient descent update.
    ASSERT_OK(gd_party_zero.ComputeGradientUpdate(
        p0_return_round3.first, p1_return_masked_x_transpose.second,
        p1_return_round3.second));
    ASSERT_OK(gd_party_one.ComputeGradientUpdate(
        p1_return_round3.first, p0_return_masked_x_transpose.second,
        p0_return_round3.second));
  }

  // Reconstruct theta, and check if the value of the parameters makes sense.
  // For the derivative of the balloons dataset used in this test,
  // we ran an identical plaintext algorithm and hardcoded the theta outputs.
  // For 5 iterations and learning rate alpha = 18, We expect:
  // 5 iters: -5.209011173974926,-2.6407028538582202,
  // -2.64070285385822,-10.915857622419917,5.392025544439257

  std::vector<double> approx_theta{-5.209011173974926,-2.6407028538582202,
                                   -2.64070285385822,-10.915857622419917,5.392025544439257};
  std::vector<uint64_t> updated_share_theta_p0 = gd_party_zero.GetTheta();
  std::vector<uint64_t> updated_share_theta_p1 = gd_party_one.GetTheta();
  ASSERT_OK_AND_ASSIGN(
      auto theta, BatchedModAdd(updated_share_theta_p0, updated_share_theta_p1,
                                gd_params.modulus));
  for (size_t idx = 0; idx < theta.size(); idx++) {
    ASSERT_NEAR(fp_factory_->ImportFixedPointElementFromUint64(theta[idx])
                    ->ExportToDouble(),
                approx_theta[idx], 0.5);
  }
}

// DATASETS FOR BENCHMARKING
// MINIBATCH
/*
TEST_F(GradientDescentNewMicTest, GradientDescentNewMicFinalModelWithLambdaSucceeds) {
  // Testing using preprocessed titanic dataset
  std::vector<double> data_x;
  // Outcome.
  std::vector<int> label_y;

  // Read in data_x and label_y
  std::vector<std::string> X_string = parseCSV(
      "applications/logistic_regression/criteo-uplift-x-option2-1000.csv");
  std::vector<std::string> y_string = parseCSV(
      "applications/logistic_regression/criteo-uplift-y-option2-1000.csv");

  // We want to train on 70% of the dataset

  size_t batch_size = 100; // User sets
  size_t original_num_examples = y_string.size();
  size_t training_num_examples = ceil(original_num_examples * 0.7);
  size_t num_features = (X_string.size() / y_string.size());
  size_t batch_iterations = training_num_examples / batch_size;
  size_t size_per_minibatch = num_features * batch_size;

  for (size_t idx = 0; idx < (num_features * training_num_examples); idx++) {
    data_x.push_back(std::stod(X_string[idx]));
  }

  //for (const auto &elem: X_string) {
  //  data_x.push_back(std::stod(elem));
  //}

  for (size_t idx = 0; idx < training_num_examples; idx++) {
    label_y.push_back(std::stoi(y_string[idx]));
  }
  //for (const auto &elem : y_string) {
  //  label_y.push_back(std::stoi(elem));
  //}

  ASSERT_FALSE(data_x.empty());
  ASSERT_FALSE(label_y.empty());
  ASSERT_GE(data_x.size(), label_y.size());

  // Debug reading in
  std::cerr << "Num examples: " << label_y.size() << std::endl;
  std::cerr << "Num features: " << (data_x.size() / label_y.size()) << std::endl;

  // zero = 0 and represented as 0*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto zero,
                       fp_factory_->CreateFixedPointElementFromInt(0));
  // one = 1 and represented as 1*2^{10}.
  ASSERT_OK_AND_ASSIGN(auto one,
                       fp_factory_->CreateFixedPointElementFromInt(1));

  std::vector<uint64_t> data_x_ring (data_x.size(), zero.ExportToUint64());
  for (size_t idx = 0; idx < data_x.size(); idx++) {
    ASSERT_OK_AND_ASSIGN(auto entry,
                         fp_factory_->CreateFixedPointElementFromDouble(data_x[idx]));
    data_x_ring[idx] = entry.ExportToUint64();
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
      .alpha = 1,  // TODO remember to update
      .lambda = 0.0001, // TODO remember to update
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

  size_t block_length = 16;
  size_t num_splits = 4;

  const applications::SecureSigmoidNewMicParameters sigmoid_params {
      kLogGroupSize,
      sigmoid_spline_params,
      kNumFractionalBits,
      kTaylorPolynomialDegree,
      kSampleLargeExpParams,
      block_length,
      num_splits
  };

  // Create batch size of sigmoids
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<applications::SecureSigmoidNewMic> secure_sigmoid,
                       applications::SecureSigmoidNewMic::Create(batch_size, sigmoid_params));

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

  // Initialize share_theta to a valid share of 0.
  ASSERT_OK_AND_ASSIGN(
      auto share_theta,
      internal::SampleShareOfZero(gd_params.num_features, gd_params.modulus));
  std::vector<uint64_t> share_theta_p0 = std::move(share_theta.first);
  std::vector<uint64_t> share_theta_p1 = std::move(share_theta.second);

  // Initialize preprocessed shares in trusted setup.
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
      beaver_triple_matrix_b_c_round1_p0;
  std::vector<applications::SigmoidPrecomputedValueNewMic> sigmoid_p0;
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
      beaver_triple_matrix_b_c_round3_p0;
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
      beaver_triple_matrix_b_c_round1_p1;
  std::vector<applications::SigmoidPrecomputedValueNewMic> sigmoid_p1;
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
      beaver_triple_matrix_b_c_round3_p1;

  //  Create vectors with precomputed stuff for the sigmoid
  // for each iteration and then Create() SigmoidShareProvider and add to
  // sigmoid_round_2_p0 and sigmoid_round_2_p1

  // Generates mask for X and for X^T. These masks are part of the correlated
  // matrix product and do not change across iterations
  ASSERT_OK_AND_ASSIGN(
      auto vector_a_masks_x,
      SampleBeaverTripleMatrixA(gd_params.num_examples, gd_params.num_features,
                                gd_params.modulus));
  ASSERT_OK_AND_ASSIGN(
      auto vector_a_masks_x_transpose,
      SampleBeaverTripleMatrixA(gd_params.num_examples, gd_params.num_features,
                                gd_params.modulus));

  // Split them into batch_size pieces
  std::vector<std::tuple<std::vector<uint64_t>, std::vector<uint64_t>,
                         std::vector<uint64_t>>> vector_a_masks_x_minibatch;
  std::vector<std::tuple<std::vector<uint64_t>, std::vector<uint64_t>,
                         std::vector<uint64_t>>> vector_a_masks_x_transpose_minibatch (batch_iterations);
  for (size_t idx_batch = 0; idx_batch < batch_iterations; idx_batch++) {
    std::vector<uint64_t> vec0 (size_per_minibatch);
    std::vector<uint64_t> vec1 (size_per_minibatch);
    std::vector<uint64_t> vec2 (size_per_minibatch);
    size_t start_idx = idx_batch * size_per_minibatch;
    for (size_t el = 0; el < size_per_minibatch; el++) {
      vec0[el] = std::get<0>(vector_a_masks_x)[start_idx + el];
      vec1[el] = std::get<1>(vector_a_masks_x)[start_idx + el];
      vec2[el] = std::get<2>(vector_a_masks_x)[start_idx + el];
    }
    //std::tuple<std::vector<uint64_t>, std::vector<uint64_t>,
    //           std::vector<uint64_t>> vector_a_masks_x_one_round = std::make_tuple(vec0, vec1, vec2);
    vector_a_masks_x_minibatch.push_back(std::make_tuple(vec0, vec1, vec2));
  }

  for (size_t idx_batch = 0; idx_batch < batch_iterations; idx_batch++) {
    std::vector<uint64_t> vec0 (size_per_minibatch);
    std::vector<uint64_t> vec1 (size_per_minibatch);
    std::vector<uint64_t> vec2 (size_per_minibatch);
    // size_t start_idx = idx_batch * size_per_minibatch;
    //for (size_t el = 0; el < size_per_minibatch; el++) {
    //  vec0[el] =  std::get<0>(vector_a_masks_x_transpose)[start_idx + el];
    //  vec1[el] =  std::get<1>(vector_a_masks_x_transpose)[start_idx + el];
    //  vec2[el] =  std::get<2>(vector_a_masks_x_transpose)[start_idx + el];
    //}

    for (size_t idx_feature = 0; idx_feature < num_features; idx_feature++) {
      for (size_t idx_batchsize = 0; idx_batchsize < batch_size; idx_batchsize++) {
        vec0[idx_feature * batch_size + idx_batchsize] =
            std::get<0>(vector_a_masks_x_transpose)[idx_feature * label_y.size() + idx_batch * batch_size + idx_batchsize];
        vec1[idx_feature * batch_size + idx_batchsize] =
            std::get<1>(vector_a_masks_x_transpose)[idx_feature * label_y.size() + idx_batch * batch_size + idx_batchsize];
        vec2[idx_feature * batch_size + idx_batchsize] =
            std::get<2>(vector_a_masks_x_transpose)[idx_feature * label_y.size() + idx_batch * batch_size + idx_batchsize];
      }
    }


    std::tuple<std::vector<uint64_t>, std::vector<uint64_t>,
               std::vector<uint64_t>> vector_a_masks_x_transpose_one_round = std::make_tuple(vec0, vec1, vec2);
    vector_a_masks_x_transpose_minibatch[idx_batch] = std::move(vector_a_masks_x_transpose_one_round);
  }

  for (size_t idx = 0; idx < gd_params.num_iterations; idx++) {
    for (size_t idx_batch = 0; idx_batch < batch_iterations; idx_batch++) {
      // u = X * Theta: X (#examples x #features), Theta (#features x 1).
      ASSERT_OK_AND_ASSIGN(auto matrix_shares_1,
                           SampleBeaverTripleMatrixBandC(
                               vector_a_masks_x_minibatch[batch_iterations - 1 - idx_batch], batch_size,
                               gd_params.num_features, 1, gd_params.modulus));
      beaver_triple_matrix_b_c_round1_p0.push_back(matrix_shares_1.first);
      beaver_triple_matrix_b_c_round1_p1.push_back(matrix_shares_1.second);

      // sigmoid(u): u (#examples x 1)
      std::pair<applications::SigmoidPrecomputedValueNewMic, applications::SigmoidPrecomputedValueNewMic> preCompRes;
      ASSERT_OK_AND_ASSIGN(preCompRes,
                           secure_sigmoid->PerformSigmoidPrecomputation());
      sigmoid_p0.push_back(std::move(preCompRes.first));
      sigmoid_p1.push_back(std::move(preCompRes.second));

      // X.transpose() * d: X.transpose() (#features * batch_size),
      // d (batch_size x 1).
      ASSERT_OK_AND_ASSIGN(auto matrix_shares_3,
                           SampleBeaverTripleMatrixBandC(
                               vector_a_masks_x_transpose_minibatch[batch_iterations - 1 - idx_batch], gd_params.num_features,
                               batch_size, 1, gd_params.modulus));
      beaver_triple_matrix_b_c_round3_p0.push_back(matrix_shares_3.first);
      beaver_triple_matrix_b_c_round3_p1.push_back(matrix_shares_3.second);
    }
  }
  // Initialize LogRegShareProvider with preprocessed shares.
  ASSERT_OK_AND_ASSIGN(
      auto logreg_share_provider_p0,
      LogRegShareProvider::CreateNewMic(
          std::move(std::get<0>(vector_a_masks_x)), std::move(beaver_triple_matrix_b_c_round1_p0),
          std::move(sigmoid_p0), std::move(std::get<0>(vector_a_masks_x_transpose)),
          std::move(beaver_triple_matrix_b_c_round3_p0), gd_params.num_examples,
          gd_params.num_features, gd_params.num_iterations * batch_iterations));
  ASSERT_OK_AND_ASSIGN(
      auto logreg_share_provider_p1,
      LogRegShareProvider::CreateNewMic(
          std::move(std::get<1>(vector_a_masks_x)), std::move(beaver_triple_matrix_b_c_round1_p1),
          std::move(sigmoid_p1), std::move(std::get<1>(vector_a_masks_x_transpose)),
          std::move(beaver_triple_matrix_b_c_round3_p1), gd_params.num_examples,
          gd_params.num_features, gd_params.num_iterations * batch_iterations));

  std::cerr << "preprocessing part done"<<std::endl;

  // Initialize GradientDescentPartyZero/One with input and preprocessed shares.
  ASSERT_OK_AND_ASSIGN(
      auto gd_party_zero,
      GradientDescentPartyZero::Init(
          std::move(share_x_p0), std::move(share_y_p0), std::move(share_theta_p0),
          absl::make_unique<LogRegShareProvider>(std::move(logreg_share_provider_p0)),
          fpe_params, gd_params));
  ASSERT_OK_AND_ASSIGN(
      auto gd_party_one,
      GradientDescentPartyOne::Init(
          std::move(share_x_p1), std::move(share_y_p1), std::move(share_theta_p1),
          absl::make_unique<LogRegShareProvider>(std::move(logreg_share_provider_p1)),
          fpe_params, gd_params));

  // Generate messages for masked X and X^T for the correlated product.
  ASSERT_OK_AND_ASSIGN(auto p0_return_masked_x,
                       gd_party_zero.GenerateCorrelatedProductMessageForX());
  ASSERT_OK_AND_ASSIGN(auto p1_return_masked_x,
                       gd_party_one.GenerateCorrelatedProductMessageForX());

  ASSERT_OK_AND_ASSIGN(
      auto p0_return_masked_x_transpose,
      gd_party_zero.GenerateCorrelatedProductMessageForXTranspose());
  ASSERT_OK_AND_ASSIGN(
      auto p1_return_masked_x_transpose,
      gd_party_one.GenerateCorrelatedProductMessageForXTranspose());

  // Since we are doing minibatch split p0/1_return_masked_x
  // and p0/1_return_masked_x_transpose into batch_size pieces
  // First state, then message
  // State:
  std::vector<StateMaskedX> p0_state_masked_x;
  std::vector<StateMaskedX> p1_state_masked_x;
  for (size_t idx_batch = 0; idx_batch < batch_iterations; idx_batch++) {
    std::vector<uint64_t> p0_share_x_minus_a (size_per_minibatch);
    std::vector<uint64_t> p1_share_x_minus_a (size_per_minibatch);
    size_t start_idx = idx_batch * size_per_minibatch;
    for (size_t el = 0; el < size_per_minibatch; el++) {
      p0_share_x_minus_a[el] =  p0_return_masked_x.first.share_x_minus_a[start_idx + el];
      p1_share_x_minus_a[el] =  p1_return_masked_x.first.share_x_minus_a[start_idx + el];
    }
    StateMaskedX p0_state_x_minus_a = {.share_x_minus_a =
    std::move(p0_share_x_minus_a)};
    StateMaskedX p1_state_x_minus_a = {.share_x_minus_a =
    std::move(p1_share_x_minus_a)};
    p0_state_masked_x.push_back(p0_state_x_minus_a);
    p1_state_masked_x.push_back(p1_state_x_minus_a);
  }

  std::vector<StateMaskedXTranspose> p0_state_masked_x_transpose;
  std::vector<StateMaskedXTranspose> p1_state_masked_x_transpose;
  for (size_t idx_batch = 0; idx_batch < batch_iterations; idx_batch++) {
    std::vector<uint64_t> p0_share_x_transpose_minus_a (size_per_minibatch);
    std::vector<uint64_t> p1_share_x_transpose_minus_a (size_per_minibatch);
    //size_t start_idx = idx_batch * size_per_minibatch;
    //for (size_t el = 0; el < size_per_minibatch; el++) {
    //  p0_share_x_transpose_minus_a[el] =  p0_return_masked_x_transpose.first.share_x_transpose_minus_a[start_idx + el];
    //  p1_share_x_transpose_minus_a[el] =  p1_return_masked_x_transpose.first.share_x_transpose_minus_a[start_idx + el];
    //}

    for (size_t idx_feature = 0; idx_feature < num_features; idx_feature++) {
      for (size_t idx_batchsize = 0; idx_batchsize < batch_size; idx_batchsize++) {
        p0_share_x_transpose_minus_a[idx_feature * batch_size + idx_batchsize] =
            p0_return_masked_x_transpose.first.share_x_transpose_minus_a[idx_feature * label_y.size() + idx_batch * batch_size + idx_batchsize];
        p1_share_x_transpose_minus_a[idx_feature * batch_size + idx_batchsize] =
            p1_return_masked_x_transpose.first.share_x_transpose_minus_a[idx_feature * label_y.size() + idx_batch * batch_size + idx_batchsize];
      }
    }

    StateMaskedXTranspose p0_state_x_transpose_minus_a = {.share_x_transpose_minus_a =
    std::move(p0_share_x_transpose_minus_a)};
    StateMaskedXTranspose p1_state_x_transpose_minus_a = {.share_x_transpose_minus_a =
    std::move(p1_share_x_transpose_minus_a)};
    p0_state_masked_x_transpose.push_back(p0_state_x_transpose_minus_a);
    p1_state_masked_x_transpose.push_back(p1_state_x_transpose_minus_a);
  }

  // Message:
  std::vector<MaskedXMessage> p0_message_masked_x (batch_iterations);
  std::vector<MaskedXMessage> p1_message_masked_x (batch_iterations);
  for (size_t idx_batch = 0; idx_batch < batch_iterations; idx_batch++) {
    size_t start_idx = idx_batch * size_per_minibatch;

    // Read the part of the full message into a vector (which belongs to the current batch)
    std::vector<uint64_t> p0_masked_x_minibatch_vec (size_per_minibatch);
    std::vector<uint64_t> p1_masked_x_minibatch_vec (size_per_minibatch);
    for (size_t el = 0; el < size_per_minibatch; el++) {
      p0_masked_x_minibatch_vec[el] = p0_return_masked_x.second.matrix_x_minus_matrix_a_message().matrix_x_minus_matrix_a_shares(start_idx + el);
      p1_masked_x_minibatch_vec[el] = p1_return_masked_x.second.matrix_x_minus_matrix_a_message().matrix_x_minus_matrix_a_shares(start_idx + el);
    }

    MatrixXminusAProductMessage p0_message_masked_x_minibatch_tmp;
    MatrixXminusAProductMessage p1_message_masked_x_minibatch_tmp;
    for (size_t idx = 0; idx < size_per_minibatch; idx++) {
      p0_message_masked_x_minibatch_tmp.add_matrix_x_minus_matrix_a_shares(p0_masked_x_minibatch_vec[idx]);
      p1_message_masked_x_minibatch_tmp.add_matrix_x_minus_matrix_a_shares(p1_masked_x_minibatch_vec[idx]);
    }
    // Copy message.
    *(p0_message_masked_x[idx_batch]).mutable_matrix_x_minus_matrix_a_message() =
        std::move(p0_message_masked_x_minibatch_tmp);
    *(p1_message_masked_x[idx_batch]).mutable_matrix_x_minus_matrix_a_message() =
        std::move(p1_message_masked_x_minibatch_tmp);
    //std::cerr << "check same" << std::endl;
    //std::cerr << p0_message_masked_x[idx_batch].matrix_x_minus_matrix_a_message().matrix_x_minus_matrix_a_shares(1) << std::endl;
    //std::cerr << p0_return_masked_x.second.matrix_x_minus_matrix_a_message().matrix_x_minus_matrix_a_shares(start_idx + 1) << std::endl;
  }

  std::vector<MaskedXTransposeMessage> p0_message_masked_x_transpose (batch_iterations);
  std::vector<MaskedXTransposeMessage> p1_message_masked_x_transpose (batch_iterations);
  for (size_t idx_batch = 0; idx_batch < batch_iterations; idx_batch++) {
    //size_t start_idx = idx_batch * size_per_minibatch;

    // Read the part of the full message into a vector (which belongs to the current batch)
    std::vector<uint64_t> p0_masked_x_transpose_minibatch_vec (size_per_minibatch);
    std::vector<uint64_t> p1_masked_x_transpose_minibatch_vec (size_per_minibatch);
    //for (size_t el = 0; el < size_per_minibatch; el++) {
    //  p0_masked_x_transpose_minibatch_vec[el] = p0_return_masked_x_transpose.second.matrix_x_transpose_minus_matrix_a_message().matrix_x_minus_matrix_a_shares(start_idx + el);
    //  p1_masked_x_transpose_minibatch_vec[el] = p1_return_masked_x_transpose.second.matrix_x_transpose_minus_matrix_a_message().matrix_x_minus_matrix_a_shares(start_idx + el);
    //}

    for (size_t idx_feature = 0; idx_feature < num_features; idx_feature++) {
      for (size_t idx_batchsize = 0; idx_batchsize < batch_size; idx_batchsize++) {
        p0_masked_x_transpose_minibatch_vec[idx_feature * batch_size + idx_batchsize] =
            p0_return_masked_x_transpose.second.matrix_x_transpose_minus_matrix_a_message().matrix_x_minus_matrix_a_shares(idx_feature * label_y.size() + idx_batch * batch_size + idx_batchsize);
        p1_masked_x_transpose_minibatch_vec[idx_feature * batch_size + idx_batchsize] =
            p1_return_masked_x_transpose.second.matrix_x_transpose_minus_matrix_a_message().matrix_x_minus_matrix_a_shares(idx_feature * label_y.size() + idx_batch * batch_size + idx_batchsize);
      }
    }

    MatrixXminusAProductMessage p0_message_masked_x_transpose_minibatch_tmp;
    MatrixXminusAProductMessage p1_message_masked_x_transpose_minibatch_tmp;
    for (size_t idx = 0; idx < size_per_minibatch; idx++) {
      p0_message_masked_x_transpose_minibatch_tmp.add_matrix_x_minus_matrix_a_shares(p0_masked_x_transpose_minibatch_vec[idx]);
      p1_message_masked_x_transpose_minibatch_tmp.add_matrix_x_minus_matrix_a_shares(p1_masked_x_transpose_minibatch_vec[idx]);
    }
    // Copy message.
    *(p0_message_masked_x_transpose[idx_batch]).mutable_matrix_x_transpose_minus_matrix_a_message() =
        std::move(p0_message_masked_x_transpose_minibatch_tmp);
    *(p1_message_masked_x_transpose[idx_batch]).mutable_matrix_x_transpose_minus_matrix_a_message() =
        std::move(p1_message_masked_x_transpose_minibatch_tmp);
  }

  // Running gradient descent for logistic regression.
  for (size_t iter = 0; iter < gd_params.num_iterations; iter++) {
    std::cerr << "IDX ITER " << iter << std::endl;
    for (size_t idx_batch = 0; idx_batch < batch_iterations; idx_batch++) {
      std::cerr << "IDX BATCH " << idx_batch << std::endl;
      // Round 1.
      ASSERT_OK_AND_ASSIGN(auto p0_return_round1,
                           gd_party_zero.GenerateGradientDescentRoundOneMessage(
                               p0_state_masked_x[idx_batch]));
      ASSERT_OK_AND_ASSIGN(auto p1_return_round1,
                           gd_party_one.GenerateGradientDescentRoundOneMessage(
                               p1_state_masked_x[idx_batch]));


      // Get Sigmoid Inputs
      ASSERT_OK_AND_ASSIGN(SigmoidInput p0_sigmoid_input,
                           gd_party_zero.GenerateSigmoidInputMinibatch(
                               p0_return_round1.first, p1_message_masked_x[idx_batch],
                               p1_return_round1.second, batch_size, idx_batch, size_per_minibatch));
      ASSERT_OK_AND_ASSIGN(SigmoidInput p1_sigmoid_input,
                           gd_party_one.GenerateSigmoidInputMinibatch(
                               p1_return_round1.first, p0_message_masked_x[idx_batch],
                               p0_return_round1.second, batch_size, idx_batch, size_per_minibatch));

      // Compute Sigmoid
      applications::SigmoidPrecomputedValueNewMic &gd_party_zero_share =
          gd_party_zero.share_provider_->GetSigmoidPrecomputedValueNewMic();
      applications::SigmoidPrecomputedValueNewMic &gd_party_one_share =
          gd_party_one.share_provider_->GetSigmoidPrecomputedValueNewMic();
      //ASSERT_OK_AND_ASSIGN(applications::SigmoidPrecomputedValueNewMic gd_party_zero_share,
      //                     gd_party_zero.share_provider_->GetSigmoidPrecomputedValueNewMic());
      //ASSERT_OK_AND_ASSIGN(applications::SigmoidPrecomputedValueNewMic gd_party_one_share,
      //                     gd_party_one.share_provider_->GetSigmoidPrecomputedValueNewMic());

      // GenerateSigmoidRoundOneMessage

      std::pair<applications::RoundOneSigmoidNewMicState, applications::RoundOneSigmoidNewMicMessage>
          round_one_res_party0;
      std::pair<applications::RoundOneSigmoidNewMicState, applications::RoundOneSigmoidNewMicMessage>
          round_one_res_party1;
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

      std::pair<applications::RoundTwoSigmoidNewMicState, applications::RoundTwoSigmoidNewMicMessage>
          round_two_res_party0,
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

      std::pair<applications::RoundThreeSigmoidNewMicState, applications::RoundThreeSigmoidNewMicMessage>
          round_three_res_party0,
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

      // GenerateSigmoidRoundThreeMessage

      std::pair<applications::RoundThreePointFiveSigmoidNewMicState,
                applications::RoundThreePointFiveSigmoidNewMicMessage> round_three_point_five_res_party0,
          round_three_point_five_res_party1;

      ASSERT_OK_AND_ASSIGN(
          round_three_point_five_res_party0,
          secure_sigmoid->GenerateSigmoidRoundThreePointFiveMessage(0,
                                                                    gd_party_zero_share,
                                                                    round_three_res_party0.first,
                                                                    round_three_res_party1.second));

      ASSERT_OK_AND_ASSIGN(
          round_three_point_five_res_party1,
          secure_sigmoid->GenerateSigmoidRoundThreePointFiveMessage(1,
                                                                    gd_party_one_share,
                                                                    round_three_res_party1.first,
                                                                    round_three_res_party0.second));


      // GenerateSigmoidRoundFourMessage

      std::pair<applications::RoundFourSigmoidNewMicState, applications::RoundFourSigmoidNewMicMessage>
          round_four_res_party0,
          round_four_res_party1;

      ASSERT_OK_AND_ASSIGN(
          round_four_res_party0,
          secure_sigmoid->GenerateSigmoidRoundFourMessage(0,
                                                          gd_party_zero_share,
                                                          round_three_point_five_res_party0.first,
                                                          round_three_point_five_res_party1.second));

      ASSERT_OK_AND_ASSIGN(
          round_four_res_party1,
          secure_sigmoid->GenerateSigmoidRoundFourMessage(1,
                                                          gd_party_one_share,
                                                          round_three_point_five_res_party1.first,
                                                          round_three_point_five_res_party0.second));
      // GenerateSigmoidRoundFiveMessage

      std::pair<applications::RoundFiveSigmoidNewMicState, applications::RoundFiveSigmoidNewMicMessage>
          round_five_res_party0,
          round_five_res_party1;

      ASSERT_OK_AND_ASSIGN(
          round_five_res_party0,
          secure_sigmoid->GenerateSigmoidRoundFiveMessage(0,
                                                          gd_party_zero_share,
                                                          round_four_res_party0.first,
                                                          round_four_res_party1.second));

      ASSERT_OK_AND_ASSIGN(
          round_five_res_party1,
          secure_sigmoid->GenerateSigmoidRoundFiveMessage(1,
                                                          gd_party_one_share,
                                                          round_four_res_party1.first,
                                                          round_four_res_party0.second));
      // GenerateSigmoidResult

      std::vector<uint64_t> final_sigmoid_outputs_share_party0,
          final_sigmoid_outputs_share_party1;

      ASSERT_OK_AND_ASSIGN(
          final_sigmoid_outputs_share_party0,
          secure_sigmoid->GenerateSigmoidResult(0,
                                                gd_party_zero_share,
                                                round_five_res_party0.first,
                                                round_five_res_party1.second));

      ASSERT_OK_AND_ASSIGN(
          final_sigmoid_outputs_share_party1,
          secure_sigmoid->GenerateSigmoidResult(1,
                                                gd_party_one_share,
                                                round_five_res_party1.first,
                                                round_five_res_party0.second));
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

      // Round 3.
      ASSERT_OK_AND_ASSIGN(
          auto p0_return_round3,
          gd_party_zero.GenerateGradientDescentRoundThreeMessageMinibatch(
              p0_sigmoid_output, p0_state_masked_x_transpose[idx_batch],
              batch_size, idx_batch));
      ASSERT_OK_AND_ASSIGN(
          auto p1_return_round3,
          gd_party_one.GenerateGradientDescentRoundThreeMessageMinibatch(
              p1_sigmoid_output, p1_state_masked_x_transpose[idx_batch],
              batch_size, idx_batch));

      // Compute gradient descent update.
      ASSERT_OK(gd_party_zero.ComputeGradientUpdateMinibatch(
          p0_return_round3.first, p1_message_masked_x_transpose[idx_batch],
          p1_return_round3.second, batch_size, idx_batch, size_per_minibatch));
      ASSERT_OK(gd_party_one.ComputeGradientUpdateMinibatch(
          p1_return_round3.first, p0_message_masked_x_transpose[idx_batch],
          p0_return_round3.second, batch_size, idx_batch, size_per_minibatch));


      // DEBUG
      std::vector<uint64_t> updated_share_theta_p0 = gd_party_zero.GetTheta();
      std::vector<uint64_t> updated_share_theta_p1 = gd_party_one.GetTheta();
      ASSERT_OK_AND_ASSIGN(
          auto theta, BatchedModAdd(updated_share_theta_p0, updated_share_theta_p1,
                                    gd_params.modulus));
      std::cerr << "Printing Thetas: " << std::endl;
      for (size_t idx = 0; idx < theta.size(); idx++) {
        std::cerr << fp_factory_->ImportFixedPointElementFromUint64(theta[idx])
            ->ExportToDouble() << ",";
      }
    }
  }

  // Reconstruct theta, and check if the value of the parameters makes sense.
  // For the derivative of the balloons dataset used in this test,
  // we ran an identical plaintext algorithm and hardcoded the theta outputs.
  // For 5 iterations and learning rate alpha = 18, We expect:
  // 5 iters: -5.209011173974926,-2.6407028538582202,
  // -2.64070285385822,-10.915857622419917,5.392025544439257

  std::vector<double> approx_theta{-5.209011173974926,-2.6407028538582202,
                                   -2.64070285385822,-10.915857622419917,5.392025544439257};
  std::vector<uint64_t> updated_share_theta_p0 = gd_party_zero.GetTheta();
  std::vector<uint64_t> updated_share_theta_p1 = gd_party_one.GetTheta();
  ASSERT_OK_AND_ASSIGN(
      auto theta, BatchedModAdd(updated_share_theta_p0, updated_share_theta_p1,
                                gd_params.modulus));

  std::cerr << "Printing Thetas: " << std::endl;
  for (size_t idx = 0; idx < theta.size(); idx++) {
  std::cerr << fp_factory_->ImportFixedPointElementFromUint64(theta[idx])
  ->ExportToDouble() << ",";
  //ASSERT_NEAR(fp_factory_->ImportFixedPointElementFromUint64(theta[idx])
  //                ->ExportToDouble(),
  //            approx_theta[idx], 0.04);
  }
}
*/

}  // namespace
}  // namespace logistic_regression
}  // namespace private_join_and_compute
