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

#include "applications/secure_spline/secure_spline.h"

#include <cstdint>
#include <vector>

#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status_testing.h"
#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace private_join_and_compute {
namespace applications {
namespace {

using ::testing::Test;

const size_t kNumFractionalBits = 20;
const size_t kLogGroupSize = 63;
const uint64_t kIntervalCount = 10;

// Fixing the input batch size to be 5.
const uint64_t kNumInputs = 5;

class SecureSplineTest : public Test {
 protected:
  StatusOr<std::unique_ptr<BasicRng>> MakePrng() {
    auto random_seed = BasicRng::GenerateSeed();
    if (!random_seed.ok()) {
      return InternalError("Random seed generation fails.");
    }
    return BasicRng::Create(random_seed.value());
  }
};

TEST_F(SecureSplineTest, CreateSucceeds) {
  const uint64_t interval_count = 2;

  const std::vector<double> spline_lower_bounds{0.0, 0.1};

  const std::vector<double> spline_upper_bounds = {0.1, 0.2};

  const std::vector<double> spline_slope = {0.24979187478940013,
                                            0.24854809833537939};

  const std::vector<double> spline_yIntercept = {0.5, 0.5001243776454021};

  const SecureSplineParameters spline_params{
      kLogGroupSize,       interval_count,      kNumFractionalBits,
      spline_lower_bounds, spline_upper_bounds, spline_slope,
      spline_yIntercept};

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<SecureSpline> secure_spline,
                       SecureSpline::Create(kNumInputs, spline_params));
}

TEST_F(SecureSplineTest, CreateFailsforNonContiguousIntervals) {
  const uint64_t interval_count = 2;

  const std::vector<double> spline_lower_bounds{0.0, 0.2};

  const std::vector<double> spline_upper_bounds = {0.1, 0.3};

  const std::vector<double> spline_slope = {0.24979187478940013,
                                            0.24854809833537939};

  const std::vector<double> spline_yIntercept = {0.5, 0.5001243776454021};

  const SecureSplineParameters spline_params{
      kLogGroupSize,       interval_count,      kNumFractionalBits,
      spline_lower_bounds, spline_upper_bounds, spline_slope,
      spline_yIntercept};

  EXPECT_THAT(
      SecureSpline::Create(kNumInputs, spline_params),
      testing::StatusIs(StatusCode::kInvalidArgument,
                        ::testing::HasSubstr(
                            "Secure Spline intervals should be contiguous")));
}

TEST_F(SecureSplineTest, CreateFailsforIncorrectLowerBoundVectorSize) {
  const uint64_t interval_count = 2;

  const std::vector<double> spline_lower_bounds{0.0, 0.1, 0.2};

  const std::vector<double> spline_upper_bounds = {0.1, 0.2};

  const std::vector<double> spline_slope = {0.24979187478940013,
                                            0.24854809833537939};

  const std::vector<double> spline_yIntercept = {0.5, 0.5001243776454021};

  const SecureSplineParameters spline_params{
      kLogGroupSize,       interval_count,      kNumFractionalBits,
      spline_lower_bounds, spline_upper_bounds, spline_slope,
      spline_yIntercept};

  EXPECT_THAT(
      SecureSpline::Create(kNumInputs, spline_params),
      testing::StatusIs(
          StatusCode::kInvalidArgument,
          ::testing::HasSubstr(
              "Size of lower bounds vector should match interval_count")));
}

TEST_F(SecureSplineTest, CreateFailsforIncorrectUpperBoundVectorSize) {
  const uint64_t interval_count = 2;

  const std::vector<double> spline_lower_bounds{0.0, 0.1};

  const std::vector<double> spline_upper_bounds = {0.1, 0.2, 0.3};

  const std::vector<double> spline_slope = {0.24979187478940013,
                                            0.24854809833537939};

  const std::vector<double> spline_yIntercept = {0.5, 0.5001243776454021};

  const SecureSplineParameters spline_params{
      kLogGroupSize,       interval_count,      kNumFractionalBits,
      spline_lower_bounds, spline_upper_bounds, spline_slope,
      spline_yIntercept};

  EXPECT_THAT(
      SecureSpline::Create(kNumInputs, spline_params),
      testing::StatusIs(
          StatusCode::kInvalidArgument,
          ::testing::HasSubstr(
              "Size of upper bounds vector should match interval_count")));
}

TEST_F(SecureSplineTest, CreateFailsforIncorrectSlopeVectorSize) {
  const uint64_t interval_count = 2;

  const std::vector<double> spline_lower_bounds{0.0, 0.1};

  const std::vector<double> spline_upper_bounds = {0.1, 0.2};

  const std::vector<double> spline_slope = {0.24979187478940013,
                                            0.24854809833537939, 0.8};

  const std::vector<double> spline_yIntercept = {0.5, 0.5001243776454021};

  const SecureSplineParameters spline_params{
      kLogGroupSize,       interval_count,      kNumFractionalBits,
      spline_lower_bounds, spline_upper_bounds, spline_slope,
      spline_yIntercept};

  EXPECT_THAT(SecureSpline::Create(kNumInputs, spline_params),
              testing::StatusIs(
                  StatusCode::kInvalidArgument,
                  ::testing::HasSubstr(
                      "Size of slope vector should match interval_count")));
}

TEST_F(SecureSplineTest, CreateFailsforIncorrectYInterceptVectorSize) {
  const uint64_t interval_count = 2;

  const std::vector<double> spline_lower_bounds{0.0, 0.1};

  const std::vector<double> spline_upper_bounds = {0.1, 0.2};

  const std::vector<double> spline_slope = {0.24979187478940013,
                                            0.24854809833537939};

  const std::vector<double> spline_yIntercept = {0.5, 0.5001243776454021, 0.6};

  const SecureSplineParameters spline_params{
      kLogGroupSize,       interval_count,      kNumFractionalBits,
      spline_lower_bounds, spline_upper_bounds, spline_slope,
      spline_yIntercept};

  EXPECT_THAT(
      SecureSpline::Create(kNumInputs, spline_params),
      testing::StatusIs(
          StatusCode::kInvalidArgument,
          ::testing::HasSubstr(
              "Size of yIntercept vector should match interval_count")));
}

TEST_F(SecureSplineTest, EndToEndSucceeds) {
  // Harcoding spline parameters for sigmoid function for input values in the
  // range x \in [0, 1) by splitting it into 10 equal-sized intervals
  // and defining a line (1-degree polynomial) on each of those intervals.
  const std::vector<double> sigmoid_spline_lower_bounds{
      0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

  const std::vector<double> sigmoid_spline_upper_bounds = {
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  const std::vector<double> sigmoid_spline_slope = {
      0.24979187478940013, 0.24854809833537939, 0.24608519499181072,
      0.24245143300792976, 0.23771671089402596, 0.23196975023940808,
      0.2253146594237077,  0.2178670895944635,  0.20975021497391394,
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

  const SecureSplineParameters sigmoid_spline_params{
      kLogGroupSize,
      kIntervalCount,
      kNumFractionalBits,
      sigmoid_spline_lower_bounds,
      sigmoid_spline_upper_bounds,
      sigmoid_spline_slope,
      sigmoid_spline_yIntercept};

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<SecureSpline> secure_spline,
                       SecureSpline::Create(kNumInputs, sigmoid_spline_params));

  std::pair<std::vector<SplinePrecomputedValue>,
            std::vector<SplinePrecomputedValue>>
      spline_precomputed_value;
  ASSERT_OK_AND_ASSIGN(spline_precomputed_value,
                       secure_spline->PerformSplinePrecomputation());

  std::vector<SplinePrecomputedValue> spline_precomputed_values_party0 =
      spline_precomputed_value.first;

  std::vector<SplinePrecomputedValue> spline_precomputed_values_party1 =
      spline_precomputed_value.second;

  std::vector<double> spline_inputs{0.15, 0.23, 0.77, 0.49, 0.94};

  std::vector<uint64_t> spline_inputs_ring;

  // Converting inputs into Ring representation (in order to secret share
  // them later).
  for (int i = 0; i < kNumInputs; i++) {
    ASSERT_OK_AND_ASSIGN(
        FixedPointElement spline_input_fxp,
        secure_spline->fixed_point_factory_->CreateFixedPointElementFromDouble(
            spline_inputs[i]));
    spline_inputs_ring.push_back(spline_input_fxp.ExportToUint64());
  }

  uint64_t modulus =
      secure_spline->fixed_point_factory_->GetParams().primary_ring_modulus;

  // Secret Sharing Fixed Point representation of spline inputs.
  // Generate random shares for vector x and y and distribute to P0 and P1.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, MakePrng());

  std::vector<uint64_t> share_of_spline_inputs_party0,
      share_of_spline_inputs_party1;

  ASSERT_OK_AND_ASSIGN(share_of_spline_inputs_party0,
                       SampleVectorFromPrng(kNumInputs, modulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(
      share_of_spline_inputs_party1,
      private_join_and_compute::BatchedModSub(spline_inputs_ring, share_of_spline_inputs_party0,
                              modulus));

  std::pair<RoundOneSplineState, RoundOneSplineMessage> round_one_res_party0,
      round_one_res_party1;

  ASSERT_OK_AND_ASSIGN(
      round_one_res_party0,
      secure_spline->GenerateSplineRoundOneMessage(
          spline_precomputed_values_party0, share_of_spline_inputs_party0));

  ASSERT_OK_AND_ASSIGN(
      round_one_res_party1,
      secure_spline->GenerateSplineRoundOneMessage(
          spline_precomputed_values_party1, share_of_spline_inputs_party1));

  std::pair<RoundTwoSplineState, RoundTwoSplineMessage> round_two_res_party0,
      round_two_res_party1;

  ASSERT_OK_AND_ASSIGN(
      round_two_res_party0,
      secure_spline->GenerateSplineRoundTwoMessage(
          0, spline_precomputed_values_party0, round_one_res_party0.first,
          round_one_res_party1.second));

  ASSERT_OK_AND_ASSIGN(
      round_two_res_party1,
      secure_spline->GenerateSplineRoundTwoMessage(
          1, spline_precomputed_values_party1, round_one_res_party1.first,
          round_one_res_party0.second));

  std::vector<uint64_t> res_party0, res_party1;

  ASSERT_OK_AND_ASSIGN(
      res_party0, secure_spline->GenerateSplineResult(
                      0, spline_precomputed_values_party0,
                      round_two_res_party0.first, round_two_res_party1.second));

  ASSERT_OK_AND_ASSIGN(
      res_party1, secure_spline->GenerateSplineResult(
                      1, spline_precomputed_values_party1,
                      round_two_res_party1.first, round_two_res_party0.second));

  std::vector<uint64_t> res;

  // Reconstructing the output of spline using the shares of both parties.
  ASSERT_OK_AND_ASSIGN(
      res, private_join_and_compute::BatchedModAdd(res_party0, res_party1, modulus));

  // Checking whether reconstructed output is close to the actual sigmoid value.
  for (int i = 0; i < kNumInputs; i++) {
    ASSERT_OK_AND_ASSIGN(
        FixedPointElement res_i,
        secure_spline->fixed_point_factory_->ImportFixedPointElementFromUint64(
            res[i]));
    double actual_sigmoid_output = 1.0 / (1 + exp(-1.0 * spline_inputs[i]));
    EXPECT_NEAR(res_i.ExportToDouble(), actual_sigmoid_output, 0.01);
  }
}

}  // namespace
}  // namespace applications
}  // namespace private_join_and_compute
