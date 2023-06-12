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

#include "applications/secure_sigmoid/secure_sigmoid.h"

#include <cstdint>
#include <cmath>
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

            const size_t kNumFractionalBits = 12;
            const size_t kLogGroupSize = 63;
            const uint64_t kIntervalCount = 10;

            // Fixing the input batch size.
            const uint64_t kNumInputs = 16;

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

            const SecureSplineParameters sigmoid_spline_params{
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



            const ExponentiationParams kSampleLargeExpParams = {
                    13,
                    kLargePrime};

            const SecureSigmoidParameters sigmoid_params{
                    kLogGroupSize,
                    sigmoid_spline_params,
                    kNumFractionalBits,
                    kTaylorPolynomialDegree,
                    kSampleLargeExpParams
            };

            class SecureSigmoidTest : public Test {
            protected:
                StatusOr<std::unique_ptr<BasicRng>> MakePrng() {
                    auto random_seed = BasicRng::GenerateSeed();
                    if (!random_seed.ok()) {
                        return InternalError("Random seed generation fails.");
                    }
                    return BasicRng::Create(random_seed.value());
                }
            };

    TEST_F(SecureSigmoidTest, BenchmarkingIntegrationTest) {


      ASSERT_OK_AND_ASSIGN(std::unique_ptr<SecureSigmoid> secure_sigmoid,
                           SecureSigmoid::Create(kNumInputs, sigmoid_params));

      std::pair<SigmoidPrecomputedValue, SigmoidPrecomputedValue> preCompRes;

      ASSERT_OK_AND_ASSIGN(preCompRes,
                           secure_sigmoid->PerformSigmoidPrecomputation());


      // Validating exp precomputation
      for(int i = 0; i < kNumInputs; i++){
        absl::uint128 alpha_zero = preCompRes.first.mta_pos.first[i];
        absl::uint128 beta_zero = preCompRes.first.mta_pos.second[i];
        absl::uint128 alpha_one = preCompRes.second.mta_pos.first[i];
        absl::uint128 beta_one = preCompRes.second.mta_pos.second[i];
      }

      //std::vector<double> sigmoid_inputs{0.15, 0.23, 0.77, 0.49, 0.94, -14.8, -0.15, 50, 1.4, -100,
      //                                   1.1, 5.1, -3.5, -10, 12.3, -11.1};
      double start_val = -20.;
      std::vector<double> sigmoid_inputs(kNumInputs);
      for (size_t idx = 0; idx < kNumInputs; idx++) {
        sigmoid_inputs[idx] = start_val;
        start_val += 0.1;
      }

      std::vector<uint64_t> sigmoid_inputs_ring;

      // Converting inputs into Ring representation (in order to secret share
      // them later).
      for (int i = 0; i<kNumInputs; i++) {
        ASSERT_OK_AND_ASSIGN(
            FixedPointElement sigmoid_input_fxp,
            secure_sigmoid->fixed_point_factory_->CreateFixedPointElementFromDouble(
                sigmoid_inputs[i]));
        sigmoid_inputs_ring.push_back(sigmoid_input_fxp.ExportToUint64());
      }

      uint64_t modulus =
          secure_sigmoid->fixed_point_factory_->GetParams().primary_ring_modulus;

      // Secret Sharing Fixed Point representation of sigmoid inputs.
      // Generate random shares for vector x and y and distribute to P0 and P1.
      ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, MakePrng());

      std::vector<uint64_t> share_of_sigmoid_inputs_party0,
          share_of_sigmoid_inputs_party1;

      ASSERT_OK_AND_ASSIGN(share_of_sigmoid_inputs_party0,
                           SampleVectorFromPrng(kNumInputs, modulus, prng.get()));
      ASSERT_OK_AND_ASSIGN(
          share_of_sigmoid_inputs_party1,
          private_join_and_compute::BatchedModSub(sigmoid_inputs_ring,
                                                  share_of_sigmoid_inputs_party0,
                                                  modulus));

      std::pair<RoundOneSigmoidState, RoundOneSigmoidMessage> round_one_res_party0;
      std::pair<RoundOneSigmoidState, RoundOneSigmoidMessage> round_one_res_party1;

      ASSERT_OK_AND_ASSIGN(
          round_one_res_party0,
          secure_sigmoid->GenerateSigmoidRoundOneMessage(0,
                                                         preCompRes.first,
                                                         share_of_sigmoid_inputs_party0));
      ASSERT_OK_AND_ASSIGN(
          round_one_res_party1,
          secure_sigmoid->GenerateSigmoidRoundOneMessage(1,
                                                         preCompRes.second,
                                                         share_of_sigmoid_inputs_party1));

      std::pair<RoundTwoSigmoidState, RoundTwoSigmoidMessage> round_two_res_party0,
          round_two_res_party1;

      ASSERT_OK_AND_ASSIGN(
          round_two_res_party0,
          secure_sigmoid->GenerateSigmoidRoundTwoMessage(0,
                                                         preCompRes.first,
                                                         round_one_res_party0.first,
                                                         round_one_res_party1.second));

      ASSERT_OK_AND_ASSIGN(
          round_two_res_party1,
          secure_sigmoid->GenerateSigmoidRoundTwoMessage(1,
                                                         preCompRes.second,
                                                         round_one_res_party1.first,
                                                         round_one_res_party0.second));

      // Validating exponentiation results
      for(int i = 0; i < kNumInputs; i++) {
        ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                             secure_sigmoid->fixed_point_factory_->ImportFixedPointElementFromUint64(
                                 (round_two_res_party0.first.exp_result_state_pos[i].ExportToUint64() +
                                     round_two_res_party1.first.exp_result_state_pos[i].ExportToUint64()) % modulus));

      }

      // Validating conditional results from MIC gate eval

      std::vector<std::vector<uint64_t>> mic_gate_result;

      // Reconstructing the output of spline using the shares of both parties.
      for(int i = 0; i < kNumInputs; i++){

        ASSERT_OK_AND_ASSIGN(
            auto mic_gate_result_i, private_join_and_compute::BatchedModAdd(
            round_two_res_party0.first.mic_gate_result_share[i],
            round_two_res_party1.first.mic_gate_result_share[i],
            modulus));

        mic_gate_result.
            push_back(mic_gate_result_i);

      }

      std::pair<RoundThreeSigmoidState, RoundThreeSigmoidMessage> round_three_res_party0,
          round_three_res_party1;

      ASSERT_OK_AND_ASSIGN(
          round_three_res_party0,
          secure_sigmoid->GenerateSigmoidRoundThreeMessage(0,
                                                           preCompRes.first,
                                                           round_two_res_party0.first,
                                                           round_two_res_party1.second));

      ASSERT_OK_AND_ASSIGN(
          round_three_res_party1,
          secure_sigmoid->GenerateSigmoidRoundThreeMessage(1,
                                                           preCompRes.second,
                                                           round_two_res_party1.first,
                                                           round_two_res_party0.second));


      // Validating spline path

      std::vector<uint64_t> res;

      // Reconstructing the output of spline using the shares of both parties.
      ASSERT_OK_AND_ASSIGN(
          res, private_join_and_compute::BatchedModAdd(
          round_three_res_party0.first.spline_res_in_secret_shared_form_pos,
          round_three_res_party1.first.spline_res_in_secret_shared_form_pos,
          modulus));

      // Checking whether reconstrcuted output is close to the actual sigmoid value.
      for (int i = 0; i < kNumInputs; i++) {
        ASSERT_OK_AND_ASSIGN(
            FixedPointElement res_i,
            secure_sigmoid->fixed_point_factory_->ImportFixedPointElementFromUint64(
                res[i]));

        double actual_sigmoid_output = 1.0 / (1 + exp(-1.0 * sigmoid_inputs[i]));
        EXPECT_NEAR(res_i.ExportToDouble(), actual_sigmoid_output, 0.01);
      }

      std::pair<RoundFourSigmoidState, RoundFourSigmoidMessage> round_four_res_party0,
          round_four_res_party1;

      ASSERT_OK_AND_ASSIGN(
          round_four_res_party0,
          secure_sigmoid->GenerateSigmoidRoundFourMessage(0,
                                                          preCompRes.first,
                                                          round_three_res_party0.first,
                                                          round_three_res_party1.second));

      ASSERT_OK_AND_ASSIGN(
          round_four_res_party1,
          secure_sigmoid->GenerateSigmoidRoundFourMessage(1,
                                                          preCompRes.second,
                                                          round_three_res_party1.first,
                                                          round_three_res_party0.second));

      std::vector<uint64_t> final_sigmoid_outputs_share_party0,
          final_sigmoid_outputs_share_party1;


      ASSERT_OK_AND_ASSIGN(
          final_sigmoid_outputs_share_party0,
          secure_sigmoid->GenerateSigmoidResult(0,
                                                preCompRes.first,
                                                round_four_res_party0.first,
                                                round_four_res_party1.second));

      ASSERT_OK_AND_ASSIGN(
          final_sigmoid_outputs_share_party1,
          secure_sigmoid->GenerateSigmoidResult(1,
                                                preCompRes.second,
                                                round_four_res_party1.first,
                                                round_four_res_party0.second));
      // Validating overall path

      std::vector<uint64_t> sigmoid_res_after_merging_branches;

      // Reconstructing the output of sigmoid using the shares of both parties.
      ASSERT_OK_AND_ASSIGN(
          sigmoid_res_after_merging_branches,
          private_join_and_compute::BatchedModAdd(
              final_sigmoid_outputs_share_party0,
              final_sigmoid_outputs_share_party1,
              modulus));

      // Checking whether reconstrcuted output is close to the actual sigmoid value.
      for (int i = 0; i < kNumInputs; i++) {
        ASSERT_OK_AND_ASSIGN(
            FixedPointElement res_i,
            secure_sigmoid->fixed_point_factory_->ImportFixedPointElementFromUint64(
                sigmoid_res_after_merging_branches[i]));

        double actual_sigmoid_output = 1.0 / (1 + exp(-1.0 * sigmoid_inputs[i]));

        std::cout << res_i.ExportToDouble() << std::endl;

        EXPECT_NEAR(res_i.ExportToDouble(), actual_sigmoid_output, 0.01);


        uint64_t minus_point_15_party0 = (-share_of_sigmoid_inputs_party0[5]) % modulus;
        uint64_t minus_point_15_party1 = (-share_of_sigmoid_inputs_party1[5]) % modulus;
        ASSERT_OK_AND_ASSIGN(FixedPointElement minus_point_15,
                               secure_sigmoid->fixed_point_factory_->ImportFixedPointElementFromUint64(
                               (minus_point_15_party0 + minus_point_15_party1) % modulus
        ));
      }

    }


} // namespace
} // namespace applications
} // namespace private_join_and_compute