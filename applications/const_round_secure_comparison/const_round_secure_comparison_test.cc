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

#include "applications/const_round_secure_comparison/const_round_secure_comparison.h"
#include <chrono>
#include <cstdint>
#include <vector>

#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status_testing.h"
#include "private_join_and_compute/util/status_testing.inc"
#include "applications/secure_comparison/secure_comparison.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace private_join_and_compute {
    namespace applications {
        namespace {

            using ::testing::Test;

            const uint64_t kStringLength = 62;
            const uint64_t kNumPieces = 4;
            const uint64_t kPieceLength = 16;

            // Fixing the input batch size.
            const uint64_t kNumInputs = 1000; //3;
            const uint64_t modulus = 1ULL << kStringLength;


            const SecureComparisonParameters secure_comparison_params{
                    kStringLength,
                    kNumPieces,
                    kPieceLength
            };


            class SecureComparisonTest : public Test {
            protected:
                StatusOr<std::unique_ptr<BasicRng>> MakePrng() {
                    auto random_seed = BasicRng::GenerateSeed();
                    if (!random_seed.ok()) {
                        return InternalError("Random seed generation fails.");
                    }
                    return BasicRng::Create(random_seed.value());
                }
            };

        TEST_F(SecureComparisonTest, EndToEndSecretSharedInputSucceeds){

            ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstRoundSecureComparison> seccomp,
                                 ConstRoundSecureComparison::Create(kNumInputs,
                                                                    secure_comparison_params));

            std::pair<SecureComparisonPrecomputedValue, SecureComparisonPrecomputedValue> preCompRes;

            ASSERT_OK_AND_ASSIGN(preCompRes,
                                 seccomp->PerformComparisonPrecomputation());

            // Comparing x > y
            std::vector<uint64_t> comparison_inputs_x (kNumInputs, 0); // = {0, 1, 2};

            std::vector<uint64_t> comparison_inputs_y (kNumInputs, 0);//= {3, 1, 1};

            // Generate random shares for vector x and y and distribute to P0 and P1.
            ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, MakePrng());

            std::vector<uint64_t> share_of_comparison_inputs_x_party0,
                    share_of_comparison_inputs_x_party1;

            ASSERT_OK_AND_ASSIGN(share_of_comparison_inputs_x_party0,
                                SampleVectorFromPrng(kNumInputs, modulus, prng.get()));

            ASSERT_OK_AND_ASSIGN(share_of_comparison_inputs_x_party1,
                    private_join_and_compute::BatchedModSub(comparison_inputs_x,
                                                            share_of_comparison_inputs_x_party0,
                                                            modulus));

            std::vector<uint64_t> share_of_comparison_inputs_y_party0,
                    share_of_comparison_inputs_y_party1;

            ASSERT_OK_AND_ASSIGN(share_of_comparison_inputs_y_party0,
                                SampleVectorFromPrng(kNumInputs, modulus, prng.get()));

            ASSERT_OK_AND_ASSIGN(share_of_comparison_inputs_y_party1,
                    private_join_and_compute::BatchedModSub(comparison_inputs_y,
                                                            share_of_comparison_inputs_y_party0,
                                                            modulus));


            // Reduce secret-shared comparison to non-secret shared

            ASSERT_OK_AND_ASSIGN(auto firstbit_comparisoninput_p0,
                                private_join_and_compute::secure_comparison::SecretSharedComparisonPrepareInputsPartyZero(
                                    share_of_comparison_inputs_x_party0,
                                    share_of_comparison_inputs_y_party0,
                                    kNumInputs,
                                    modulus,
                                    kStringLength));

            auto first_bit_share_p0 = firstbit_comparisoninput_p0.first;
            auto comparison_input_p0 = firstbit_comparisoninput_p0.second;

            ASSERT_OK_AND_ASSIGN(auto firstbit_comparisoninput_p1,
                                private_join_and_compute::secure_comparison::SecretSharedComparisonPrepareInputsPartyOne(
                                    share_of_comparison_inputs_x_party1,
                                    share_of_comparison_inputs_y_party1,
                                    kNumInputs,
                                    modulus,
                                    kStringLength));

            auto first_bit_share_p1 = firstbit_comparisoninput_p1.first;
            auto comparison_input_p1 = firstbit_comparisoninput_p1.second;

	        auto start = std::chrono::high_resolution_clock::now();


            std::pair<RoundOneSecureComparisonState, RoundOneSecureComparisonMessage> round_one_res_party0;
            std::pair<RoundOneSecureComparisonState, RoundOneSecureComparisonMessage> round_one_res_party1;

            ASSERT_OK_AND_ASSIGN(round_one_res_party0,
                    seccomp->GenerateComparisonRoundOneMessage(0,
                                                                   preCompRes.first,
                                                                   comparison_input_p0));

            ASSERT_OK_AND_ASSIGN(round_one_res_party1,
                                 seccomp->GenerateComparisonRoundOneMessage(1,
                                                                            preCompRes.second,
                                                                            comparison_input_p1));

            std::pair<RoundTwoSecureComparisonState, RoundTwoSecureComparisonMessage> round_two_res_party0;
            std::pair<RoundTwoSecureComparisonState, RoundTwoSecureComparisonMessage> round_two_res_party1;

            ASSERT_OK_AND_ASSIGN(round_two_res_party0,
                    seccomp->GenerateComparisonRoundTwoMessage(0,
                                                                   preCompRes.first,
                                                                   round_one_res_party0.first,
                                                                   round_one_res_party1.second));

            ASSERT_OK_AND_ASSIGN(round_two_res_party1,
                                 seccomp->GenerateComparisonRoundTwoMessage(1,
                                                                            preCompRes.second,
                                                                            round_one_res_party1.first,
                                                                            round_one_res_party0.second));

            std::pair<RoundThreeSecureComparisonState, RoundThreeSecureComparisonMessage> round_three_res_party0;
            std::pair<RoundThreeSecureComparisonState, RoundThreeSecureComparisonMessage> round_three_res_party1;

            ASSERT_OK_AND_ASSIGN(round_three_res_party0,
                    seccomp->GenerateComparisonRoundThreeMessage(0,
                                                                   preCompRes.first,
                                                                   round_two_res_party0.first,
                                                                   round_two_res_party1.second));

            ASSERT_OK_AND_ASSIGN(round_three_res_party1,
                                  seccomp->GenerateComparisonRoundThreeMessage(1,
                                                                             preCompRes.second,
                                                                             round_two_res_party1.first,
                                                                             round_two_res_party0.second));
            std::vector<uint64_t> seccomp_res_party0;
            std::vector<uint64_t> seccomp_res_party1;

            ASSERT_OK_AND_ASSIGN(seccomp_res_party0,
                    seccomp->GenerateComparisonResult(0,
                                                     preCompRes.first,
                                                     round_three_res_party0.first,
                                                     round_three_res_party1.second));


            ASSERT_OK_AND_ASSIGN(seccomp_res_party1,
                    seccomp->GenerateComparisonResult(1,
                                                     preCompRes.second,
                                                     round_three_res_party1.first,
                                                     round_three_res_party0.second));


            // For secret-shared comparison, there is extra step of adding shares of first bit
            ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> comparison_output_share_p0,
                                private_join_and_compute::secure_comparison::SecretSharedComparisonFinishReduction(
                                    first_bit_share_p0,
                                    seccomp_res_party0,
                                    kNumInputs));


            ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> comparison_output_share_p1,
                                 private_join_and_compute::secure_comparison::SecretSharedComparisonFinishReduction(
                                    first_bit_share_p1,
                                    seccomp_res_party1,
                                    kNumInputs));
            std::vector<uint64_t> seccomp_res;

            ASSERT_OK_AND_ASSIGN(seccomp_res, BatchedModAdd(
                comparison_output_share_p0,
                comparison_output_share_p1,
                2
            ));

            for(int i = 0; i < kNumInputs; i++){
                uint64_t actual_res = (comparison_inputs_x[i] > comparison_inputs_y[i]);
                EXPECT_EQ(actual_res, seccomp_res[i]);
            }
        }

} // namespace
} // namespace applications
} // namespace private_join_and_compute
