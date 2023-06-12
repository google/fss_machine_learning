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

#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic.pb.h"
#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/numeric/int128.h"
#include "secret_sharing_mpc/gates/hadamard_product.h"
#include "applications/secure_spline/secure_spline.h"
#include "secret_sharing_mpc/gates/polynomial.h"
#include "absl/memory/memory.h"
#include "secret_sharing_mpc/gates/vector_exponentiation.h"
//#include "dpf/status_macros.h"

namespace private_join_and_compute {
    namespace applications {

        using ::private_join_and_compute::FixedPointElement;
        using ::private_join_and_compute::FixedPointElementFactory;

        absl::StatusOr<std::unique_ptr<SecureSigmoidNewMic>> SecureSigmoidNewMic::Create(
                uint64_t num_inputs, SecureSigmoidNewMicParameters sigmoid_params) {

            // Setting up fxp factory


            ASSIGN_OR_RETURN(
                    FixedPointElementFactory fixed_point_factory,
                    FixedPointElementFactory::Create(sigmoid_params.spline_params.num_fractional_bits,
                                                     sigmoid_params.spline_params.log_group_size));



            // Hardcoding the different interval values
            const uint64_t two_raised_to_lf =
                    fixed_point_factory.GetParams().fractional_multiplier;
            const uint64_t two_raised_to_l =
                    fixed_point_factory.GetParams().primary_ring_modulus;
            const uint64_t lf = fixed_point_factory.GetParams().num_fractional_bits;




            // Check for possible rounding errors in setting the intervals

            // Natural log of 2
            const double ln2 = 0.69314718055994530941;

            const std::vector<double> bounds = {
                    0.0, 1.0, lf * ln2, two_raised_to_l / (2.0 * 2.0 * two_raised_to_lf)
            };


            std::vector<FixedPointElement> bounds_fxp;


            for (int i = 0; i < 4; ++i) {

                ASSIGN_OR_RETURN(FixedPointElement b,
                                 fixed_point_factory.CreateFixedPointElementFromDouble(
                                         bounds[i]));                                         

                bounds_fxp.push_back(b);
            }

            std::vector<uint64_t> intervals_uint64;

            // Adding positive intervals
            for (int i = 0; i < 4; i++){
                intervals_uint64.push_back(bounds_fxp[i].ExportToUint64());
            }

            // Adding negative intervals - Need to switch the ordering of upper and lower bound
            for (int i = 3; i > 0; i--){
                intervals_uint64.push_back(two_raised_to_l - bounds_fxp[i].ExportToUint64());

            }

            // Creating a Secure Comparison gate

            SecureComparisonParameters secure_comparison_params{
                    sigmoid_params.log_group_size,
                    sigmoid_params.num_splits,
                    sigmoid_params.block_length
            };

            ASSIGN_OR_RETURN(std::unique_ptr<ConstRoundSecureComparison> seccomp,
                             ConstRoundSecureComparison::Create(num_inputs,
                                                                secure_comparison_params));

            // Creating a Spline gate
            ASSIGN_OR_RETURN(std::unique_ptr<SecureSpline> secure_spline,
                             SecureSpline::Create(num_inputs, sigmoid_params.spline_params));


            // Create the protocol parties for the exponentiation protocol.

            // Check the first parameter in kSampleLargeExpParams

            // Ideally, only party should be instantiated. However, this is not
            // possible with the way exponentiation protocol is currently implemented.

            ASSIGN_OR_RETURN(auto temp_zero,
                             SecureExponentiationPartyZero::Create(
                                     fixed_point_factory.GetParams(),
                                     sigmoid_params.exp_params));


            ASSIGN_OR_RETURN(auto temp_one,
                             SecureExponentiationPartyOne::Create(
                                     fixed_point_factory.GetParams(),
                                     sigmoid_params.exp_params));

//            std::cout << "Prime : " << sigmoid_params.exp_params.prime_q << std::endl;

//            std::cout << "Create() Party 0 exp prime " << temp_zero->exp_params_->prime_q << std::endl;


            return absl::WrapUnique(new SecureSigmoidNewMic(
                    num_inputs,
//                    num_secure_comparison_rounds_,
                    std::move(seccomp),
                    intervals_uint64,
                    std::move(secure_spline),
                    absl::make_unique<FixedPointElementFactory>(fixed_point_factory),
                    sigmoid_params,
                    std::move(temp_zero),
                    std::move(temp_one)));
        }


        SecureSigmoidNewMic::SecureSigmoidNewMic(
                uint64_t num_inputs,
//                uint64_t num_secure_comparison_rounds,
                std::unique_ptr<ConstRoundSecureComparison> seccomp,
                std::vector<uint64_t> intervals_uint64,
                std::unique_ptr<SecureSpline> secure_spline,
                std::unique_ptr<FixedPointElementFactory> fixed_point_factory,
                SecureSigmoidNewMicParameters sigmoid_params,
                std::unique_ptr<SecureExponentiationPartyZero> exp_party_zero,
                std::unique_ptr<SecureExponentiationPartyOne> exp_party_one)
                : secure_spline_(std::move(secure_spline)),
                seccomp_(std::move(seccomp)),
                num_inputs_(num_inputs),
//				num_secure_comparison_rounds_(num_secure_comparison_rounds),
				intervals_uint64(intervals_uint64),
                sigmoid_params_(sigmoid_params),
                exp_party_zero_(std::move(exp_party_zero)),
                exp_party_one_(std::move(exp_party_one)),
				fixed_point_factory_(std::move(fixed_point_factory)) {}



        absl::StatusOr<std::pair<SigmoidPrecomputedValueNewMic, SigmoidPrecomputedValueNewMic>>
        SecureSigmoidNewMic::PerformSigmoidPrecomputation() {

            SigmoidPrecomputedValueNewMic sigmoid_precomputation_party_0;
            SigmoidPrecomputedValueNewMic sigmoid_precomputation_party_1;


            // Sampling seed and PRNG
            const absl::string_view kSampleSeed = absl::string_view();
            ASSIGN_OR_RETURN(auto rng, private_join_and_compute::BasicRng::Create(kSampleSeed));


            // The modulus is needed to ensure that all the arithemtic operations happen
            // over the primary ring which might be of size < 2^64
            uint64_t modulus = fixed_point_factory_->GetParams().primary_ring_modulus;


            // Generate preprocessed shares of ones for Algorithm 3

            double one = 1.0;

            std::vector<uint64_t> share_of_one_party0, share_of_one_party1;

            ASSIGN_OR_RETURN(FixedPointElement one_fxp,
                             fixed_point_factory_->CreateFixedPointElementFromDouble(one));

            ASSIGN_OR_RETURN(share_of_one_party0,
                             SampleVectorFromPrng(num_inputs_, modulus, rng.get()));


            ASSIGN_OR_RETURN(share_of_one_party1,
                             BatchedModSub(
                                     std::vector<uint64_t>(num_inputs_, one_fxp.ExportToUint64()),
                                     share_of_one_party0,
                                     modulus));

            sigmoid_precomputation_party_0.share_of_one =
                    share_of_one_party0;

            sigmoid_precomputation_party_1.share_of_one =
                    share_of_one_party1;


            // Generate preprocessed shares of zeros for Algorithm 4

            double zero = 0.0;

            ASSIGN_OR_RETURN(FixedPointElement zero_fxp,
                             fixed_point_factory_->CreateFixedPointElementFromDouble(zero));

            std::vector<uint64_t> share_of_zero_party0, share_of_zero_party1;


            ASSIGN_OR_RETURN(share_of_zero_party0,
                             SampleVectorFromPrng(num_inputs_, modulus, rng.get()));

            ASSIGN_OR_RETURN(share_of_zero_party1,
                             BatchedModSub(
                                     std::vector<uint64_t>(num_inputs_, zero_fxp.ExportToUint64()),
                                     share_of_zero_party0,
                                     modulus));

            sigmoid_precomputation_party_0.share_of_zero =
                    share_of_zero_party0;

            sigmoid_precomputation_party_1.share_of_zero =
                    share_of_zero_party1;

            // Generating preprocessing for branching


            for (int i = 0; i < 7; i++) {

                std::pair<SecureComparisonPrecomputedValue, SecureComparisonPrecomputedValue> preCompRes;

                ASSIGN_OR_RETURN(preCompRes, seccomp_->PerformComparisonPrecomputation());

                sigmoid_precomputation_party_0.branching_precomp.comparison_preprocess.push_back(preCompRes.first);

                sigmoid_precomputation_party_1.branching_precomp.comparison_preprocess.push_back(preCompRes.second);


                // Generate ROT correlations for branching

                std::pair<private_join_and_compute::PolynomialRandomOTPrecomputation,
                        private_join_and_compute::PolynomialRandomOTPrecomputation>
                        rot_precomputed_value;

                ASSIGN_OR_RETURN(rot_precomputed_value,
                                 private_join_and_compute::internal::PolynomialPreprocessRandomOTs(
                                         num_inputs_, fixed_point_factory_));

                sigmoid_precomputation_party_0.branching_precomp.rot_corr.push_back(rot_precomputed_value.first);
                sigmoid_precomputation_party_1.branching_precomp.rot_corr.push_back(rot_precomputed_value.second);

            }
                

                // Generate beaver triples for ANDing the comparison outputs.
                for(int i = 0; i < 6; i++){
                        // Generate Beaver triple vector for P0 and P1.
                        ASSIGN_OR_RETURN(auto beaver_vector_shares,
                                                private_join_and_compute::SampleBeaverTripleVector(
                                                        num_inputs_, 2));

                        sigmoid_precomputation_party_0.branching_precomp.hadamard_triple.push_back(
                                beaver_vector_shares.first
                        );
                                
                        sigmoid_precomputation_party_1.branching_precomp.hadamard_triple.push_back(
                                beaver_vector_shares.second
                        );
                }
                

                // Generate spline gate preprocessing for P0 and P1

                std::pair<std::vector<SplinePrecomputedValue>,
                        std::vector<SplinePrecomputedValue>>
                        spline_precomputed_value;

                // Spline gate preprocessing for positive x
                ASSIGN_OR_RETURN(spline_precomputed_value,
                                 secure_spline_->PerformSplinePrecomputation());

                sigmoid_precomputation_party_0.spline_precomp_pos =
                        spline_precomputed_value.first;

                sigmoid_precomputation_party_1.spline_precomp_pos =
                        spline_precomputed_value.second;

                // Spline gate preprocessing for negative x
                ASSIGN_OR_RETURN(spline_precomputed_value,
                                 secure_spline_->PerformSplinePrecomputation());

                sigmoid_precomputation_party_0.spline_precomp_neg =
                        spline_precomputed_value.first;

                sigmoid_precomputation_party_1.spline_precomp_neg =
                        spline_precomputed_value.second;


                // Generate exponentiation gate preprocessing for P0 and P1


                // Exponentiation gate preprocessing for positive x
                std::pair<MultToAddShare, MultToAddShare> exp_precomputed_value_pos;

                ASSIGN_OR_RETURN(exp_precomputed_value_pos,
                                 SampleMultToAddSharesVector(
                                         num_inputs_,
                                         sigmoid_params_.exp_params.prime_q));

                sigmoid_precomputation_party_0.mta_pos = exp_precomputed_value_pos.first;
                sigmoid_precomputation_party_1.mta_pos = exp_precomputed_value_pos.second;

                // Exponentiation gate preprocessing for negative x
                std::pair<MultToAddShare, MultToAddShare> exp_precomputed_value_neg;

                ASSIGN_OR_RETURN(exp_precomputed_value_neg,
                                 SampleMultToAddSharesVector(
                                         num_inputs_,
                                         sigmoid_params_.exp_params.prime_q));

                sigmoid_precomputation_party_0.mta_neg = exp_precomputed_value_neg.first;
                sigmoid_precomputation_party_1.mta_neg = exp_precomputed_value_neg.second;

                // Generate ROT correlations for evaluation of Polynomial gate


                // ROT correlations for positive x
                std::pair<private_join_and_compute::PolynomialRandomOTPrecomputation,
                        private_join_and_compute::PolynomialRandomOTPrecomputation>
                        rot_precomputed_value_pos;

                ASSIGN_OR_RETURN(rot_precomputed_value_pos,
                                 private_join_and_compute::internal::PolynomialPreprocessRandomOTs(
                                         num_inputs_, fixed_point_factory_));

                sigmoid_precomputation_party_0.rot_corr_pos = rot_precomputed_value_pos.first;
                sigmoid_precomputation_party_1.rot_corr_pos = rot_precomputed_value_pos.second;

                // ROT correlations for negative x
                std::pair<private_join_and_compute::PolynomialRandomOTPrecomputation,
                        private_join_and_compute::PolynomialRandomOTPrecomputation>
                        rot_precomputed_value_neg;

                ASSIGN_OR_RETURN(rot_precomputed_value_neg,
                                 private_join_and_compute::internal::PolynomialPreprocessRandomOTs(
                                         num_inputs_, fixed_point_factory_));

                sigmoid_precomputation_party_0.rot_corr_neg = rot_precomputed_value_neg.first;
                sigmoid_precomputation_party_1.rot_corr_neg = rot_precomputed_value_neg.second;


                // Generate powers correlation for evaluation of Polynomial gate

                // Powers correlation for positive x
                std::pair<std::vector<std::vector<uint64_t>>,
                        std::vector<std::vector<uint64_t>>> powers_precomputed_value_pos;

                ASSIGN_OR_RETURN(powers_precomputed_value_pos,
                                 private_join_and_compute::internal::PolynomialSamplePowersOfRandomVector(
                                         sigmoid_params_.taylor_polynomial_degree,
                                         num_inputs_,
                                         fixed_point_factory_,
                                         fixed_point_factory_->GetParams().primary_ring_modulus
                                 ));

                sigmoid_precomputation_party_0.powers_of_random_vector_pos =
                        powers_precomputed_value_pos.first;

                sigmoid_precomputation_party_1.powers_of_random_vector_pos =
                        powers_precomputed_value_pos.second;

                // Powers correlation for negative x
                std::pair<std::vector<std::vector<uint64_t>>,
                        std::vector<std::vector<uint64_t>>> powers_precomputed_value_neg;

                ASSIGN_OR_RETURN(powers_precomputed_value_neg,
                                 private_join_and_compute::internal::PolynomialSamplePowersOfRandomVector(
                                         sigmoid_params_.taylor_polynomial_degree,
                                         num_inputs_,
                                         fixed_point_factory_,
                                         fixed_point_factory_->GetParams().primary_ring_modulus
                                 ));

                sigmoid_precomputation_party_0.powers_of_random_vector_neg =
                        powers_precomputed_value_neg.first;

                sigmoid_precomputation_party_1.powers_of_random_vector_neg =
                        powers_precomputed_value_neg.second;

//            std::cout << "Precomp() Party 0 exp prime first : "
//            << exp_party_zero_->exp_params_->prime_q << std::endl;

                return std::make_pair(std::move(sigmoid_precomputation_party_0),
                                      std::move(sigmoid_precomputation_party_1));

            }



        StatusOr<std::pair<RoundOneSigmoidNewMicState, RoundOneSigmoidNewMicMessage>>
        SecureSigmoidNewMic::GenerateSigmoidRoundOneMessage(int partyid,
                                                      SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                                                      std::vector<uint64_t> &share_of_sigmoid_inputs) {

            RoundOneSigmoidNewMicState round_one_sigmoid_state;
            RoundOneSigmoidNewMicMessage round_one_sigmoid_message;



            round_one_sigmoid_state.shares_of_sigmoid_inputs = share_of_sigmoid_inputs;

            // Computing branching related next-message-function


            uint64_t modulus = fixed_point_factory_->GetParams().primary_ring_modulus;
            size_t ringBits = fixed_point_factory_->GetParams().num_ring_bits;

            // Creating a default sharing of lower and upper intervals depending on partyid
            std::vector<uint64_t> intervals_share;
            if(partyid == 0)
                intervals_share = std::vector<uint64_t>(7, 0);
            else
                intervals_share = intervals_uint64;


            // Iterating over each interval i
            for(int i = 0; i < 7; i++) {


                std::vector<uint64_t> intervals_share_interval_i(num_inputs_, intervals_share[i]);


                // Reduction from secret-shared comparison to private-input comparison
                std::pair<std::vector<uint64_t>, std::vector<uint64_t>> firstbit_comparisoninput;

                if (partyid == 0) {
                    ASSIGN_OR_RETURN(firstbit_comparisoninput,
                                     private_join_and_compute::secure_comparison::SecretSharedComparisonPrepareInputsPartyZero(
                                             intervals_share_interval_i,
                                             share_of_sigmoid_inputs,
                                             num_inputs_,
                                             modulus,
                                             ringBits));
                } else {
                    ASSIGN_OR_RETURN(firstbit_comparisoninput,
                                     private_join_and_compute::secure_comparison::SecretSharedComparisonPrepareInputsPartyOne(
                                             intervals_share_interval_i,
                                             share_of_sigmoid_inputs,
                                             num_inputs_,
                                             modulus,
                                             ringBits));
                }


                auto first_bit_share = firstbit_comparisoninput.first;
                round_one_sigmoid_state.first_bit_share.push_back(first_bit_share);

                auto comparison_input = firstbit_comparisoninput.second;

                // Generate next-message function for Secure Comparison Round 1

                ASSIGN_OR_RETURN(auto seccomp_round_one,
                                 seccomp_->GenerateComparisonRoundOneMessage(
                                         partyid,
                                         sigmoid_precomputed_value.branching_precomp.comparison_preprocess[i],
                                         comparison_input));


                round_one_sigmoid_state.secure_comparison_round_one_state.push_back(seccomp_round_one.first);


                *(round_one_sigmoid_message.add_round_one_comparison()) = seccomp_round_one.second;


            }


            // Computing Spline related next-message function


            // Spline for positive x

            ASSIGN_OR_RETURN(auto round_one_spline_pos,
                             secure_spline_->GenerateSplineRoundOneMessage(
                                     sigmoid_precomputed_value.spline_precomp_pos,
                                     share_of_sigmoid_inputs
                             ));

            round_one_sigmoid_state.round_one_spline_state_pos = round_one_spline_pos.first;

//             proto set() *was* not working
            *(round_one_sigmoid_message.mutable_round_one_spline_message_pos()) =
                    round_one_spline_pos.second;


            // Spline for negative x



            std::vector<uint64_t> share_of_sigmoid_inputs_neg;

            for(int i = 0; i < num_inputs_; i++){
                share_of_sigmoid_inputs_neg.push_back((-share_of_sigmoid_inputs[i]) % modulus);
            }

            ASSIGN_OR_RETURN(auto round_one_spline_neg,
                             secure_spline_->GenerateSplineRoundOneMessage(
                                     sigmoid_precomputed_value.spline_precomp_neg,
                                     share_of_sigmoid_inputs_neg
                             ));

            round_one_sigmoid_state.round_one_spline_state_neg = round_one_spline_neg.first;

//             proto set() *was* not working
            *(round_one_sigmoid_message.mutable_round_one_spline_message_neg()) =
                    round_one_spline_neg.second;

            // Computing Exp related next-message function

            // Converting share_of_sigmoid_inputs_neg into Fixed Point format
            // because the interface of SecureExponentiationPartyZero and
            // SecureExponentiationPartyOne requires so

            std::vector<FixedPointElement> share_of_sigmoid_inputs_neg_fxp;

            std::vector<FixedPointElement> share_of_sigmoid_inputs_pos_fxp;

            for(int i = 0; i < num_inputs_; i++) {

                ASSIGN_OR_RETURN(FixedPointElement fpe_neg,
                                 fixed_point_factory_->ImportFixedPointElementFromUint64(
                                         share_of_sigmoid_inputs_neg[i]));
                share_of_sigmoid_inputs_neg_fxp.push_back(fpe_neg);

                ASSIGN_OR_RETURN(FixedPointElement fpe_pos,
                                 fixed_point_factory_->ImportFixedPointElementFromUint64(
                                         share_of_sigmoid_inputs[i]));
                share_of_sigmoid_inputs_pos_fxp.push_back(fpe_pos);

            }

            // TODO: proto set() not working
            if (partyid == 0){

                // TODO : Temporary fix for zeroing error in exp_party_zero_
                ASSIGN_OR_RETURN(auto temp_zero,
                                 SecureExponentiationPartyZero::Create(
                                         fixed_point_factory_->GetParams(),
                                         sigmoid_params_.exp_params));


                std::pair<BatchedExponentiationPartyZeroMultToAddMessage,
                        std::vector<private_join_and_compute::SecureExponentiationPartyZero::State>>
                        round_one_exp_party0_pos, round_one_exp_party0_neg;

//                std::cout << "Party 0 exp prime : " << exp_party_zero_->exp_params_->prime_q << std::endl;

                ASSIGN_OR_RETURN(round_one_exp_party0_pos,
                                 GenerateVectorMultToAddMessagePartyZero(
                                         temp_zero,
                                         share_of_sigmoid_inputs_neg_fxp,
                                         sigmoid_precomputed_value.mta_pos.first,
                                         sigmoid_precomputed_value.mta_pos.second));

                ASSIGN_OR_RETURN(round_one_exp_party0_neg,
                                 GenerateVectorMultToAddMessagePartyZero(
                                         temp_zero,
                                         share_of_sigmoid_inputs_pos_fxp,
                                         sigmoid_precomputed_value.mta_neg.first,
                                         sigmoid_precomputed_value.mta_neg.second));

                round_one_sigmoid_state.round_one_exp_state_party0_pos =
                        round_one_exp_party0_pos.second;

                round_one_sigmoid_state.round_one_exp_state_party0_neg =
                        round_one_exp_party0_neg.second;
//

                // TODO : Migrate to oneof in secure_sigmoid.proto

                *(round_one_sigmoid_message.mutable_round_one_exp_message_party0_pos()) =
                        round_one_exp_party0_pos.first;

                *(round_one_sigmoid_message.mutable_round_one_exp_message_party0_neg()) =
                        round_one_exp_party0_neg.first;
            }

            if (partyid == 1){

                // TODO : Temporary fix for zeroing error in exp_party_one_
                ASSIGN_OR_RETURN(auto temp_one,
                                 SecureExponentiationPartyOne::Create(
                                         fixed_point_factory_->GetParams(),
                                         sigmoid_params_.exp_params));

                std::pair<BatchedExponentiationPartyOneMultToAddMessage,
                        std::vector<private_join_and_compute::SecureExponentiationPartyOne::State>>
                        round_one_exp_party1_pos, round_one_exp_party1_neg;

//                std::cout << "Party 1 exp prime : " << exp_party_one_->exp_params_->prime_q << std::endl;


                ASSIGN_OR_RETURN(round_one_exp_party1_pos,
                                 GenerateVectorMultToAddMessagePartyOne(
                                         temp_one,
                                         share_of_sigmoid_inputs_neg_fxp,
                                         sigmoid_precomputed_value.mta_pos.first,
                                         sigmoid_precomputed_value.mta_pos.second));


                ASSIGN_OR_RETURN(round_one_exp_party1_neg,
                                 GenerateVectorMultToAddMessagePartyOne(
                                         temp_one,
                                         share_of_sigmoid_inputs_pos_fxp,
                                         sigmoid_precomputed_value.mta_neg.first,
                                         sigmoid_precomputed_value.mta_neg.second));

                round_one_sigmoid_state.round_one_exp_state_party1_pos =
                        round_one_exp_party1_pos.second;

                round_one_sigmoid_state.round_one_exp_state_party1_neg =
                        round_one_exp_party1_neg.second;


                // TODO : Migrate to oneof in secure_sigmoid.proto


                *(round_one_sigmoid_message.mutable_round_one_exp_message_party1_pos()) =
                        round_one_exp_party1_pos.first;

                *(round_one_sigmoid_message.mutable_round_one_exp_message_party1_neg()) =
                        round_one_exp_party1_neg.first;
            }


            return std::pair<RoundOneSigmoidNewMicState, RoundOneSigmoidNewMicMessage>(
                    round_one_sigmoid_state, round_one_sigmoid_message);

        }



               StatusOr<std::pair<RoundTwoSigmoidNewMicState, RoundTwoSigmoidNewMicMessage>>
               SecureSigmoidNewMic::GenerateSigmoidRoundTwoMessage(
                       int partyid,
                       SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                       RoundOneSigmoidNewMicState round_one_state_this_party,
                       RoundOneSigmoidNewMicMessage round_one_msg_other_party) {


                   RoundTwoSigmoidNewMicState round_two_sigmoid_state;
                   RoundTwoSigmoidNewMicMessage round_two_sigmoid_message;

                   uint64_t modulus = fixed_point_factory_->GetParams().primary_ring_modulus;


                   // Computing Spline related next-message function

                   // Spline for positive x

                   ASSIGN_OR_RETURN(auto round_two_spline_pos,
                                    secure_spline_->GenerateSplineRoundTwoMessage(
                                            partyid,
                                            sigmoid_precomputed_value.spline_precomp_pos,
                                            round_one_state_this_party.round_one_spline_state_pos,
                                            round_one_msg_other_party.round_one_spline_message_pos()
                                    ));
       //
                   round_two_sigmoid_state.round_two_spline_state_pos = round_two_spline_pos.first;

                   // proto set() *was* not working
                   *(round_two_sigmoid_message.mutable_round_two_spline_message_pos()) =
                           round_two_spline_pos.second;

                   // Spline for negative x

                   ASSIGN_OR_RETURN(auto round_two_spline_neg,
                                    secure_spline_->GenerateSplineRoundTwoMessage(
                                            partyid,
                                            sigmoid_precomputed_value.spline_precomp_neg,
                                            round_one_state_this_party.round_one_spline_state_neg,
                                            round_one_msg_other_party.round_one_spline_message_neg()
                                    ));

                   round_two_sigmoid_state.round_two_spline_state_neg = round_two_spline_neg.first;

                   *(round_two_sigmoid_message.mutable_round_two_spline_message_neg()) =
                           round_two_spline_neg.second;


                   // Computing Exp related final output function
                   if(partyid == 0){


                       // TODO : Temporary fix for zeroing error in exp_party_zero_
                       ASSIGN_OR_RETURN(auto temp_zero,
                                        SecureExponentiationPartyZero::Create(
                                                fixed_point_factory_->GetParams(),
                                                sigmoid_params_.exp_params));

                       // Exp output share for positive x
                       ASSIGN_OR_RETURN(std::vector<FixedPointElement> output_fpe_p0_pos,
                                        VectorExponentiationPartyZero(temp_zero,
                                                                      round_one_msg_other_party.round_one_exp_message_party1_pos(),
                                                                      round_one_state_this_party.round_one_exp_state_party0_pos));

                       round_two_sigmoid_state.exp_result_state_pos = output_fpe_p0_pos;

                       // Exp output share for negative x
                       ASSIGN_OR_RETURN(std::vector<FixedPointElement> output_fpe_p0_neg,
                                        VectorExponentiationPartyZero(temp_zero,
                                                                      round_one_msg_other_party.round_one_exp_message_party1_neg(),
                                                                      round_one_state_this_party.round_one_exp_state_party0_neg));

                       round_two_sigmoid_state.exp_result_state_neg = output_fpe_p0_neg;
                   }

                   if(partyid == 1){

                       // TODO : Temporary fix for zeroing error in exp_party_one_
                       ASSIGN_OR_RETURN(auto temp_one,
                                        SecureExponentiationPartyOne::Create(
                                                fixed_point_factory_->GetParams(),
                                                sigmoid_params_.exp_params));

                       // Exp output share for positive x
                       ASSIGN_OR_RETURN(std::vector<FixedPointElement> output_fpe_p1_pos,
                                        VectorExponentiationPartyOne(temp_one,
                                                                     round_one_msg_other_party.round_one_exp_message_party0_pos(),
                                                                     round_one_state_this_party.round_one_exp_state_party1_pos));

                       round_two_sigmoid_state.exp_result_state_pos = output_fpe_p1_pos;

                       // Exp output share for negative x
                       ASSIGN_OR_RETURN(std::vector<FixedPointElement> output_fpe_p1_neg,
                                        VectorExponentiationPartyOne(temp_one,
                                                                     round_one_msg_other_party.round_one_exp_message_party0_neg(),
                                                                     round_one_state_this_party.round_one_exp_state_party1_neg));

                       round_two_sigmoid_state.exp_result_state_neg = output_fpe_p1_neg;
                   }



                   // Computing Polynomial related next-message function

                   std::vector<uint64_t> exp_result_state_uint64_pos, exp_result_state_uint64_neg;

                   for(int i = 0; i < num_inputs_; i++){
                       exp_result_state_uint64_pos.push_back(
                               round_two_sigmoid_state.exp_result_state_pos[i].ExportToUint64()
                       );

                       exp_result_state_uint64_neg.push_back(
                               round_two_sigmoid_state.exp_result_state_neg[i].ExportToUint64()
                       );
                   }

                   // Polynomial next message function for positive x
                   ASSIGN_OR_RETURN(auto round_one_poly_pos,
                                    PolynomialGenerateRoundOneMessage(
                                            exp_result_state_uint64_pos,
                                            sigmoid_precomputed_value.powers_of_random_vector_pos,
                                            sigmoid_precomputed_value.rot_corr_pos,
                                            fixed_point_factory_,
                                            modulus
                                    ));


                   round_two_sigmoid_state.round_one_polynomial_state_pos =
                           round_one_poly_pos.first;

                   *(round_two_sigmoid_message.mutable_round_one_polynomial_message_pos()) =
                           round_one_poly_pos.second;


                   // Polynomial next message function for negative x
                   ASSIGN_OR_RETURN(auto round_one_poly_neg,
                                    PolynomialGenerateRoundOneMessage(
                                            exp_result_state_uint64_neg,
                                            sigmoid_precomputed_value.powers_of_random_vector_neg,
                                            sigmoid_precomputed_value.rot_corr_neg,
                                            fixed_point_factory_,
                                            modulus
                                    ));

                   round_two_sigmoid_state.round_one_polynomial_state_neg =
                           round_one_poly_neg.first;

                   *(round_two_sigmoid_message.mutable_round_one_polynomial_message_neg()) =
                           round_one_poly_neg.second;

                   // Next message for Secure Comparison

                   round_two_sigmoid_state.secure_comparison_round_two_state.reserve(7);

                   // Copying forward first_bit_share info onto next state
                   round_two_sigmoid_state.first_bit_share = round_one_state_this_party.first_bit_share;


                   // new batched
                   std::vector<RoundOneSecureComparisonMessage> round_one_msgs_other_party;
                   for (size_t i = 0; i < 7; i++) {
                       round_one_msgs_other_party.push_back(round_one_msg_other_party.round_one_comparison(i));
                   }
                   ASSIGN_OR_RETURN(
                           auto seccom_round_two_state_message_batched,
                           seccomp_->BatchGenerateComparisonRoundTwoMessage(
                                   partyid,
                                   7,
                                   sigmoid_precomputed_value.branching_precomp.comparison_preprocess,
                                   round_one_state_this_party.secure_comparison_round_one_state,
                                   round_one_msgs_other_party));
                   for (int i = 0; i < 7; i++) {
                       round_two_sigmoid_state.secure_comparison_round_two_state.push_back(
                               seccom_round_two_state_message_batched[i].first);

                       *(round_two_sigmoid_message.add_round_two_comparison()) =
                               seccom_round_two_state_message_batched[i].second;

                   }

                   // old (not batched)
                   /*(for (int i = 0; i < 7; i++){
                   ASSIGN_OR_RETURN(
                                   auto seccom_round_two_state_message,
                                   seccomp_->GenerateComparisonRoundTwoMessage(
                                           partyid,
                                           sigmoid_precomputed_value.branching_precomp.comparison_preprocess[i],
                                           round_one_state_this_party.secure_comparison_round_one_state[i],
                                           round_one_msg_other_party.round_one_comparison(i)));

                       round_two_sigmoid_state.secure_comparison_round_two_state.push_back(
                                   seccom_round_two_state_message.first);

                           *(round_two_sigmoid_message.add_round_two_comparison()) =
                                   seccom_round_two_state_message.second;

                   }*/


                   return std::pair<RoundTwoSigmoidNewMicState, RoundTwoSigmoidNewMicMessage>(
                           round_two_sigmoid_state,
                           round_two_sigmoid_message
                   );

               }




               StatusOr<std::pair<RoundThreeSigmoidNewMicState, RoundThreeSigmoidNewMicMessage>>
               SecureSigmoidNewMic::GenerateSigmoidRoundThreeMessage(
                       int partyid,
                       SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                       RoundTwoSigmoidNewMicState round_two_state_this_party,
                       RoundTwoSigmoidNewMicMessage round_two_msg_other_party) {

                   RoundThreeSigmoidNewMicState round_three_sigmoid_state;
                   RoundThreeSigmoidNewMicMessage round_three_sigmoid_message;

                   uint64_t modulus =
                           fixed_point_factory_->GetParams().primary_ring_modulus;

                   // Computing final result of spline branch in secret shared form


                   // Spline for positive x

                   std::vector<uint64_t> spline_res_in_secret_shared_form_pos;

                   ASSIGN_OR_RETURN(
                           spline_res_in_secret_shared_form_pos,
                           secure_spline_->GenerateSplineResult(
                                   partyid, sigmoid_precomputed_value.spline_precomp_pos,
                                   round_two_state_this_party.round_two_spline_state_pos,
                                   round_two_msg_other_party.round_two_spline_message_pos()));

                   round_three_sigmoid_state.spline_res_in_secret_shared_form_pos =
                           spline_res_in_secret_shared_form_pos;


                   // Spline for negative x

                   std::vector<uint64_t> spline_res_in_secret_shared_form_neg;

                   ASSIGN_OR_RETURN(
                           spline_res_in_secret_shared_form_neg,
                           secure_spline_->GenerateSplineResult(
                                   partyid, sigmoid_precomputed_value.spline_precomp_neg,
                                   round_two_state_this_party.round_two_spline_state_neg,
                                   round_two_msg_other_party.round_two_spline_message_neg()));

                   round_three_sigmoid_state.spline_res_in_secret_shared_form_neg =
                           spline_res_in_secret_shared_form_neg;

                   // Computing next message function for polynomial

                   // Define coefficients based on taylor_polynomial_degree

                   PolynomialCoefficients coefficients;

                   for(int i = 0; i <= sigmoid_params_.taylor_polynomial_degree; i++){
                       if(i % 2 == 0)
                           coefficients.coefficients.push_back(1);
                       else
                           coefficients.coefficients.push_back(-1);
                   }


                   if(partyid == 0){

                       // Polynomial next message function for positive x
                       ASSIGN_OR_RETURN(auto round_two_poly_pos,
                                        PolynomialGenerateRoundTwoMessagePartyZero(
                                                coefficients,
                                                round_two_state_this_party.round_one_polynomial_state_pos,
                                                sigmoid_precomputed_value.powers_of_random_vector_pos,
                                                sigmoid_precomputed_value.rot_corr_pos,
                                                round_two_msg_other_party.round_one_polynomial_message_pos(),
                                                fixed_point_factory_,
                                                modulus));

                       round_three_sigmoid_state.round_two_polynomial_state_pos =
                               round_two_poly_pos.first;

                       *(round_three_sigmoid_message.mutable_round_two_polynomial_message_pos()) =
                               round_two_poly_pos.second;

                       // Polynomial next message function for negative x
                       ASSIGN_OR_RETURN(auto round_two_poly_neg,
                                        PolynomialGenerateRoundTwoMessagePartyZero(
                                                coefficients,
                                                round_two_state_this_party.round_one_polynomial_state_neg,
                                                sigmoid_precomputed_value.powers_of_random_vector_neg,
                                                sigmoid_precomputed_value.rot_corr_neg,
                                                round_two_msg_other_party.round_one_polynomial_message_neg(),
                                                fixed_point_factory_,
                                                modulus));

                       round_three_sigmoid_state.round_two_polynomial_state_neg =
                               round_two_poly_neg.first;

                       *(round_three_sigmoid_message.mutable_round_two_polynomial_message_neg()) =
                               round_two_poly_neg.second;

                   }

                   if(partyid == 1){

                       // Polynomial next message function for positive x
                       ASSIGN_OR_RETURN(auto round_two_poly_pos,
                                        PolynomialGenerateRoundTwoMessagePartyOne(
                                                coefficients,
                                                round_two_state_this_party.round_one_polynomial_state_pos,
                                                sigmoid_precomputed_value.powers_of_random_vector_pos,
                                                sigmoid_precomputed_value.rot_corr_pos,
                                                round_two_msg_other_party.round_one_polynomial_message_pos(),
                                                fixed_point_factory_,
                                                modulus));

                       round_three_sigmoid_state.round_two_polynomial_state_pos =
                               round_two_poly_pos.first;

                       *(round_three_sigmoid_message.mutable_round_two_polynomial_message_pos()) =
                               round_two_poly_pos.second;


                       // Polynomial next message function for negative x
                       ASSIGN_OR_RETURN(auto round_two_poly_neg,
                                        PolynomialGenerateRoundTwoMessagePartyOne(
                                                coefficients,
                                                round_two_state_this_party.round_one_polynomial_state_neg,
                                                sigmoid_precomputed_value.powers_of_random_vector_neg,
                                                sigmoid_precomputed_value.rot_corr_neg,
                                                round_two_msg_other_party.round_one_polynomial_message_neg(),
                                                fixed_point_factory_,
                                                modulus));

                       round_three_sigmoid_state.round_two_polynomial_state_neg =
                               round_two_poly_neg.first;

                       *(round_three_sigmoid_message.mutable_round_two_polynomial_message_neg()) =
                               round_two_poly_neg.second;

                   }

                   // Carrying forward the information about Round 1 poly state
                   round_three_sigmoid_state.round_one_polynomial_state_pos =
                           round_two_state_this_party.round_one_polynomial_state_pos;

                   round_three_sigmoid_state.round_one_polynomial_state_neg =
                           round_two_state_this_party.round_one_polynomial_state_neg;

                   // Secure comparison

                   // Copying forward first_bit_share onto next state
                   round_three_sigmoid_state.first_bit_share = round_two_state_this_party.first_bit_share;


                   for(int i = 0; i < 7; i++){

                       ASSIGN_OR_RETURN(
                               auto seccom_round_three_state_message,
                               seccomp_->GenerateComparisonRoundThreeMessage(
                                       partyid,
                                       sigmoid_precomputed_value.branching_precomp.comparison_preprocess[i],
                                       round_two_state_this_party.secure_comparison_round_two_state[i],
                                       round_two_msg_other_party.round_two_comparison(i)));


                       round_three_sigmoid_state.secure_comparison_round_three_state.push_back(
                               seccom_round_three_state_message.first);

                       *(round_three_sigmoid_message.add_round_three_comparison()) =
                               seccom_round_three_state_message.second;

                   }

                   return std::pair<RoundThreeSigmoidNewMicState, RoundThreeSigmoidNewMicMessage>(
                           round_three_sigmoid_state,
                           round_three_sigmoid_message
                   );
               }



               StatusOr<std::pair<RoundThreePointFiveSigmoidNewMicState, RoundThreePointFiveSigmoidNewMicMessage>>
               SecureSigmoidNewMic::GenerateSigmoidRoundThreePointFiveMessage(
                       int partyid,
                       SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                       RoundThreeSigmoidNewMicState round_three_state_this_party,
                       RoundThreeSigmoidNewMicMessage round_three_msg_other_party) {

                   RoundThreePointFiveSigmoidNewMicState round_three_point_five_sigmoid_state;
                   RoundThreePointFiveSigmoidNewMicMessage round_three_point_five_sigmoid_message;

                   uint64_t modulus = fixed_point_factory_->GetParams().primary_ring_modulus;

                   // Compute Share of Polynomial output by unmasking round 2 message and adding shares

                   // Polynomial output share for positive x
                   ASSIGN_OR_RETURN(
                           auto poly_output_share_pos,
                           PolynomialOutput(
                                   round_three_msg_other_party.round_two_polynomial_message_pos(),
                                   sigmoid_precomputed_value.rot_corr_pos,
                                   round_three_state_this_party.round_two_polynomial_state_pos,
                                   round_three_state_this_party.round_one_polynomial_state_pos,
                                   modulus));


                   // Polynomial output share for negative x
                   ASSIGN_OR_RETURN(
                           auto poly_output_share_neg,
                           PolynomialOutput(
                                   round_three_msg_other_party.round_two_polynomial_message_neg(),
                                   sigmoid_precomputed_value.rot_corr_neg,
                                   round_three_state_this_party.round_two_polynomial_state_neg,
                                   round_three_state_this_party.round_one_polynomial_state_neg,
                                   modulus));

                   // First, create a vector of length 6 containing the output of
                   // each of the 6 branches on input x :
                   // branch_res_in_secret_shared_form = [ _, _, _, _, _, _ ]
                   // Since we are in the batched setting, we will actually need
                   // a vector of vectors

                   // Outer vector is over batch of inputs, inner vector is over 6 intervals
                   std::vector<std::vector<uint64_t>> branch_res_in_secret_shared_form;

                   for(int i = 0; i < num_inputs_; i++){
                       std::vector<uint64_t> branch_res_in_secret_shared_form_i;

                       // Filling the result for Algorithm 1
                       branch_res_in_secret_shared_form_i.push_back(
                               round_three_state_this_party.spline_res_in_secret_shared_form_pos[i]);

                       //Filling the result for Algorithm 2

                       branch_res_in_secret_shared_form_i.push_back(
                               poly_output_share_pos[i]
                       );


                       // Filling the result for Algorithm 3

                       branch_res_in_secret_shared_form_i.push_back(
                               sigmoid_precomputed_value.share_of_one[i]);


                       // Filling the result for Algorithm 4

                       branch_res_in_secret_shared_form_i.push_back(
                               sigmoid_precomputed_value.share_of_zero[i]);


                       // Filling the result for Algorithm 5

                       branch_res_in_secret_shared_form_i.push_back(
                               (sigmoid_precomputed_value.share_of_one[i] -
                                poly_output_share_neg[i]) % modulus);


                       // Filling the result for Algorithm 6

                       branch_res_in_secret_shared_form_i.push_back(
                               (sigmoid_precomputed_value.share_of_one[i] -
                                round_three_state_this_party.spline_res_in_secret_shared_form_neg[i]
                               ) % modulus);

                       branch_res_in_secret_shared_form.push_back(
                               branch_res_in_secret_shared_form_i
                       );
                   }

                   round_three_point_five_sigmoid_state.branch_res_in_secret_shared_form = branch_res_in_secret_shared_form;

                   // Second, create a vector of length 7 containing the output of
                   // comparison on each of the 7 interval bounds for input x :
                   // branch_conditional_in_secret_shared_form = [ _, _, _, _, _, _, _ ]
                   // Again, since we are in the batched setting, we will actually need
                   // a vector of vectors


                  // Secure comparison

                   for(int i = 0; i < 7; i++){

                       // Compute carry bit (as written in secure_comparison_test)
//                       std::vector<uint64_t> carry_share =
//                               round_three_point_five_sigmoid_state.combination_input[i].even_comparison_even_equality_output_shares;

                       std::vector<uint64_t> carry_share;

                       ASSIGN_OR_RETURN(carry_share,
                                        seccomp_->GenerateComparisonResult(partyid,
                                                                          sigmoid_precomputed_value.branching_precomp.comparison_preprocess[i],
                                                                          round_three_state_this_party.secure_comparison_round_three_state[i],
                                                                          round_three_msg_other_party.round_three_comparison(i)));

                       // Execute SecretSharedComparisonFinishReduction to get comparison secret share result
                       // For secret-shared comparison, there is extra step of adding shares of first bit
                       ASSIGN_OR_RETURN(std::vector<uint64_t> comparison_output_share_interval_i,
                                        private_join_and_compute::secure_comparison::SecretSharedComparisonFinishReduction(
                                       round_three_state_this_party.first_bit_share[i],
                                       carry_share,
                                       num_inputs_));

                       round_three_point_five_sigmoid_state.comparison_output_share.push_back(
                               comparison_output_share_interval_i);

                   }


                   // Perform Hadamard next msg for each of the 6 intervals
                   // a = [0, 1.0, l_f * ln 2, 2^(l-2 -l_f), -2^(l-2-l_f), -l_f * ln 2, -1.0]
                   // a_0 > x = r_0
                   // a_1 > x = r_1
                   // a_2 > x = r_2
                   // a_3 > x = r_3
                   // a_4 > x = r_4
                   // a_5 > x = r_5
                   // a_6 > x = r_6

                   // P = [r'_0, r'_1, r'_2, r'_4, r'_5, r'_6]


                   // Q = [r_1, r_2, r_3, r_5, r_6, r_0]


                   // R = P AND Q = [(r'_0 AND r_1), (r'_1 AND r_2), (r'_2 AND r_3), (r'_4 AND r_5), (r'_5 AND r_6), (r'_6 AND r_0)]

                   // [(x >= 0 AND x < 1.0),
                   // (x >= 1.0 AND x < lf * ln 2),
                   // (x >= lf * ln 2 AND x < 2^(l-2-l_f)),
                   // (x >= -2^(l-2-l_f) AND x < -l_f * ln 2),
                   // (x >= -l_f * ln 2 AND x < -1.0),
                   // (x >= -1.0 AND x < 0)]

                   // Computing negated comparison output share
                   for(int i = 0; i < 7; i++){
                           std::vector<uint64_t> negated_comparison_output_share_interval_i;

                           if(partyid == 0){
                                   negated_comparison_output_share_interval_i =
                                   round_three_point_five_sigmoid_state.comparison_output_share[i];
                           }
                           else{
                                   ASSIGN_OR_RETURN(negated_comparison_output_share_interval_i,
                                   BatchedModAdd(round_three_point_five_sigmoid_state.comparison_output_share[i],
                                   std::vector<uint64_t>(num_inputs_, 1),
                                   2));
                           }
                           round_three_point_five_sigmoid_state.negated_comparison_output_share.push_back(
                                   negated_comparison_output_share_interval_i
                           );
                   }

                   // Computing Hadamard product next message function

                   // Each party generates its batched multiplication message.
                   std::pair<private_join_and_compute::BatchedMultState, private_join_and_compute::MultiplicationGateMessage>
                           hadamard_state_plus_msg;

                   // Interval 1
                   ASSIGN_OR_RETURN(
                           hadamard_state_plus_msg,
                           GenerateHadamardProductMessage(
                                   round_three_point_five_sigmoid_state.negated_comparison_output_share[0],
                                   round_three_point_five_sigmoid_state.comparison_output_share[1],
                                   sigmoid_precomputed_value.branching_precomp.hadamard_triple[0],
                                   2));

                   round_three_point_five_sigmoid_state.hadamard_state_for_ANDing_comparison_results.push_back(
                           hadamard_state_plus_msg.first
                   );

                   *(round_three_point_five_sigmoid_message.add_hadamard_message_for_anding_comparison_results()) =
                   hadamard_state_plus_msg.second;

                   // Interval 2
                   ASSIGN_OR_RETURN(
                           hadamard_state_plus_msg,
                           GenerateHadamardProductMessage(
                                   round_three_point_five_sigmoid_state.negated_comparison_output_share[1],
                                   round_three_point_five_sigmoid_state.comparison_output_share[2],
                                   sigmoid_precomputed_value.branching_precomp.hadamard_triple[1],
                                   2));

                   round_three_point_five_sigmoid_state.hadamard_state_for_ANDing_comparison_results.push_back(
                           hadamard_state_plus_msg.first
                   );

                   *(round_three_point_five_sigmoid_message.add_hadamard_message_for_anding_comparison_results()) =
                   hadamard_state_plus_msg.second;


                   // Interval 3
                   ASSIGN_OR_RETURN(
                           hadamard_state_plus_msg,
                           GenerateHadamardProductMessage(
                                   round_three_point_five_sigmoid_state.negated_comparison_output_share[2],
                                   round_three_point_five_sigmoid_state.comparison_output_share[3],
                                   sigmoid_precomputed_value.branching_precomp.hadamard_triple[2],
                                   2));

                   round_three_point_five_sigmoid_state.hadamard_state_for_ANDing_comparison_results.push_back(
                           hadamard_state_plus_msg.first
                   );

                   *(round_three_point_five_sigmoid_message.add_hadamard_message_for_anding_comparison_results()) =
                   hadamard_state_plus_msg.second;

                   // Interval 4
                   ASSIGN_OR_RETURN(
                           hadamard_state_plus_msg,
                           GenerateHadamardProductMessage(
                                   round_three_point_five_sigmoid_state.negated_comparison_output_share[4],
                                   round_three_point_five_sigmoid_state.comparison_output_share[5],
                                   sigmoid_precomputed_value.branching_precomp.hadamard_triple[3],
                                   2));

                   round_three_point_five_sigmoid_state.hadamard_state_for_ANDing_comparison_results.push_back(
                           hadamard_state_plus_msg.first
                   );

                   *(round_three_point_five_sigmoid_message.add_hadamard_message_for_anding_comparison_results()) =
                   hadamard_state_plus_msg.second;

                   // Interval 5
                   ASSIGN_OR_RETURN(
                           hadamard_state_plus_msg,
                           GenerateHadamardProductMessage(
                                   round_three_point_five_sigmoid_state.negated_comparison_output_share[5],
                                   round_three_point_five_sigmoid_state.comparison_output_share[6],
                                   sigmoid_precomputed_value.branching_precomp.hadamard_triple[4],
                                   2));

                   round_three_point_five_sigmoid_state.hadamard_state_for_ANDing_comparison_results.push_back(
                           hadamard_state_plus_msg.first
                   );

                   *(round_three_point_five_sigmoid_message.add_hadamard_message_for_anding_comparison_results()) =
                   hadamard_state_plus_msg.second;

                   // Interval 6
                   ASSIGN_OR_RETURN(
                           hadamard_state_plus_msg,
                           GenerateHadamardProductMessage(
                                   round_three_point_five_sigmoid_state.negated_comparison_output_share[6],
                                   round_three_point_five_sigmoid_state.comparison_output_share[0],
                                   sigmoid_precomputed_value.branching_precomp.hadamard_triple[5],
                                   2));

                   round_three_point_five_sigmoid_state.hadamard_state_for_ANDing_comparison_results.push_back(
                           hadamard_state_plus_msg.first
                   );

                   *(round_three_point_five_sigmoid_message.add_hadamard_message_for_anding_comparison_results()) =
                   hadamard_state_plus_msg.second;

                   return std::pair<RoundThreePointFiveSigmoidNewMicState, RoundThreePointFiveSigmoidNewMicMessage>(
                           round_three_point_five_sigmoid_state,
                           round_three_point_five_sigmoid_message
                   );
               }


           std::pair<uint64_t, uint64_t > swap(std::pair<uint64_t, uint64_t > inp, bool is_swap){
               if(is_swap == false) return inp;
               else return std::make_pair(inp.second, inp.first);
           }




            StatusOr<std::pair<RoundFourSigmoidNewMicState, RoundFourSigmoidNewMicMessage>>
           SecureSigmoidNewMic::GenerateSigmoidRoundFourMessage(
                   int partyid,
                   SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                   RoundThreePointFiveSigmoidNewMicState round_three_point_five_state_this_party,
                   RoundThreePointFiveSigmoidNewMicMessage round_three_point_five_msg_other_party){

                   RoundFourSigmoidNewMicState round_four_sigmoid_state;
                   RoundFourSigmoidNewMicMessage round_four_sigmoid_message;

                   //Copy forward branch_res_in_secret_shared_form from previous state
                   round_four_sigmoid_state.branch_res_in_secret_shared_form =
                   round_three_point_five_state_this_party.branch_res_in_secret_shared_form;


                   // Compute the AND result to derive interval containment result
                   for(int i = 0; i < 6; i++){
                           std::vector<uint64_t> mic_output_share_interval_i;

                           // Compute hadamard product result share
                           if (partyid == 0) {
                       // Execute HadamardProductPartyZero.
                           ASSIGN_OR_RETURN(
                                   mic_output_share_interval_i,
                                   private_join_and_compute::HadamardProductPartyZero(
                                           round_three_point_five_state_this_party.hadamard_state_for_ANDing_comparison_results[i],
                                           sigmoid_precomputed_value.branching_precomp.hadamard_triple[i],
                                           round_three_point_five_msg_other_party.hadamard_message_for_anding_comparison_results(i),
                                           0,
                                           2));

                           }

                           else {
                           // Execute HadamardProductPartyOne.
                           ASSIGN_OR_RETURN(
                                   mic_output_share_interval_i,
                                   private_join_and_compute::HadamardProductPartyOne(
                                           round_three_point_five_state_this_party.hadamard_state_for_ANDing_comparison_results[i],
                                           sigmoid_precomputed_value.branching_precomp.hadamard_triple[i],
                                           round_three_point_five_msg_other_party.hadamard_message_for_anding_comparison_results(i),
                                           0,
                                           2));
                           }

                           round_four_sigmoid_state.mic_output_share.push_back(mic_output_share_interval_i);
                   }

                   // Let's say c = (c0, c1) represents the comparison output share - boolean sharing
                   // Let's say that we want to select k (e.g. k will be the output of exp+taylor)
                   // where k = (k0, k1) arithmetic shared

                   // Protocol : Run 2 OTs in parallel - OT_1 and OT_2

                   // In OT_1, Party 0 is the sender and Party 1 is the receiver
                   // Sender samples r_0
                   // If c_0 = 0, Sender sets its inputs as (-r_0, -r_0 + k_0)
                   // If c_0 = 1, Sender sets its inputs as (-r_0 + k_0, -r_0)
                   // Receiver sets its choice bit as c_1
                   // Receiver outputs out_OT_1

                   // In OT_2, Party 1 is the sender and Party 0 is the receiver
                   // Sender samples r_1
                   // If c_1 = 0, Sender sets its inputs as (-r_1, -r_1 + k_1)
                   // If c_1 = 1, Sender sets its inputs as (-r_1 + k_1, -r_1)
                   // Receiver sets its choice bit as c_0
                   // Receiver outputs out_OT_2

                   // Finally,
                   // Party 0 outputs (r_0 + out_OT_2)
                   // Party 1 outputs (r_1 + out_OT_1)
                   // These 2 outputs will form an arithmetic sharing of either k (if c = 1) or 0 (if c = 0)

                     // Perform MUX Round 1

                     for(int i = 0; i < 6; i++){

                       MUX_round_1 mux_round_1;

                       for(int j = 0 ; j < num_inputs_; j++){

                         //  uint64_t receiver_choice_bit = comparison_output_share_interval_i[j];
                         uint64_t receiver_choice_bit = round_four_sigmoid_state.mic_output_share[i][j];

                           // XORing receiver choice bit with its ROT choice bit
                           uint64_t receiver_choice_bit_xor_rot_choice_bit =
                                            ModAdd(receiver_choice_bit,
                                                       sigmoid_precomputed_value.branching_precomp.rot_corr[i].receiver_msgs[j].receiver_choice,
                                                       2);
                           mux_round_1.add_receiver_bit_xor_rot_choice_bit(receiver_choice_bit_xor_rot_choice_bit);
                       }


                       *(round_four_sigmoid_message.add_mux_round_1()) = mux_round_1;

                     }

                     return std::pair<RoundFourSigmoidNewMicState, RoundFourSigmoidNewMicMessage>(
                       round_four_sigmoid_state,
                       round_four_sigmoid_message
               );

           }



           StatusOr<std::pair<RoundFiveSigmoidNewMicState, RoundFiveSigmoidNewMicMessage>>
           SecureSigmoidNewMic::GenerateSigmoidRoundFiveMessage(
                   int partyid,
                   SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                   RoundFourSigmoidNewMicState round_four_state_this_party,
                   RoundFourSigmoidNewMicMessage round_four_msg_other_party){

               RoundFiveSigmoidNewMicState round_five_sigmoid_state;
               RoundFiveSigmoidNewMicMessage round_five_sigmoid_message;


               // Copy forward comparison_output_share from previous state
               round_five_sigmoid_state.mic_output_share.reserve(6);
               round_five_sigmoid_state.mic_output_share = round_four_state_this_party.mic_output_share;

               // Generate MUX Round 2
               // Mask the polynomial_input_{0,1}

               const absl::string_view kSampleSeed = absl::string_view();
               ASSIGN_OR_RETURN(auto rng, private_join_and_compute::BasicRng::Create(kSampleSeed));
               uint64_t modulus = fixed_point_factory_->GetParams().primary_ring_modulus;

               // Iterate over each of the 6 intervals
               for(int i = 0; i < 6; i++){
                   std::vector<uint64_t> masked_mux_sender_first_msg (num_inputs_, 0);
                   std::vector<uint64_t> masked_mux_sender_second_msg (num_inputs_, 0);

                   MUX_round_2 mux_round_2_interval_i;


                   std::vector<uint64_t> mux_randomness_i;

                   for (size_t idx = 0; idx < num_inputs_; idx++) {


                       std::pair<uint64_t, uint64_t> masks(sigmoid_precomputed_value.branching_precomp.rot_corr[i].sender_msgs[idx].sender_msg0,
                                                           sigmoid_precomputed_value.branching_precomp.rot_corr[i].sender_msgs[idx].sender_msg1);


                       std::pair<uint64_t, uint64_t> swapped_masks = swap(masks, round_four_msg_other_party.mux_round_1(i).receiver_bit_xor_rot_choice_bit(idx));


                       // Sampling r for MUX sender strings
                       ASSIGN_OR_RETURN(uint64_t r, rng->Rand64());
                       r = r % modulus;

                   //     std::pair<uint64_t, uint64_t> masked_sender_strings(
                   //             (-r % modulus),
                   //             ModSub(round_four_state_this_party.branch_res_in_secret_shared_form[idx][i],r,modulus)
                   //             );

                           std::pair<uint64_t, uint64_t> masked_sender_strings(
                           ModSub(0,r,modulus),
                           ModSub(round_four_state_this_party.branch_res_in_secret_shared_form[idx][i],r,modulus)
                           );

                       std::pair<uint64_t, uint64_t> swapped_masked_sender_strings = swap(
                               masked_sender_strings,
                               round_four_state_this_party.mic_output_share[i][idx]
                               );

                       // XORing ROT sender masks with the sender strings
                       uint64_t swapped_masked_first_sender_string_xored_with_ROT_masks = ModAdd(
                               swapped_masked_sender_strings.first,
                               swapped_masks.first,
                               modulus
                               );

                       uint64_t swapped_masked_second_sender_string_xored_with_ROT_masks = ModAdd(
                               swapped_masked_sender_strings.second,
                               swapped_masks.second,
                               modulus
                       );

                       mux_randomness_i.push_back(r);
       //
       //                    round_five_sigmoid_state.mux_randomness[i][idx] = r;

                       mux_round_2_interval_i.add_ot_round_2_first_sender_msg(swapped_masked_first_sender_string_xored_with_ROT_masks);
                       mux_round_2_interval_i.add_ot_round_2_second_sender_msg(swapped_masked_second_sender_string_xored_with_ROT_masks);

                   }
                   round_five_sigmoid_state.mux_randomness.push_back(mux_randomness_i);


                   *(round_five_sigmoid_message.add_mux_round_2()) = mux_round_2_interval_i;

               }


               return std::pair<RoundFiveSigmoidNewMicState, RoundFiveSigmoidNewMicMessage>(
                       round_five_sigmoid_state,
                       round_five_sigmoid_message
               );
           }



           StatusOr<std::vector<uint64_t>> SecureSigmoidNewMic::GenerateSigmoidResult(
                   int partyid,
                   SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                   RoundFiveSigmoidNewMicState round_five_state_this_party,
                   RoundFiveSigmoidNewMicMessage round_five_msg_other_party) {

               uint64_t modulus = fixed_point_factory_->GetParams().primary_ring_modulus;

               // Vector is of length = batch-size
               std::vector<uint64_t> final_sigmoid_outputs_share;
               final_sigmoid_outputs_share.reserve(num_inputs_);

                           //
               for (int i = 0; i < num_inputs_; i++) {
                   std::vector<uint64_t> final_sigmoid_output_i_share;
                   final_sigmoid_output_i_share.reserve(6);


                   //First, compute the outputs of MUX for each of the 6 intervals for input i in the batch

                   for (int j = 0; j < 6; j++) {


                       // Compute OT outputs
                       uint64_t out_OT;

                       if (round_five_state_this_party.mic_output_share[j][i]) {
                           out_OT = ModSub(round_five_msg_other_party.mux_round_2(j).ot_round_2_second_sender_msg(i),
                                           sigmoid_precomputed_value.branching_precomp.rot_corr[j].receiver_msgs[i].receiver_msg,
                                           modulus);
                       } else {
                           out_OT = ModSub(round_five_msg_other_party.mux_round_2(j).ot_round_2_first_sender_msg(i),
                                           sigmoid_precomputed_value.branching_precomp.rot_corr[j].receiver_msgs[i].receiver_msg,
                                           modulus);
                       }

                       // Compute MUX outputs
                       uint64_t mux_output;

                       mux_output = ModAdd(out_OT, round_five_state_this_party.mux_randomness[j][i], modulus);

                   //     std::cout << "Party :" << partyid <<
                   //     " Interval : " << j << " : " <<
                   //     " Mux output share: " << mux_output << std::endl;


                       final_sigmoid_output_i_share.push_back(mux_output);

                   }


                   // Second, add the shares (lying in the vector of length = 6)
                   uint64_t final_sigmoid_output_i_share_combined = 0;

                   for (int j = 0; j < 6; j++) {
                       final_sigmoid_output_i_share_combined = ModAdd(
                               final_sigmoid_output_i_share_combined,
                               final_sigmoid_output_i_share[j],
                               modulus);
                   }

                   final_sigmoid_outputs_share.push_back(final_sigmoid_output_i_share_combined);
               }

               return final_sigmoid_outputs_share;
           }



        }  // namespace applications
    }// namespace private_join_and_compute