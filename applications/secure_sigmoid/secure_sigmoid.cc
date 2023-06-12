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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "applications/secure_sigmoid/secure_sigmoid.pb.h"
#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/numeric/int128.h"
#include "dcf/fss_gates/multiple_interval_containment.h"
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
        using ::distributed_point_functions::fss_gates::MultipleIntervalContainmentGate;

        absl::StatusOr<std::unique_ptr<SecureSigmoid>> SecureSigmoid::Create(
                uint64_t num_inputs, SecureSigmoidParameters sigmoid_params){

            // Setting up fxp factory


            ASSIGN_OR_RETURN(
                    FixedPointElementFactory fixed_point_factory,
                    FixedPointElementFactory::Create(sigmoid_params.spline_params.num_fractional_bits,
                                                     sigmoid_params.spline_params.log_group_size));



            // Setting up MIC gate

            distributed_point_functions::fss_gates::MicParameters mic_parameters;

            // Setting input and output group
            mic_parameters.set_log_group_size(sigmoid_params.spline_params.log_group_size);


            // Hardcoding the different interval values
            const uint64_t two_raised_to_lf =
                    fixed_point_factory.GetParams().fractional_multiplier;
            const uint64_t two_raised_to_l =
                    fixed_point_factory.GetParams().primary_ring_modulus;
            const uint64_t lf = fixed_point_factory.GetParams().num_fractional_bits;




            // TODO : Check for possible rounding errors in setting the intervals

            // Natural log of 2
            const double ln2 = 0.69314718055994530941;

            const std::vector<double> positive_lower_bounds{
                    0.0, 1.0, lf * ln2};

            const std::vector<double> positive_upper_bounds = {
                    1.0, lf * ln2, two_raised_to_l / (2.0 * two_raised_to_lf) - 1.0};



            // Converting the MIC parameters into fixed point representation
            std::vector<FixedPointElement> lower_bounds_fxp;
            std::vector<FixedPointElement> upper_bounds_fxp;
            lower_bounds_fxp.reserve(3);
            upper_bounds_fxp.reserve(3);



            for (int i = 0; i < 3; ++i) {
                ASSIGN_OR_RETURN(FixedPointElement lb,
                                 fixed_point_factory.CreateFixedPointElementFromDouble(
                                         positive_lower_bounds[i]));



                ASSIGN_OR_RETURN(FixedPointElement ub,
                                 fixed_point_factory.CreateFixedPointElementFromDouble(
                                         positive_upper_bounds[i]));



                lower_bounds_fxp.push_back(lb);
                upper_bounds_fxp.push_back(ub);
            }



            // Adding positive intervals
            for (int i = 0; i < 3; i++){
                distributed_point_functions::fss_gates::Interval* interval =
                        mic_parameters.add_intervals();
                interval->mutable_lower_bound()->mutable_value_uint128()->set_low(
                        lower_bounds_fxp[i].ExportToUint64());
                interval->mutable_upper_bound()->mutable_value_uint128()->set_low(
                        upper_bounds_fxp[i].ExportToUint64() - 1);
            }

            // Adding negative intervals - Need to switch the ordering of upper and lower bound
            for (int i = 2; i >= 0; i--){
                distributed_point_functions::fss_gates::Interval* interval =
                        mic_parameters.add_intervals();
                interval->mutable_lower_bound()->mutable_value_uint128()->set_low(
                        two_raised_to_l - upper_bounds_fxp[i].ExportToUint64());
                interval->mutable_upper_bound()->mutable_value_uint128()->set_low(
                        two_raised_to_l - lower_bounds_fxp[i].ExportToUint64() - 1);
            }



            // Creating a MIC gate
            DPF_ASSIGN_OR_RETURN(std::unique_ptr<MultipleIntervalContainmentGate> MicGate,
                    MultipleIntervalContainmentGate::Create(mic_parameters));



            // Creating a Spline gate
            ASSIGN_OR_RETURN(std::unique_ptr<SecureSpline> secure_spline,
                                 SecureSpline::Create(num_inputs, sigmoid_params.spline_params));


            // Create the protocol parties for the exponentiation protocol.

            // TODO: Check the first parameter in kSampleLargeExpParams


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


            return absl::WrapUnique(new SecureSigmoid(
                    num_inputs,
                    std::move(MicGate),
                    std::move(secure_spline),
                    absl::make_unique<FixedPointElementFactory>(fixed_point_factory),
                    sigmoid_params,
                    std::move(temp_zero),
                    std::move(temp_one)));
        }


        SecureSigmoid::SecureSigmoid(
                uint64_t num_inputs,
                std::unique_ptr<MultipleIntervalContainmentGate> MicGate,
                std::unique_ptr<SecureSpline> secure_spline,
                std::unique_ptr<FixedPointElementFactory> fixed_point_factory,
                SecureSigmoidParameters sigmoid_params,
                std::unique_ptr<SecureExponentiationPartyZero> exp_party_zero,
                std::unique_ptr<SecureExponentiationPartyOne> exp_party_one)
                : num_inputs_(num_inputs),
                mic_gate_(std::move(MicGate)),
                secure_spline_(std::move(secure_spline)),
                fixed_point_factory_(std::move(fixed_point_factory)),
                sigmoid_params_(sigmoid_params),
                exp_party_zero_(std::move(exp_party_zero)),
                exp_party_one_(std::move(exp_party_one))
                {}


        absl::StatusOr<std::pair<SigmoidPrecomputedValue,SigmoidPrecomputedValue>>
        SecureSigmoid::PerformSigmoidPrecomputation()  {

            SigmoidPrecomputedValue sigmoid_precomputation_party_0;
            SigmoidPrecomputedValue sigmoid_precomputation_party_1;

            sigmoid_precomputation_party_0.branching_precomp.reserve(num_inputs_);
            sigmoid_precomputation_party_1.branching_precomp.reserve(num_inputs_);

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


            // Generate preprocessing for branching
            for (int i = 0; i < num_inputs_; i++) {
                distributed_point_functions::fss_gates::MicKey key_0, key_1;

                uint64_t r_in;

                // Initializing the input and output masks uniformly at random;
                ASSIGN_OR_RETURN(r_in, rng->Rand64());
                r_in = r_in % modulus;

                std::vector<absl::uint128> r_outs;
                std::vector<uint64_t> r_outs_uint64;

                for (int j = 0; j < 6; j++) {
                    ASSIGN_OR_RETURN(uint64_t r_out, rng->Rand64());
                    r_out = r_out % modulus;
                    r_outs.push_back(r_out);
                    r_outs_uint64.push_back(r_out);
                }

                // Generating MIC gate keys
                DPF_ASSIGN_OR_RETURN(std::tie(key_0, key_1), mic_gate_->Gen(r_in, r_outs));

                // Generate Beaver triple vector for P0 and P1.
                ASSIGN_OR_RETURN(auto beaver_vector_shares,
                                 private_join_and_compute::SampleBeaverTripleVector(
                                         6, modulus));
                auto beaver_vector_share_0 = beaver_vector_shares.first;
                auto beaver_vector_share_1 = beaver_vector_shares.second;

                ASSIGN_OR_RETURN(uint64_t mic_input_mask_share_party_0, rng->Rand64());
                mic_input_mask_share_party_0 = mic_input_mask_share_party_0 % modulus;
                uint64_t mic_input_mask_share_party_1 =
                        private_join_and_compute::ModSub(r_in, mic_input_mask_share_party_0, modulus);

                // Generate random shares of input and output masks
                ASSIGN_OR_RETURN(auto mic_output_mask_share_party_0,
                                 SampleVectorFromPrng(6,
                                                      modulus, rng.get()));
                ASSIGN_OR_RETURN(
                        auto mic_output_mask_share_party_1,
                        private_join_and_compute::BatchedModSub(r_outs_uint64, mic_output_mask_share_party_0,
                                                                modulus));

                BranchingPrecomputedValue branching_party_0{key_0, beaver_vector_share_0,
                                               mic_input_mask_share_party_0,
                                               mic_output_mask_share_party_0};

                BranchingPrecomputedValue branching_party_1{key_1, beaver_vector_share_1,
                                               mic_input_mask_share_party_1,
                                               mic_output_mask_share_party_1};

                sigmoid_precomputation_party_0.branching_precomp.push_back(branching_party_0);
                sigmoid_precomputation_party_1.branching_precomp.push_back(branching_party_1);

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


        StatusOr<std::pair<RoundOneSigmoidState, RoundOneSigmoidMessage>>
        SecureSigmoid::GenerateSigmoidRoundOneMessage(int partyid,
                SigmoidPrecomputedValue sigmoid_precomputed_value,
                std::vector<uint64_t> &share_of_sigmoid_inputs){

            RoundOneSigmoidState round_one_sigmoid_state;
            RoundOneSigmoidMessage round_one_sigmoid_message;

//            std::cout << "GenerateSigmoidRoundOneMessage() Party 0 exp prime : "
//                      << exp_party_zero_->exp_params_->prime_q << std::endl;

            round_one_sigmoid_state.shares_of_sigmoid_inputs = share_of_sigmoid_inputs;

           // Computing branching related next-message-function

            uint64_t modulus = fixed_point_factory_->GetParams().primary_ring_modulus;

            for (int i = 0; i < num_inputs_; i++) {
                // uint64 share_of_spline_input_uint64 =
                // share_of_spline_inputs[i].ExportToUint64();
                uint64_t share_of_masked_input_for_branching = private_join_and_compute::ModAdd(
                        share_of_sigmoid_inputs[i],
                        sigmoid_precomputed_value.branching_precomp[i].mic_input_mask_share, modulus);

                round_one_sigmoid_state.shares_of_masked_input_for_branching.push_back(
                        share_of_masked_input_for_branching);

                round_one_sigmoid_message.add_shares_of_masked_input_for_branching(
                        share_of_masked_input_for_branching);
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

            for(int i = 0; i < num_inputs_; i++){

                ASSIGN_OR_RETURN(FixedPointElement fpe_neg,
                                     fixed_point_factory_->ImportFixedPointElementFromUint64(
                                             share_of_sigmoid_inputs_neg[i]));
                share_of_sigmoid_inputs_neg_fxp.push_back(fpe_neg);

                ASSIGN_OR_RETURN(FixedPointElement fpe_pos,
                                 fixed_point_factory_->ImportFixedPointElementFromUint64(
                                         share_of_sigmoid_inputs[i]));
                share_of_sigmoid_inputs_pos_fxp.push_back(fpe_pos);

//                std::cout << "Party " << partyid << " Index " << i << " : " <<
//                fpe.ExportToUint64() << std::endl;
            }

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

//                *(round_one_sigmoid_message.round_one_exp_message_union().
//                mutable_round_one_exp_message_party0()) = round_one_exp_party0_pos.second;

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

//                round_one_sigmoid_message.round_one_exp_message().
//                        set_round_one_exp_message_party1(round_one_exp_message_party1);

//                *(round_one_sigmoid_message.round_one_exp_message().
//                        mutable_round_one_exp_message_party1()) = round_one_exp_party1_pos.second;

                *(round_one_sigmoid_message.mutable_round_one_exp_message_party1_pos()) =
                        round_one_exp_party1_pos.first;

                *(round_one_sigmoid_message.mutable_round_one_exp_message_party1_neg()) =
                        round_one_exp_party1_neg.first;
                }


            return std::pair<RoundOneSigmoidState, RoundOneSigmoidMessage>(
                    round_one_sigmoid_state, round_one_sigmoid_message);
        }


        StatusOr<std::pair<RoundTwoSigmoidState, RoundTwoSigmoidMessage>>
        SecureSigmoid::GenerateSigmoidRoundTwoMessage(
                int partyid,
                SigmoidPrecomputedValue sigmoid_precomputed_value,
                RoundOneSigmoidState round_one_state_this_party,
                RoundOneSigmoidMessage round_one_msg_other_party){

            RoundTwoSigmoidState round_two_sigmoid_state;
            RoundTwoSigmoidMessage round_two_sigmoid_message;

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



            // Computing MIC gate output in secret shared form and store it in
            // RoundTwoSigmoidState

            // Reconstruct the actual masked input from the shares;
            std::vector<absl::uint128> masked_inputs_for_mic_gate;
            masked_inputs_for_mic_gate.reserve(num_inputs_);

            for (int i = 0; i < num_inputs_; i++) {
                uint64_t masked_input = private_join_and_compute::ModAdd(
                        round_one_state_this_party.shares_of_masked_input_for_branching[i],
                        round_one_msg_other_party.shares_of_masked_input_for_branching(i),
                        modulus);
                masked_inputs_for_mic_gate.push_back(masked_input);
            }

            // Invoke MIC.Eval on the masked inputs to get secret shares of the masked
            // output. Note that the outer vector indexes into one out of the n different
            // outputs (corresponding to each input) whereas the inner vector indexes into
            // one out of the m different outputs (corresponding to each interval for a
            // fixed input). Therefore the outer vector is of length n and the inner
            // vector is of length m.
            std::vector<std::vector<uint64_t>> shares_of_masked_outputs_for_mic_gate;
            shares_of_masked_outputs_for_mic_gate.reserve(num_inputs_);

            // new (optimized dcf)

            // refactor sigmoid precomputed value to avoid this unnecessary copy
            std::vector<const distributed_point_functions::fss_gates::MicKey *> keys(num_inputs_);
            for (size_t idx = 0; idx < num_inputs_; idx++) {
                keys[idx] = &sigmoid_precomputed_value.branching_precomp[idx].mic_key;
            }

            // call eval one by one version (much slower)
            /*std::vector<absl::uint128> share_of_masked_output;
            for (size_t i = 0; i < num_inputs_; i++) {
                DPF_ASSIGN_OR_RETURN(std::vector<absl::uint128> share_of_masked_output_temp,
                                     mic_gate_->Eval(keys[i],
                                                     masked_inputs_for_mic_gate[i]));
                for (size_t j = 0; j < 6; j++) {
                    share_of_masked_output.push_back(share_of_masked_output_temp[j]);
                }
            }*/

            // batched version
            DPF_ASSIGN_OR_RETURN(std::vector<absl::uint128> share_of_masked_output,
                                   mic_gate_->BatchEval(keys, masked_inputs_for_mic_gate));

            size_t offset = 0;
            for (size_t i = 0; i < num_inputs_; i++) {
                // Static casting uint128 (the output type of MIC gate) into uint64 (the
                // input type of Secret Sharing MPC codebase).
                std::vector<uint64_t> share_of_masked_output_uint64;
                share_of_masked_output_uint64.reserve(6);
                for (int j = 0; j < 6; j++)
                    share_of_masked_output_uint64.push_back(
                            static_cast<uint64_t>(share_of_masked_output[offset + j]));

                shares_of_masked_outputs_for_mic_gate.push_back(share_of_masked_output_uint64);
                offset += 6;
            }

            // old (unoptimized dcf)
//            for (int i = 0; i < num_inputs_; i++) {
//                DPF_ASSIGN_OR_RETURN(std::vector<absl::uint128> share_of_masked_output,
//                        mic_gate_->Eval(sigmoid_precomputed_value.branching_precomp[i].mic_key,
//                                        masked_inputs_for_mic_gate[i]));
//
//                // Static casting uint128 (the output type of MIC gate) into uint64 (the
//                // input type of Secret Sharing MPC codebase).
//                std::vector<uint64_t> share_of_masked_output_uint64;
//                share_of_masked_output_uint64.reserve(6);
//                for (int j = 0; j < 6; j++)
//                    share_of_masked_output_uint64.push_back(
//                            static_cast<uint64_t>(share_of_masked_output[j]));
//
//                shares_of_masked_outputs_for_mic_gate.push_back(share_of_masked_output_uint64);
//            }

            // Converting shares of masked output to shares of actual output.
            std::vector<std::vector<uint64_t>> shares_of_actual_outputs_for_mic_gate;
            shares_of_actual_outputs_for_mic_gate.reserve(num_inputs_);

            for (int i = 0; i < num_inputs_; i++) {
                ASSIGN_OR_RETURN(
                        std::vector<uint64_t> shares_of_actual_output,
                        private_join_and_compute::BatchedModSub(
                                shares_of_masked_outputs_for_mic_gate[i],
                                sigmoid_precomputed_value.branching_precomp[i].mic_output_mask_shares,
                                modulus));
                shares_of_actual_outputs_for_mic_gate.push_back(shares_of_actual_output);
            }

            round_two_sigmoid_state.mic_gate_result_share = shares_of_actual_outputs_for_mic_gate;


            return std::pair<RoundTwoSigmoidState, RoundTwoSigmoidMessage>(
                    round_two_sigmoid_state,
                    round_two_sigmoid_message
            );

        }

        StatusOr<std::pair<RoundThreeSigmoidState, RoundThreeSigmoidMessage>>
        SecureSigmoid::GenerateSigmoidRoundThreeMessage(
                int partyid,
                SigmoidPrecomputedValue sigmoid_precomputed_value,
                RoundTwoSigmoidState round_two_state_this_party,
                RoundTwoSigmoidMessage round_two_msg_other_party){

            RoundThreeSigmoidState round_three_sigmoid_state;
            RoundThreeSigmoidMessage round_three_sigmoid_message;

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

            // TODO : Define coefficients based on taylor_polynomial_degree

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

            // Carrying forward the information about MIC gate results from
            // Round 2 state to Round 3 state
            round_three_sigmoid_state.mic_gate_result_share =
                    round_two_state_this_party.mic_gate_result_share;

            // Carrying forward the information about Round 1 poly state
            round_three_sigmoid_state.round_one_polynomial_state_pos =
                    round_two_state_this_party.round_one_polynomial_state_pos;

            round_three_sigmoid_state.round_one_polynomial_state_neg =
                    round_two_state_this_party.round_one_polynomial_state_neg;


            return std::pair<RoundThreeSigmoidState, RoundThreeSigmoidMessage>(
                    round_three_sigmoid_state,
                    round_three_sigmoid_message
                    );
        }


        StatusOr<std::pair<RoundFourSigmoidState, RoundFourSigmoidMessage>>
        SecureSigmoid::GenerateSigmoidRoundFourMessage(
                int partyid,
                SigmoidPrecomputedValue sigmoid_precomputed_value,
                RoundThreeSigmoidState round_three_state_this_party,
                RoundThreeSigmoidMessage round_three_msg_other_party){

            RoundFourSigmoidState round_four_sigmoid_state;
            RoundFourSigmoidMessage round_four_sigmoid_message;

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


            // Second, create a vector of length 6 containing the output of
            // MIC gate on each of the 6 intervals for input x :
            // branch_conditional_in_secret_shared_form = [ _, _, _, _, _, _ ]
            // Again, since we are in the batched setting, we will actually need
            // a vector of vectors

            // This information is already stored in
            // RoundThreeSigmoidState.mic_gate_result_share.
            //
            // But we need to convert that
            // from int to fxp format because branch_res_in_secret_shared_form is in
            // fxp format. The following lines are copied from Line 401 - Line 414 in
            // secure_spline.cc


            // Convert shares of actual output of spline into shares of
            // output in FixedPoint representation (by multiplying the share value
            // locally by 2^lf).
            std::vector<std::vector<uint64_t>> mic_gate_result_share_in_fxp;
            mic_gate_result_share_in_fxp.reserve(num_inputs_);

            std::vector<uint64_t> fractional_multiplier_vector(
                    6,
                    fixed_point_factory_->GetParams().fractional_multiplier);

            for (int i = 0; i < num_inputs_; i++) {
                ASSIGN_OR_RETURN(
                        std::vector<uint64_t> mic_gate_result_share_in_fxp_i,
                        private_join_and_compute::BatchedModMul(
                                round_three_state_this_party.mic_gate_result_share[i],
                                fractional_multiplier_vector,
                                modulus));
                mic_gate_result_share_in_fxp.push_back(
                        mic_gate_result_share_in_fxp_i);
            }


            // Third, for each of the input x, compute a hadamard product
            // between branch_res_in_secret_shared_form and
            // mic_gate_result_share_in_fxp
            // (using SigmoidPrecomputedValue.BranchingPrecomputedValue.hadamard_triple)
            // and store the state in RoundFourSigmoidState.hadamard_state_for_branching
            // and message in RoundFourSigmoidMessage.hadamard_message_for_branching

            for(int i = 0; i < num_inputs_; i++){

                // Hadamard product between
                // branch_res_in_secret_shared_form[i] and
                // mic_gate_result_share_in_fxp[i]

                // Each party generates its batched multiplication message.
                std::pair<private_join_and_compute::BatchedMultState,
                private_join_and_compute::MultiplicationGateMessage>
                        hadamard_state_plus_msg;

                ASSIGN_OR_RETURN(
                        hadamard_state_plus_msg,
                        GenerateHadamardProductMessage(
                                branch_res_in_secret_shared_form[i],
                                mic_gate_result_share_in_fxp[i],
                                sigmoid_precomputed_value.branching_precomp[i].hadamard_triple,
                                modulus));


                round_four_sigmoid_state.hadamard_state_for_branching.push_back(
                        hadamard_state_plus_msg.first);

                *(round_four_sigmoid_message.add_hadamard_message_for_branching()) =
                        hadamard_state_plus_msg.second;

            }

            return std::pair<RoundFourSigmoidState, RoundFourSigmoidMessage>(
                    round_four_sigmoid_state,
                    round_four_sigmoid_message
                    );
        }


        StatusOr<std::vector<uint64_t>> SecureSigmoid::GenerateSigmoidResult(
                int partyid,
                SigmoidPrecomputedValue sigmoid_precomputed_value,
                RoundFourSigmoidState round_four_state_this_party,
                RoundFourSigmoidMessage round_four_msg_other_party){

            uint64_t modulus = fixed_point_factory_->GetParams().primary_ring_modulus;

            std::vector<uint64_t> final_sigmoid_outputs_share;
            final_sigmoid_outputs_share.reserve(num_inputs_);

            for (int i = 0; i < num_inputs_; i++) {
                std::vector<uint64_t> final_sigmoid_output_i_share;
                final_sigmoid_output_i_share.reserve(6);

                // First, compute shares of the hadamard product result
                if (partyid == 0) {
                    // Execute HadamardProductPartyZero.
                    ASSIGN_OR_RETURN(
                            final_sigmoid_output_i_share,
                            private_join_and_compute::HadamardProductPartyZero(
                                    round_four_state_this_party.hadamard_state_for_branching[i],
                                    sigmoid_precomputed_value.branching_precomp[i].hadamard_triple,
                                    round_four_msg_other_party.hadamard_message_for_branching(i),
                                    fixed_point_factory_->GetParams().num_fractional_bits,
                                    fixed_point_factory_->GetParams().primary_ring_modulus));
                } else {
                    // Execute HadamardProductPartyOne.
                    ASSIGN_OR_RETURN(
                            final_sigmoid_output_i_share,
                            private_join_and_compute::HadamardProductPartyOne(
                                    round_four_state_this_party.hadamard_state_for_branching[i],
                                    sigmoid_precomputed_value.branching_precomp[i].hadamard_triple,
                                    round_four_msg_other_party.hadamard_message_for_branching(i),
                                    fixed_point_factory_->GetParams().num_fractional_bits,
                                    fixed_point_factory_->GetParams().primary_ring_modulus));
                }



                // Second, add the shares (lying in the vector of length = 6)
                uint64_t final_sigmoid_output_i_share_combined = 0;

                for(int i = 0; i < 6; i++){
                    final_sigmoid_output_i_share_combined = ModAdd(
                            final_sigmoid_output_i_share_combined,
                            final_sigmoid_output_i_share[i],
                            modulus);
                }

                final_sigmoid_outputs_share.push_back(final_sigmoid_output_i_share_combined);

            }

            return final_sigmoid_outputs_share;
        }


    }  // namespace applications
}  // namespace private_join_and_compute