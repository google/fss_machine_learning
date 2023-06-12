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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <algorithm>

#include "applications/const_round_secure_comparison/const_round_secure_comparison.pb.h"
#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/numeric/int128.h"
#include "secret_sharing_mpc/gates/hadamard_product.h"
#include "dpf/distributed_point_function.h"
#include "fss_gates/equality.pb.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/status_macros.h"
#include "absl/memory/memory.h"


namespace private_join_and_compute {
    namespace applications {

        using ::private_join_and_compute::FixedPointElement;
        using ::private_join_and_compute::FixedPointElementFactory;

        absl::StatusOr<std::unique_ptr<ConstRoundSecureComparison>> ConstRoundSecureComparison::Create(
                uint64_t num_inputs, SecureComparisonParameters comparison_params){


            // If num_pieces (q) * piece length(m) != string_length (l), then extend the string
            uint64_t l, m, q;

            l = comparison_params.string_length;
            m = comparison_params.piece_length;
            q = comparison_params.num_pieces;

            // Initialize cmp and eq gates parameters
            distributed_point_functions::fss_gates::CmpParameters cmp_parameters;
            cmp_parameters.set_log_group_size(m + 1); // gate domain must be 1 bit larger 

            // Creating a Comparison gate
            DPF_ASSIGN_OR_RETURN(
                    std::unique_ptr<distributed_point_functions::fss_gates::ComparisonGate> CmpGate,
                    distributed_point_functions::fss_gates::ComparisonGate::Create(cmp_parameters));

            distributed_point_functions::fss_gates::EqParameters eq_parameters;
            eq_parameters.set_log_group_size(m); // Check domain size
//            eq_parameters.set_output_group_modulus(1);
//            uint64_t Q = 1ULL << q;
//            eq_parameters.set_output_group_modulus(Q);

            // Creating an Equality gate
            DPF_ASSIGN_OR_RETURN(
                    std::unique_ptr<distributed_point_functions::fss_gates::EqualityGate> EqGate,
                    distributed_point_functions::fss_gates::EqualityGate::Create(eq_parameters));

            // Create iDPF with log_group_size = q-1
            // Declaring the parameters for a incremental Distributed Point Function (DPF).

            // Create parameter vector for the incremental DPF.
            std::vector<distributed_point_functions::DpfParameters> idpf_parameters(q);

            // We are iterating from i = 0 to i = q - 1. When i = 0, a beta value will be encoded at the root
            // (which will not be used in our protocol). When i is between [1, q-1], a beta value will be encoded
            // at each level of th iDPF tree (which will be used in our protocol).
            for (int i = 0; i < static_cast<int>(idpf_parameters.size()); i++) {

                // Setting the `log_domain_size` of the i^th level of iDPF to be same as the i
                idpf_parameters[i].set_log_domain_size(i);

                // Setting the output ValueType of the DPF so that it can store 128 bit integers.
                *(idpf_parameters[i].mutable_value_type()) =
                        distributed_point_functions::ToValueType<absl::uint128>();
            }

//             Check that parameters are valid. We can use the DPF proto validator
//             directly.
            DPF_RETURN_IF_ERROR(
                    distributed_point_functions::dpf_internal::ProtoValidator::ValidateParameters(idpf_parameters));

            // Creating a DPF with appropriate parameters.
            DPF_ASSIGN_OR_RETURN(std::unique_ptr<distributed_point_functions::DistributedPointFunction> idpf,
                                 distributed_point_functions::DistributedPointFunction::CreateIncremental(idpf_parameters));


            return absl::WrapUnique(new ConstRoundSecureComparison(
                    num_inputs,
                    comparison_params,
                    std::move(EqGate),
                    std::move(CmpGate),
                    std::move(idpf)
                    ));
        }

        ConstRoundSecureComparison::ConstRoundSecureComparison(uint64_t num_inputs,
        SecureComparisonParameters comparison_params,
        std::unique_ptr<distributed_point_functions::fss_gates::EqualityGate> eq_gate,
        std::unique_ptr<distributed_point_functions::fss_gates::ComparisonGate> lt_gate,
        std::unique_ptr<distributed_point_functions::DistributedPointFunction> idpf):
        num_inputs_(num_inputs),
        comparison_params_(comparison_params),
        eq_gate_(std::move(eq_gate)),
        lt_gate_(std::move(lt_gate)),
        idpf_(std::move(idpf)) {}


        StatusOr<std::pair<SecureComparisonPrecomputedValue, SecureComparisonPrecomputedValue>>
        ConstRoundSecureComparison::PerformComparisonPrecomputation(){

            SecureComparisonPrecomputedValue seccomp_precomputation_party0, seccomp_precomputation_party1;

            // Generate input masks for short comparison and short equality

            const absl::string_view kSampleSeed = absl::string_view();
            ASSIGN_OR_RETURN(auto prng, private_join_and_compute::BasicRng::Create(kSampleSeed));

            uint64_t piece_modulus_lt = 1ULL << (comparison_params_.piece_length + 1);
            uint64_t piece_modulus_eq = 1ULL << comparison_params_.piece_length;
            uint64_t idpf_domain_modulus = 1ULL << (comparison_params_.num_pieces - 1);


            // Generating input masks for short comparison and short equality.
            for(int i = 0; i < num_inputs_; i++){

                // short comparison input masks
                ASSIGN_OR_RETURN(std::vector<uint64_t> input_masks_comparison_party0,
                                 SampleVectorFromPrng(comparison_params_.num_pieces,
                                                      piece_modulus_lt,
                                                      prng.get()));
                seccomp_precomputation_party0.short_comparison_input_masks.push_back(input_masks_comparison_party0);

                ASSIGN_OR_RETURN(std::vector<uint64_t> input_masks_comparison_party1,
                                 SampleVectorFromPrng(comparison_params_.num_pieces,
                                                      piece_modulus_lt,
                                                      prng.get()));
                seccomp_precomputation_party1.short_comparison_input_masks.push_back(input_masks_comparison_party1);


                // short equality input masks
                // Need one less because equality check of last piece is redundant.
                std::vector<uint64_t> input_masks_equality_party0, input_masks_equality_party1;

                if(comparison_params_.num_pieces > 1) {
                    ASSIGN_OR_RETURN(input_masks_equality_party0,
                                     SampleVectorFromPrng(comparison_params_.num_pieces - 1,
                                                          piece_modulus_eq,
                                                          prng.get()));


                    seccomp_precomputation_party0.short_equality_input_masks.push_back(input_masks_equality_party0);

                    ASSIGN_OR_RETURN(input_masks_equality_party1,
                                     SampleVectorFromPrng(comparison_params_.num_pieces - 1,
                                                          piece_modulus_eq,
                                                          prng.get()));

                    seccomp_precomputation_party1.short_equality_input_masks.push_back(input_masks_equality_party1);
                }

            }

            // Generate output mask shares for short comparison and equality
            for(int i = 0; i < num_inputs_; i++){
                ASSIGN_OR_RETURN(std::vector<uint64_t> ouput_masks_share_comparison_party0,
                                 SampleVectorFromPrng(comparison_params_.num_pieces,
                                                      2,
                                                      prng.get()));
                seccomp_precomputation_party0.short_comparison_output_masks_share.push_back(ouput_masks_share_comparison_party0);

                ASSIGN_OR_RETURN(std::vector<uint64_t> ouput_masks_share_comparison_party1,
                                 SampleVectorFromPrng(comparison_params_.num_pieces,
                                                      2,
                                                      prng.get()));
                seccomp_precomputation_party1.short_comparison_output_masks_share.push_back(ouput_masks_share_comparison_party1);

                // Need one less because equality check of last piece is redundant.
                if(comparison_params_.num_pieces > 1) {
                    ASSIGN_OR_RETURN(std::vector<uint64_t> ouput_masks_share_equality_party0,
                                     SampleVectorFromPrng(comparison_params_.num_pieces - 1,
                                                          2,
                                                          prng.get()));
                    seccomp_precomputation_party0.short_equality_output_masks_share.push_back(
                            ouput_masks_share_equality_party0);

                    ASSIGN_OR_RETURN(std::vector<uint64_t> ouput_masks_share_equality_party1,
                                     SampleVectorFromPrng(comparison_params_.num_pieces - 1,
                                                          2,
                                                          prng.get()));
                    seccomp_precomputation_party1.short_equality_output_masks_share.push_back(
                            ouput_masks_share_equality_party1);

                }

            }


            // Generate FSS keys for short comparison and short equality using input and output masks
            for(int i = 0; i < num_inputs_; i++){

                std::vector<distributed_point_functions::fss_gates::CmpKey> short_comp_keys_party0, short_comp_keys_party1;
                std::vector<distributed_point_functions::fss_gates::EqKey> short_eq_keys_party0, short_eq_keys_party1;

                for(int j = 0; j < comparison_params_.num_pieces; j++){

                    // Generating Comparison gate keys

                    uint64_t r_out = (seccomp_precomputation_party0.short_comparison_output_masks_share[i][j] +
                            seccomp_precomputation_party1.short_comparison_output_masks_share[i][j]) % 2;


                    distributed_point_functions::fss_gates::CmpKey cmp_key_0, cmp_key_1;
                    DPF_ASSIGN_OR_RETURN(std::tie(cmp_key_0, cmp_key_1),
                                         lt_gate_->Gen(seccomp_precomputation_party0.short_comparison_input_masks[i][j],
                                                      seccomp_precomputation_party1.short_comparison_input_masks[i][j],
                                                      r_out));

                    short_comp_keys_party0.push_back(cmp_key_0);
                    short_comp_keys_party1.push_back(cmp_key_1);
                    // Generating Equality gate keys

                    // Skipping the last piece for short equality
                    if(j < comparison_params_.num_pieces - 1) {
                        r_out = (seccomp_precomputation_party0.short_equality_output_masks_share[i][j] +
                                 seccomp_precomputation_party1.short_equality_output_masks_share[i][j]) % 2;

                        distributed_point_functions::fss_gates::EqKey eq_key_0, eq_key_1;

                        DPF_ASSIGN_OR_RETURN(std::tie(eq_key_0, eq_key_1),
                                             eq_gate_->Gen(
                                                     seccomp_precomputation_party0.short_equality_input_masks[i][j],
                                                     seccomp_precomputation_party1.short_equality_input_masks[i][j],
                                                     r_out));

                        short_eq_keys_party0.push_back(eq_key_0);
                        short_eq_keys_party1.push_back(eq_key_1);
                    }

                }

                seccomp_precomputation_party0.fss_comparison_keys.push_back(short_comp_keys_party0);
                seccomp_precomputation_party0.fss_equality_keys.push_back(short_eq_keys_party0);

                seccomp_precomputation_party1.fss_comparison_keys.push_back(short_comp_keys_party1);
                seccomp_precomputation_party1.fss_equality_keys.push_back(short_eq_keys_party1);

            }

            // Generate iDPF input mask
            std::vector<uint64_t> idpf_input_masks;

            ASSIGN_OR_RETURN(idpf_input_masks,
                             SampleVectorFromPrng(num_inputs_,
                                                  idpf_domain_modulus,
                                                  prng.get()));

            // Create bit sharings of the iDPF input masks
            for(int i = 0; i < num_inputs_; i++){

                std::vector<uint64_t> idpf_input_mask_for_ith_input_bitvector;

                for(int j = 0; j < (comparison_params_.num_pieces - 1); j++){
                    if((1ULL << (comparison_params_.num_pieces - 1 - 1 - j)) & idpf_input_masks[i]) {
                        idpf_input_mask_for_ith_input_bitvector.push_back(1);
                    }
                    else{
                        idpf_input_mask_for_ith_input_bitvector.push_back(0);
                    }

                }
                std::vector<uint64_t> idpf_mask_bit_shares_for_ith_input_party0,
                        idpf_mask_bit_shares_for_ith_input_party1;


                ASSIGN_OR_RETURN(idpf_mask_bit_shares_for_ith_input_party0,
                                 SampleVectorFromPrng((comparison_params_.num_pieces - 1),
                                                      2,
                                                      prng.get()));

                ASSIGN_OR_RETURN(idpf_mask_bit_shares_for_ith_input_party1,
                                 BatchedModSub(idpf_input_mask_for_ith_input_bitvector,
                                               idpf_mask_bit_shares_for_ith_input_party0,
                                               2));

                seccomp_precomputation_party0.idpf_mask_bit_shares.push_back(
                        idpf_mask_bit_shares_for_ith_input_party0);

                seccomp_precomputation_party1.idpf_mask_bit_shares.push_back(
                        idpf_mask_bit_shares_for_ith_input_party1);
            }

            // Generate iDPF keys
            for(int i = 0; i < num_inputs_; i++){
                uint64_t alpha = ((idpf_domain_modulus - 1) ^  idpf_input_masks[i]) % idpf_domain_modulus;

//                std::cout << "Alpha = " << alpha << std::endl;

//                std::cout << "IDPF input mask = " << idpf_input_masks[i] << std::endl;

                // Add a bogus beta value at the root.
                std::vector<absl::uint128> idpf_values(comparison_params_.num_pieces, absl::uint128(1));

                distributed_point_functions::DpfKey idpf_key_0, idpf_key_1;
                
                DPF_ASSIGN_OR_RETURN(std::tie(idpf_key_0, idpf_key_1),
                                     idpf_->GenerateKeysIncremental(alpha, idpf_values));

                seccomp_precomputation_party0.idpf_keys.push_back(idpf_key_0);
                seccomp_precomputation_party1.idpf_keys.push_back(idpf_key_1);
            }


            // Generate Beaver triples
            // Size of the input vectors in the Hadamard product only need to be q-1 instead of q.
            // For ease of implementation, currently we are doing hadamard product on q-length vectors.
            for(int i = 0; i < num_inputs_; i++) {
                ASSIGN_OR_RETURN(auto beaver_vector_shares,
                                 private_join_and_compute::SampleBeaverTripleVector(
                                         comparison_params_.num_pieces, 2));
                seccomp_precomputation_party0.beaver_vector_shares.push_back(beaver_vector_shares.first);
                seccomp_precomputation_party1.beaver_vector_shares.push_back(beaver_vector_shares.second);
            }

            return std::make_pair(std::move(seccomp_precomputation_party0), std::move(seccomp_precomputation_party1));
        }


        StatusOr<std::pair<RoundOneSecureComparisonState, RoundOneSecureComparisonMessage>>
        ConstRoundSecureComparison::GenerateComparisonRoundOneMessage(int partyid,
                                          SecureComparisonPrecomputedValue comparison_precomputed_value,
                                          std::vector<uint64_t> &comparison_private_inputs){

            uint64_t piece_modulus_lt = 1ULL << (comparison_params_.piece_length + 1);
            uint64_t piece_modulus_eq = 1ULL << comparison_params_.piece_length;

            // Displaying input masks
            /*
            std::cout << "Short LT Input masks for Party " << partyid << " : ";
            for(int j = 0; j < comparison_precomputed_value.short_comparison_input_masks[0].size(); j++)
                std::cout << comparison_precomputed_value.short_comparison_input_masks[0][j] << ", ";
            std::cout << std::endl;

            // Displaying input masks
            if(comparison_params_.num_pieces > 1) {
                std::cout << "Short EQ Input masks for Party " << partyid << " : ";
                for (int j = 0; j < comparison_precomputed_value.short_equality_input_masks[0].size(); j++)
                    std::cout << comparison_precomputed_value.short_equality_input_masks[0][j] << ", ";
                std::cout << std::endl;
            }

            // Displaying input masks
            std::cout << "Short LT Output mask share for Party " << partyid << " : ";
            for(int j = 0; j < comparison_precomputed_value.short_comparison_output_masks_share[0].size(); j++)
                std::cout << comparison_precomputed_value.short_comparison_output_masks_share[0][j] << ", ";
            std::cout << std::endl;

            // Displaying input masks
            if(comparison_params_.num_pieces > 1) {
                std::cout << "Short EQ Output mask share for Party " << partyid << " : ";
                for (int j = 0; j < comparison_precomputed_value.short_equality_output_masks_share[0].size(); j++)
                    std::cout << comparison_precomputed_value.short_equality_output_masks_share[0][j] << ", ";
                std::cout << std::endl;
            }
             */

            // Split the input into pieces
            std::vector<std::vector<uint64_t>> input_pieces;

            for(int i = 0; i < num_inputs_; i++){

                uint64_t private_input = comparison_private_inputs[i];

                std::vector<uint64_t> input_pieces_for_ith_input;

                // Extracting pieces sequentially from the LSB
                for(int j = 0; j < comparison_params_.num_pieces; j++){

                    uint64_t piece = private_input & (piece_modulus_eq - 1);

                    input_pieces_for_ith_input.push_back(piece);

                    private_input = private_input >> comparison_params_.piece_length;
                }

                std::reverse(input_pieces_for_ith_input.begin(), input_pieces_for_ith_input.end());


                // Displaying input pieces
//                std::cout << "Input pieces for Party " << partyid << " : ";
//                for(int j = 0; j < comparison_params_.num_pieces; j++)
//                    std::cout << input_pieces_for_ith_input[j] << ", ";
//                std::cout << std::endl;




                input_pieces.push_back(input_pieces_for_ith_input);
            }

            // Mask each of the input pieces

            RoundOneSecureComparisonState round_one_state;
            RoundOneSecureComparisonMessage round_one_msg;



            for(int i = 0; i < num_inputs_; i++) {

                std::vector<uint64_t> short_equality_round_one_state_for_ith_input,
                        short_comparison_round_one_state_for_ith_input;

                // Skipping last piece for short equality
                std::vector<uint64_t> input_pieces_last_removed(input_pieces[i].begin(), input_pieces[i].end() - 1);

                if(comparison_params_.num_pieces > 1) {

                    ASSIGN_OR_RETURN(
                            short_equality_round_one_state_for_ith_input,
                            private_join_and_compute::BatchedModAdd(input_pieces_last_removed,
                                                                    comparison_precomputed_value.short_equality_input_masks[i],
                                                                    piece_modulus_eq));
                }

                ASSIGN_OR_RETURN(
                        short_comparison_round_one_state_for_ith_input,
                        private_join_and_compute::BatchedModAdd(input_pieces[i],
                                                                comparison_precomputed_value.short_comparison_input_masks[i],
                                                                piece_modulus_lt));


                round_one_state.short_equality_masked_input_pieces.push_back(
                        short_equality_round_one_state_for_ith_input);

                round_one_state.short_comparison_masked_input_pieces.push_back(
                        short_comparison_round_one_state_for_ith_input);

                /*
                // Displaying masked input
                std::cout << "Short LT Masked Input for Party " << partyid << " : ";
                for(int j = 0; j < round_one_state.short_comparison_masked_input_pieces[0].size(); j++)
                    std::cout << round_one_state.short_comparison_masked_input_pieces[0][j] << ", ";
                std::cout << std::endl;

                std::cout << "Short EQ Masked Input for Party " << partyid << " : ";
                for(int j = 0; j < round_one_state.short_equality_masked_input_pieces[0].size(); j++)
                    std::cout << round_one_state.short_equality_masked_input_pieces[0][j] << ", ";
                std::cout << std::endl;

                 */

                // Fill up the round_one_msg
                SharesOfMaskedInputPieces masked_input_pieces_proto;

                // Copying the vector data into proto repeated field.
                // Source: https://stackoverflow.com/questions/15499641/copy-a-stdvector-to-a-repeated-field-from-protobuf-with-memcpy
                *masked_input_pieces_proto.mutable_shares_of_masked_input_pieces_for_eq() = {
                        short_equality_round_one_state_for_ith_input.begin(),
                        short_equality_round_one_state_for_ith_input.end()
                };

                *masked_input_pieces_proto.mutable_shares_of_masked_input_pieces_for_lt() = {
                        short_comparison_round_one_state_for_ith_input.begin(),
                        short_comparison_round_one_state_for_ith_input.end()
                };

                *(round_one_msg.add_shares_of_masked_input_pieces()) = masked_input_pieces_proto;

            }

            return std::pair<RoundOneSecureComparisonState, RoundOneSecureComparisonMessage>(
                    round_one_state, round_one_msg
                    );
        }

        StatusOr<std::vector<std::pair<RoundTwoSecureComparisonState, RoundTwoSecureComparisonMessage>>>
        ConstRoundSecureComparison::BatchGenerateComparisonRoundTwoMessage(
                int partyid,
                size_t batch_size,
                std::vector<SecureComparisonPrecomputedValue> &comparison_precomputed_values,
                std::vector<RoundOneSecureComparisonState> &round_one_state_this_party,
                std::vector<RoundOneSecureComparisonMessage> &round_one_msgs_other_party) {
            std::vector<std::vector<std::vector<uint64_t>>> shares_of_short_lt_output (7), shares_of_short_eq_output (7);

            std::vector<RoundTwoSecureComparisonMessage> roundTwoSecureComparisonMessage (7);


            // new with optimized dcf

            // Invoke short comparison and short equality gate on the masked input pieces to get (shares of) masked output.

            // Refactor SecureComparisonPrecomputedValue to avoid this
            std::vector<distributed_point_functions::fss_gates::CmpKey> fss_comparison_keys;
            fss_comparison_keys.reserve(7 * num_inputs_ * comparison_params_.num_pieces);
            for (size_t i = 0; i < 7; i++) {
                for (size_t j = 0; j < num_inputs_; j++) {
                    for (size_t k = 0; k < comparison_params_.num_pieces; k++) {
                        fss_comparison_keys.push_back(std::move(comparison_precomputed_values[i].fss_comparison_keys[j][k]));
                    }
                }
            }

            std::vector<std::uint64_t> short_lt_masked_party0_input_piece, short_lt_masked_party1_input_piece;
            for (size_t i = 0; i < 7; i++) {
                for (size_t j = 0; j < num_inputs_; j++) {
                    for (size_t k = 0; k < comparison_params_.num_pieces; k++) {
                        if (partyid == 0) {
                            short_lt_masked_party0_input_piece.push_back(
                                    round_one_state_this_party[i].short_comparison_masked_input_pieces[j][k]);
                            short_lt_masked_party1_input_piece.push_back(
                                    (round_one_msgs_other_party[i].shares_of_masked_input_pieces(
                                            j)).shares_of_masked_input_pieces_for_lt(k));

                        } else {
                            short_lt_masked_party1_input_piece.push_back(
                                    round_one_state_this_party[i].short_comparison_masked_input_pieces[j][k]);
                            short_lt_masked_party0_input_piece.push_back(
                                    (round_one_msgs_other_party[i].shares_of_masked_input_pieces(
                                            j)).shares_of_masked_input_pieces_for_lt(k));
                        }
                    }
                }
            }

            // batched version
            ASSIGN_OR_RETURN(std::vector<absl::uint128> res_lt_masked,
                             lt_gate_->BatchEval(partyid,
                                                 fss_comparison_keys,
                                                 short_lt_masked_party0_input_piece,
                                                 short_lt_masked_party1_input_piece));

            size_t offset_out = 0;
            for (size_t i = 0; i < 7; i++) {
                for (size_t j = 0; j < num_inputs_; j++) {
                    std::vector<uint64_t> shares_of_short_lt_output_for_ith_input;
                    shares_of_short_lt_output_for_ith_input.reserve(comparison_params_.num_pieces);
                    for (size_t k = 0; k < comparison_params_.num_pieces; k++) {
                        // Remove (shares of) masks from the (shares of) masked output to get (shares of) actual output.
                        std::uint64_t lt_output_modulus = 2;
                        std::uint64_t res_lt = ModSub(uint64_t(res_lt_masked[offset_out + k]),
                                                      comparison_precomputed_values[i].short_comparison_output_masks_share[j][k],
                                                      lt_output_modulus);

                        shares_of_short_lt_output_for_ith_input.push_back(res_lt);
                    }
                    shares_of_short_lt_output[i].push_back(std::move(shares_of_short_lt_output_for_ith_input));
                    offset_out += comparison_params_.num_pieces;
                }
            }

            for (size_t i = 0; i < 7; i++) {
                for (int j = 0; j < num_inputs_; j++) {
                    std::vector<uint64_t> shares_of_short_eq_output_for_ith_input;
                    shares_of_short_eq_output_for_ith_input.reserve(comparison_params_.num_pieces);

                    // Short equality
                    for (int k = 0; k < comparison_params_.num_pieces - 1; k++) {
                        std::uint64_t short_eq_masked_party0_input_piece, short_eq_masked_party1_input_piece;

                        if (partyid == 0) {
                            short_eq_masked_party0_input_piece = round_one_state_this_party[i].short_equality_masked_input_pieces[j][k];
                            short_eq_masked_party1_input_piece = (round_one_msgs_other_party[i].shares_of_masked_input_pieces(
                                    j)).shares_of_masked_input_pieces_for_eq(k);

                        } else {
                            short_eq_masked_party1_input_piece = round_one_state_this_party[i].short_equality_masked_input_pieces[j][k];
                            short_eq_masked_party0_input_piece = (round_one_msgs_other_party[i].shares_of_masked_input_pieces(
                                    j)).shares_of_masked_input_pieces_for_eq(k);
                        }

                        absl::uint128 res_eq_masked;

                        DPF_ASSIGN_OR_RETURN(res_eq_masked,
                                             eq_gate_->Eval(comparison_precomputed_values[i].fss_equality_keys[j][k],
                                                            short_eq_masked_party0_input_piece,
                                                            short_eq_masked_party1_input_piece));

                        std::uint64_t eq_output_modulus = 2;

                        std::uint64_t res_eq = ModSub(uint64_t(res_eq_masked),
                                                      comparison_precomputed_values[i].short_equality_output_masks_share[j][k],
                                                      eq_output_modulus);

                        shares_of_short_eq_output_for_ith_input.push_back(res_eq);
                    }
                    shares_of_short_eq_output[i].push_back(std::move(shares_of_short_eq_output_for_ith_input));
                }
            }

            uint64_t idpf_domain_modulus = 1ULL << (comparison_params_.num_pieces - 1);

            std::vector<std::vector<std::vector<uint64_t>>> masked_input_bit_shares_for_idpf (7);
            for (size_t i = 0; i < 7; i++) {
                for (int j = 0; j < num_inputs_; j++) {

                    ASSIGN_OR_RETURN(std::vector<uint64_t> masked_ith_input_bit_shares_for_idpf,
                                     BatchedModAdd(shares_of_short_eq_output[i][j],
                                                   comparison_precomputed_values[i].idpf_mask_bit_shares[j],
                                                   2));

                    masked_input_bit_shares_for_idpf[i].push_back(masked_ith_input_bit_shares_for_idpf);

                    BitSharingsOfMaskedIDPFInput bit_shares_of_masked_input_for_idpf_ith_input;

                    *bit_shares_of_masked_input_for_idpf_ith_input.mutable_bit_shares() = {
                            masked_ith_input_bit_shares_for_idpf.begin(),
                            masked_ith_input_bit_shares_for_idpf.end()
                    };

                    *(roundTwoSecureComparisonMessage[i].add_bit_shares_of_masked_input_for_idpf()) =
                            std::move(bit_shares_of_masked_input_for_idpf_ith_input);

                }
            }

            std::vector<RoundTwoSecureComparisonState> roundTwoSecureComparisonStates (7);
            for (size_t i = 0; i < 7; i++) {
                RoundTwoSecureComparisonState roundTwoSecureComparisonState{
                        std::move(shares_of_short_lt_output[i]),
                        std::move(masked_input_bit_shares_for_idpf[i])
                };
                roundTwoSecureComparisonStates[i] = std::move(roundTwoSecureComparisonState);
            }

            std::vector<std::pair<RoundTwoSecureComparisonState, RoundTwoSecureComparisonMessage>> roundTwoStatesMessages;
            roundTwoStatesMessages.reserve(7);
            for (size_t i = 0; i < 7; i++) {
                roundTwoStatesMessages.push_back(std::pair<RoundTwoSecureComparisonState, RoundTwoSecureComparisonMessage>(
                        std::move(roundTwoSecureComparisonStates[i]),
                        std::move(roundTwoSecureComparisonMessage[i])
                ));
            }

            return roundTwoStatesMessages;
        }

        StatusOr<std::pair<RoundTwoSecureComparisonState, RoundTwoSecureComparisonMessage>>
        ConstRoundSecureComparison::GenerateComparisonRoundTwoMessage(
                int partyid,
                SecureComparisonPrecomputedValue comparison_precomputed_value,
                RoundOneSecureComparisonState round_one_state_this_party,
                RoundOneSecureComparisonMessage round_one_msg_other_party){

            std::vector<std::vector<uint64_t>> shares_of_short_lt_output, shares_of_short_eq_output;
            shares_of_short_lt_output.reserve(num_inputs_);
            shares_of_short_eq_output.reserve(num_inputs_);

            RoundTwoSecureComparisonMessage roundTwoSecureComparisonMessage;


            // new with optimized dcf

            // Invoke short comparison and short equality gate on the masked input pieces to get (shares of) masked output.

            // Refactor SecureComparisonPrecomputedValue to avoid this
            std::vector<distributed_point_functions::fss_gates::CmpKey> fss_comparison_keys;
            fss_comparison_keys.reserve(num_inputs_ * comparison_params_.num_pieces);
            for (size_t i = 0; i < num_inputs_; i++) {
                for (size_t j = 0; j < comparison_params_.num_pieces; j++) {
                    fss_comparison_keys.push_back(std::move(comparison_precomputed_value.fss_comparison_keys[i][j]));
                }
            }

            std::vector<std::uint64_t> short_lt_masked_party0_input_piece, short_lt_masked_party1_input_piece;
            short_lt_masked_party0_input_piece.reserve(num_inputs_ * comparison_params_.num_pieces);
            short_lt_masked_party1_input_piece.reserve(num_inputs_ * comparison_params_.num_pieces);
            for (size_t i = 0; i < num_inputs_; i++) {
                for (size_t j = 0; j < comparison_params_.num_pieces; j++) {
                    if (partyid == 0) {
                        short_lt_masked_party0_input_piece.push_back(round_one_state_this_party.short_comparison_masked_input_pieces[i][j]);
                        short_lt_masked_party1_input_piece.push_back((round_one_msg_other_party.shares_of_masked_input_pieces(
                                i)).shares_of_masked_input_pieces_for_lt(j));

                    } else {
                        short_lt_masked_party1_input_piece.push_back(round_one_state_this_party.short_comparison_masked_input_pieces[i][j]);
                        short_lt_masked_party0_input_piece.push_back((round_one_msg_other_party.shares_of_masked_input_pieces(
                                i)).shares_of_masked_input_pieces_for_lt(j));
                    }
                }
            }

            // call eval one by one version (much slower)
            /*std::vector<absl::uint128> res_lt_masked;
            size_t offset_eval = 0;
            for (size_t i = 0; i < num_inputs_; i++) {
                for (size_t j = 0; j < comparison_params_.num_pieces; j++) {
                    absl::uint128 res_lt_masked_temp;

                    DPF_ASSIGN_OR_RETURN(res_lt_masked_temp,
                                         lt_gate_->Eval(partyid,
                                                        fss_comparison_keys[offset_eval + j],
                                                        short_lt_masked_party0_input_piece[offset_eval + j],
                                                        short_lt_masked_party1_input_piece[offset_eval + j]));
                    res_lt_masked.push_back(res_lt_masked_temp);
                }
                offset_eval += comparison_params_.num_pieces;
            }*/

            // batched version
            ASSIGN_OR_RETURN(std::vector<absl::uint128> res_lt_masked,
                             lt_gate_->BatchEval(partyid,
                                                 fss_comparison_keys,
                                                 short_lt_masked_party0_input_piece,
                                                 short_lt_masked_party1_input_piece));

            size_t offset_out = 0;
            for (size_t i = 0; i < num_inputs_; i++) {
                std::vector<uint64_t> shares_of_short_lt_output_for_ith_input;
                shares_of_short_lt_output_for_ith_input.reserve(comparison_params_.num_pieces);
                for (size_t j = 0; j < comparison_params_.num_pieces; j++) {
                    // Remove (shares of) masks from the (shares of) masked output to get (shares of) actual output.
                    std::uint64_t lt_output_modulus = 2;
                    std::uint64_t res_lt = ModSub(uint64_t(res_lt_masked[offset_out + j]),
                                                  comparison_precomputed_value.short_comparison_output_masks_share[i][j],
                                                  lt_output_modulus);

                    shares_of_short_lt_output_for_ith_input.push_back(res_lt);
                }
                shares_of_short_lt_output.push_back(std::move(shares_of_short_lt_output_for_ith_input));
                offset_out += comparison_params_.num_pieces;
            }

            for (int i = 0; i < num_inputs_; i++) {
                std::vector<uint64_t> shares_of_short_eq_output_for_ith_input;
                shares_of_short_eq_output_for_ith_input.reserve(comparison_params_.num_pieces);

                // Short equality
                for(int j = 0; j < comparison_params_.num_pieces - 1; j++){
                    std::uint64_t short_eq_masked_party0_input_piece, short_eq_masked_party1_input_piece;

                    if(partyid == 0){
                        short_eq_masked_party0_input_piece = round_one_state_this_party.short_equality_masked_input_pieces[i][j];
                        short_eq_masked_party1_input_piece = (round_one_msg_other_party.shares_of_masked_input_pieces(i)).shares_of_masked_input_pieces_for_eq(j);

                    } else{
                        short_eq_masked_party1_input_piece = round_one_state_this_party.short_equality_masked_input_pieces[i][j];
                        short_eq_masked_party0_input_piece = (round_one_msg_other_party.shares_of_masked_input_pieces(i)).shares_of_masked_input_pieces_for_eq(j);
                    }

                    absl::uint128 res_eq_masked;

                    DPF_ASSIGN_OR_RETURN(res_eq_masked,
                                         eq_gate_->Eval(comparison_precomputed_value.fss_equality_keys[i][j],
                                                        short_eq_masked_party0_input_piece,
                                                        short_eq_masked_party1_input_piece));

                    // Remove (shares of) masks from the (shares of) masked output to get (shares of) actual output.
//                    std::uint64_t eq_output_modulus = 1ULL << (comparison_params_.num_pieces - 1);
                    std::uint64_t eq_output_modulus = 2;

                    std::uint64_t res_eq = ModSub(uint64_t(res_eq_masked),
                                                  comparison_precomputed_value.short_equality_output_masks_share[i][j],
                                                  eq_output_modulus);

                    shares_of_short_eq_output_for_ith_input.push_back(res_eq);

                }

                shares_of_short_eq_output.push_back(std::move(shares_of_short_eq_output_for_ith_input));



                // Displaying short EQ output
//                std::cout << "Short EQ Output share for Party " << partyid << " : ";
//                for(int j = 0; j < shares_of_short_eq_output_for_ith_input.size(); j++)
//                    std::cout << shares_of_short_eq_output_for_ith_input[j] << ", ";
//                std::cout << std::endl;



            }


            // Old (before batched optimization)

//            // Invoke short comparison and short equality gate on the masked input pieces to get (shares of) masked output.
//            for (int i = 0; i < num_inputs_; i++) {
//
//                std::vector<uint64_t> shares_of_short_lt_output_for_ith_input, shares_of_short_eq_output_for_ith_input;
//
//
//               // std::cout << "Masked LT input piece Party " << partyid << " : ";
//
//               // Short comparison
//               for (int j = 0; j < comparison_params_.num_pieces; j++){
//                    std::uint64_t short_lt_masked_party0_input_piece, short_lt_masked_party1_input_piece;
//
//                    if (partyid == 0){
//                        short_lt_masked_party0_input_piece = round_one_state_this_party.short_comparison_masked_input_pieces[i][j];
//                        short_lt_masked_party1_input_piece = (round_one_msg_other_party.shares_of_masked_input_pieces(i)).shares_of_masked_input_pieces_for_lt(j);
//
//                    } else{
//                        short_lt_masked_party1_input_piece = round_one_state_this_party.short_comparison_masked_input_pieces[i][j];
//                        short_lt_masked_party0_input_piece = (round_one_msg_other_party.shares_of_masked_input_pieces(i)).shares_of_masked_input_pieces_for_lt(j);
//                    }
//
//               //     std::cout << "(" << short_lt_masked_party0_input_piece << ", "
//                 //   << short_lt_masked_party1_input_piece << ") ";
//
//                    absl::uint128 res_lt_masked;
//
//
//                    DPF_ASSIGN_OR_RETURN(res_lt_masked,
//                                         lt_gate_->Eval(partyid,
//                                                        comparison_precomputed_value.fss_comparison_keys[i][j],
//                                                        short_lt_masked_party0_input_piece,
//                                                        short_lt_masked_party1_input_piece));
//
//                    // Remove (shares of) masks from the (shares of) masked output to get (shares of) actual output.
//                    std::uint64_t lt_output_modulus = 2;
//                    std::uint64_t res_lt = ModSub(uint64_t(res_lt_masked),
//                                                  comparison_precomputed_value.short_comparison_output_masks_share[i][j],
//                                                  lt_output_modulus);
//
//                    shares_of_short_lt_output_for_ith_input.push_back(res_lt);
//                }
//
//              //  std::cout << "\n";
//
//
//              /*
//
//                // Displaying short LT output
//                std::cout << "Short LT Output share for Party " << partyid << " : ";
//                for(int j = 0; j < shares_of_short_lt_output_for_ith_input.size(); j++)
//                    std::cout << shares_of_short_lt_output_for_ith_input[j] << ", ";
//                std::cout << std::endl;
//
//               */
//
//
//
//                // Short equality
//                for(int j = 0; j < comparison_params_.num_pieces - 1; j++){
//                    std::uint64_t short_eq_masked_party0_input_piece, short_eq_masked_party1_input_piece;
//
//                    if(partyid == 0){
//                        short_eq_masked_party0_input_piece = round_one_state_this_party.short_equality_masked_input_pieces[i][j];
//                        short_eq_masked_party1_input_piece = (round_one_msg_other_party.shares_of_masked_input_pieces(i)).shares_of_masked_input_pieces_for_eq(j);
//
//                    }else{
//                        short_eq_masked_party1_input_piece = round_one_state_this_party.short_equality_masked_input_pieces[i][j];
//                        short_eq_masked_party0_input_piece = (round_one_msg_other_party.shares_of_masked_input_pieces(i)).shares_of_masked_input_pieces_for_eq(j);
//                    }
//
//                    absl::uint128 res_eq_masked;
//
//                    DPF_ASSIGN_OR_RETURN(res_eq_masked,
//                                         eq_gate_->Eval(comparison_precomputed_value.fss_equality_keys[i][j],
//                                                        short_eq_masked_party0_input_piece,
//                                                        short_eq_masked_party1_input_piece));
//
//                    // Remove (shares of) masks from the (shares of) masked output to get (shares of) actual output.
////                    std::uint64_t eq_output_modulus = 1ULL << (comparison_params_.num_pieces - 1);
//                    std::uint64_t eq_output_modulus = 2;
//
//                    std::uint64_t res_eq = ModSub(uint64_t(res_eq_masked),
//                                                  comparison_precomputed_value.short_equality_output_masks_share[i][j],
//                                                  eq_output_modulus);
//
//                    shares_of_short_eq_output_for_ith_input.push_back(res_eq);
//
//                }
//
//                shares_of_short_lt_output.push_back(shares_of_short_lt_output_for_ith_input);
//                shares_of_short_eq_output.push_back(shares_of_short_eq_output_for_ith_input);
//
//
//
//                // Displaying short EQ output
////                std::cout << "Short EQ Output share for Party " << partyid << " : ";
////                for(int j = 0; j < shares_of_short_eq_output_for_ith_input.size(); j++)
////                    std::cout << shares_of_short_eq_output_for_ith_input[j] << ", ";
////                std::cout << std::endl;
//
//
//
//            }

            uint64_t idpf_domain_modulus = 1ULL << (comparison_params_.num_pieces - 1);

            /*

            // Convert the bit representation of the (shares of) outputs of the short equality gate into
            //  (shares of) an integer.
            std::vector<uint64_t> eq_integer;
            for(int i = 0; i < num_inputs_; i++){
                uint64_t eq_integer_for_ith_input = 0;

                // for short equality, we only have pieces - 1 iterations
                for(int j = 0; j < comparison_params_.num_pieces - 1; j++){

                    uint64_t temp = ModMul(shares_of_short_eq_output[i][j],
                                           1ULL << ((comparison_params_.num_pieces - 1) - (j + 1)),
                                           idpf_domain_modulus);

                    eq_integer_for_ith_input = ModAdd(temp ,
                                                      eq_integer_for_ith_input,
                                                      idpf_domain_modulus);
                }

                // Displaying short EQ integer output share
                std::cout << "Short EQ Integer Output share for Party " << partyid << " : ";
                std::cout << eq_integer_for_ith_input << ", ";
                std::cout << std::endl;

                eq_integer.push_back(eq_integer_for_ith_input);
            }






            // Create (shares of) the masked integer output of short equality gates.
            std::vector<uint64_t> eq_integer_masked;

            ASSIGN_OR_RETURN(eq_integer_masked, BatchedModAdd(
                    eq_integer,
                    comparison_precomputed_value.idpf_mask_share,
                    idpf_domain_modulus
                    ));

             */
            std::vector<std::vector<uint64_t>> masked_input_bit_shares_for_idpf;
            masked_input_bit_shares_for_idpf.reserve(num_inputs_);
            for(int i = 0; i < num_inputs_; i++){

                ASSIGN_OR_RETURN(std::vector<uint64_t> masked_ith_input_bit_shares_for_idpf,
                                 BatchedModAdd(shares_of_short_eq_output[i],
                                               comparison_precomputed_value.idpf_mask_bit_shares[i],
                                               2));

                masked_input_bit_shares_for_idpf.push_back(masked_ith_input_bit_shares_for_idpf);

                BitSharingsOfMaskedIDPFInput bit_shares_of_masked_input_for_idpf_ith_input;

                *bit_shares_of_masked_input_for_idpf_ith_input.mutable_bit_shares() = {
                        masked_ith_input_bit_shares_for_idpf.begin(),
                        masked_ith_input_bit_shares_for_idpf.end()
                };

                *(roundTwoSecureComparisonMessage.add_bit_shares_of_masked_input_for_idpf()) =
                        std::move(bit_shares_of_masked_input_for_idpf_ith_input);

                // Displaying masked iDPF input share
//                std::cout << "Masked iDPF input share for Party " << partyid << " : ";
//                for(int j = 0; j < masked_ith_input_bit_shares_for_idpf.size(); j++)
//                    std::cout << masked_ith_input_bit_shares_for_idpf[j] << ", ";
//                std::cout << std::endl;

            }

            RoundTwoSecureComparisonState roundTwoSecureComparisonState{
                shares_of_short_lt_output,
                masked_input_bit_shares_for_idpf
            };


            return std::pair<RoundTwoSecureComparisonState, RoundTwoSecureComparisonMessage>(
                    std::move(roundTwoSecureComparisonState),
                    std::move(roundTwoSecureComparisonMessage)
                    );
        }


        StatusOr<std::pair<RoundThreeSecureComparisonState, RoundThreeSecureComparisonMessage>>
        ConstRoundSecureComparison::GenerateComparisonRoundThreeMessage(
                int partyid,
                SecureComparisonPrecomputedValue comparison_precomputed_value,
                RoundTwoSecureComparisonState round_two_state_this_party,
                RoundTwoSecureComparisonMessage round_two_msg_other_party){


            RoundThreeSecureComparisonState roundThreeSecureComparisonState;
            RoundThreeSecureComparisonMessage roundThreeSecureComparisonMessage;

            // Reconstruct the masked iDPF input
            std::vector<uint64_t> masked_idpf_input;
            masked_idpf_input.reserve(num_inputs_);

            uint64_t idpf_domain_modulus = 1ULL << (comparison_params_.num_pieces - 1);

            for(int i = 0; i < num_inputs_; i++){
                std::uint64_t masked_idpf_ith_input = 0;
                for(int j = 0; j < (comparison_params_.num_pieces - 1); j++){

                    std::uint64_t masked_idpf_ith_input_jth_bit = ModAdd(
                            round_two_state_this_party.masked_input_bit_shares_for_idpf[i][j],
                            (round_two_msg_other_party.bit_shares_of_masked_input_for_idpf(i)).bit_shares(j),
                            2);

                    std::uint64_t temp = ModMul((1ULL << (comparison_params_.num_pieces - 1 - 1 - j)),
                                                masked_idpf_ith_input_jth_bit,
                                                idpf_domain_modulus);

//                    std::cout << "masked_idpf_ith_input_j = << " << j << "_bit for Party " << partyid << " : " <<
//                    masked_idpf_ith_input_jth_bit << std::endl;

                    masked_idpf_ith_input = ModAdd(masked_idpf_ith_input,
                                                   temp,
                                                   idpf_domain_modulus);


                }

                masked_idpf_input.push_back(masked_idpf_ith_input);
            }

//            ASSIGN_OR_RETURN(masked_idpf_input, BatchedModAdd(
//                    round_two_state_this_party.masked_shares_of_eq_integer,
//                    std::vector<uint64_t> (round_two_msg_other_party.shares_of_masked_input_for_idpf().begin(),
//                                           round_two_msg_other_party.shares_of_masked_input_for_idpf().end()),
//                                           idpf_domain_modulus
//                    ));

//            std::cout << "Masked idpf input for Party " << partyid << " : " << masked_idpf_input[0] << std::endl;

            // Invoke iDPF at every level on the masked integer output of the short equality gate to get shares
            // of prefix AND

            std::vector<std::vector<uint64_t>> shares_of_prefixAND_of_eq;
            shares_of_prefixAND_of_eq.reserve(num_inputs_);

//            using ValueType = absl::uint128;
            for(int i = 0; i < num_inputs_; i++){

                std::vector<uint64_t> shares_of_prefixAND_of_eq_for_ith_input;
                shares_of_prefixAND_of_eq_for_ith_input.reserve(comparison_params_.num_pieces);

                // Adding a (share of) 1 at the beginning of the vector so that it is later compatible with the dot product logic.
                shares_of_prefixAND_of_eq_for_ith_input.push_back(partyid);

                // Evaluate iDPF on empty prefix to obtain the beta at the root. This is just bogus output and
                // will not be used. It is just being done in order to be compliant with the incremental evaluation
                // of the iDPF.
                DPF_ASSIGN_OR_RETURN(std::vector<absl::uint128> idpf_res,
                                     idpf_->EvaluateAt<absl::uint128>(
                                             comparison_precomputed_value.idpf_keys[i],
                                             0,
                                             {}
                                             ));

                for(int j = 0; j < comparison_params_.num_pieces - 1; j++){

                    uint64_t hierarchy_level = j + 1;

                    DPF_ASSIGN_OR_RETURN(std::vector<absl::uint128> idpf_res,
                                         idpf_->EvaluateAt<absl::uint128>(
                                                 comparison_precomputed_value.idpf_keys[i],
                                                 hierarchy_level,
                                                 {masked_idpf_input[i] >> (comparison_params_.num_pieces - 1 - (j + 1))}
                                         ));

                    shares_of_prefixAND_of_eq_for_ith_input.push_back(uint64_t(idpf_res[0]) % 2);

                }

                // Displaying prefix AND output
//                std::cout << "PREFIX AND share for Party " << partyid << " : ";
//                for(int j = 0; j < shares_of_prefixAND_of_eq_for_ith_input.size(); j++)
//                    std::cout << shares_of_prefixAND_of_eq_for_ith_input[j] << ", ";
//                std::cout << std::endl;

                shares_of_prefixAND_of_eq.push_back(shares_of_prefixAND_of_eq_for_ith_input);

            }

            roundThreeSecureComparisonState.shares_of_prefixAND_of_eq = shares_of_prefixAND_of_eq;

            // Invoke Hadamard product on the (shares of) prefix AND outputs and (shares of) short comparison outputs

            // Size of the input vectors in the Hadamard product only need to be q-1 instead of q.
            // For ease of implementation, currently we are doing hadamard product on q-length vectors.

            for(int i = 0; i < num_inputs_; i++){
                std::pair<BatchedMultState, MultiplicationGateMessage> hadamard_state_plus_msg;

                ASSIGN_OR_RETURN(hadamard_state_plus_msg,
                        GenerateHadamardProductMessage(
                                shares_of_prefixAND_of_eq[i],
                                round_two_state_this_party.shares_of_lt[i],
                                comparison_precomputed_value.beaver_vector_shares[i],
                                2));

                roundThreeSecureComparisonState.hadamard_state.push_back(hadamard_state_plus_msg.first);

                *(roundThreeSecureComparisonMessage.add_hadamard_message_for_branching()) =
                        hadamard_state_plus_msg.second;
            }


            return std::pair<RoundThreeSecureComparisonState, RoundThreeSecureComparisonMessage>(
                    std::move(roundThreeSecureComparisonState),
                    std::move(roundThreeSecureComparisonMessage)
                    );
        }

        StatusOr<std::vector<uint64_t>> ConstRoundSecureComparison::GenerateComparisonResult(
                int partyid,
                SecureComparisonPrecomputedValue comparison_precomputed_value,
                RoundThreeSecureComparisonState round_three_state_this_party,
                RoundThreeSecureComparisonMessage round_three_msg_other_party){

            std::vector<uint64_t> comparison_result_share;
            comparison_result_share.reserve(num_inputs_);


            // Do the final beaver triple computation
            for(int i = 0; i < num_inputs_; i++){
                // First, compute shares of the hadamard product result

                std::vector<uint64_t> hadamard_result_share;

                if (partyid == 0) {
                    // Execute HadamardProductPartyZero.
                    ASSIGN_OR_RETURN(
                            hadamard_result_share,
                            private_join_and_compute::HadamardProductPartyZero(
                                    round_three_state_this_party.hadamard_state[i],
                                    comparison_precomputed_value.beaver_vector_shares[i],
                                    round_three_msg_other_party.hadamard_message_for_branching(i),
                                    0,
                                    2));
                } else {
                    // Execute HadamardProductPartyOne.
                    ASSIGN_OR_RETURN(
                            hadamard_result_share,
                            private_join_and_compute::HadamardProductPartyOne(
                                    round_three_state_this_party.hadamard_state[i],
                                    comparison_precomputed_value.beaver_vector_shares[i],
                                    round_three_msg_other_party.hadamard_message_for_branching(i),
                                    0,
                                    2));
                }

                // For computing shares of the dot-product, we need to add up the elements in the hadamard product
                // output.
                uint64_t dot_product_share = 0;
                for(int j = 0; j < comparison_params_.num_pieces; j++){
                    dot_product_share = ModAdd(
                            dot_product_share,
                            hadamard_result_share[j],
                            2);
                }

                comparison_result_share.push_back(dot_product_share);

            }

            return comparison_result_share;
        }
        

    }  // namespace applications
}  // namespace private_join_and_compute
