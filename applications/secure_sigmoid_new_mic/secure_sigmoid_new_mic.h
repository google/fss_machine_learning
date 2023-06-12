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

#ifndef GOOGLE_CODE_SECURE_SIGMOID_NEW_MIC_H
#define GOOGLE_CODE_SECURE_SIGMOID_NEW_MIC_H

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
#include "absl/numeric/int128.h"
#include "dcf/fss_gates/multiple_interval_containment.h"
#include "applications/secure_spline/secure_spline.h"
#include "applications/secure_comparison/secure_comparison.h"
#include "secret_sharing_mpc/gates/polynomial.h"
#include "dcf/fss_gates/multiple_interval_containment.h"
#include "dcf/fss_gates/multiple_interval_containment.pb.h"
#include "secret_sharing_mpc/gates/vector_exponentiation.h"
#include "applications/const_round_secure_comparison/const_round_secure_comparison.h"

namespace private_join_and_compute {
    namespace applications {

// Steps:

// In our actual application, invoke sigmoid on a bunch of inputs
// [x_1, .... x_n].

// Input : x

// Invoke MIC gate on x for the following intervals: (on the Ring : 0 - 2^l )
// Interval 1 : [0, 2^lf)
// Interval 2 : [2^lf, 2^lf. loge 2^lf)
// Interval 3 : [2^lf. loge 2^lf, 2^(l-1) )
// Interval 4 : [2^(l-1), 2^l - 2^lf. loge 2^lf)
// Interval 5 : [2^l - 2^lf. loge 2^lf, 2^l - 2^lf)
// Interval 6 : [2^l - 2^lf, 2^l)

// [0, 0, 1, 0, 0, 0]

// [b_1, b_2, b_3, b_4, b_5, b_6]



// We will compute sigmoid on x for each Interval i using Algorithm i
// Algorithm 1 : Invoke Secure_Spline on x
// Algorithm 2 : Invoke Poisson exponentiation and then tailor approximation on -x
// Algorithm 3 : Output 1
// Algorithm 4 : Output 1 - Algorithm_3 (-x)
// Algorithm 5 : Output 1 - Algorithm_2 (-x)
// Algorithm 6 : Output 1 - Algorithm_1 (-x)

// [res_1, res_2, res_3, res_4, res_5, res_6]


// Final output : Dot product of [b_1, b_2, b_3, b_4, b_5, b_6] with [res_1, res_2, res_3, res_4, res_5, res_6]


// Preprocessing:
// 1. Secure Comparison gate : Generate secure comparison preprocessing and call SampleBeaverTripleVector
// 2. Secure Spline
// 3. Exponentiation : SampleMultToAddSharesVector in vector_exponentiation.h
// 4. Polynomial : PolynomialSamplePowersOfRandomVector and PolynomialPreprocessRandomOTs in polynomial.h


        struct SecureSigmoidNewMicParameters{
            // Input and output bit size of the ring that sigmoid will operate on.
            uint64_t log_group_size;

            private_join_and_compute::applications::SecureSplineParameters spline_params;

            // Number of bits of precision needed for fractional values.
            uint64_t num_fractional_bits;

            // Degree of taylor series polynomial approximation.
            uint64_t taylor_polynomial_degree;

            // Parameters needed for the exponentiation protocol.
            private_join_and_compute::ExponentiationParams exp_params;

            // Parameters for secure comparison
            uint64_t block_length;
            uint64_t num_splits;
        };

// Each BranchingPrecomputedValueNewMic struct holds preprocessing information needed to evaluate
// all the branches of sigmoid flowchart in parallel and then combine the result. This is done
// using 6 secure comparison gate (to execute the result of conditionals) and hadamard product (to retain the
// results of the relevant branches)
        struct BranchingPrecomputedValueNewMic {

            // Secure comparison preprocessing for each of the 7 bounds.
            std::vector<private_join_and_compute::applications::SecureComparisonPrecomputedValue>
            comparison_preprocess;


            // Using OT to perform the MUX instead of hadamard product
            // Vector of size 7
            std::vector<private_join_and_compute::PolynomialRandomOTPrecomputation> rot_corr;

            // Add bit beaver triples to perform AND gate - beavertriplevector is of
            // size = n i.e. batch size. Outer vector is of size = 6
            std::vector<private_join_and_compute::BeaverTripleVector<uint64_t>> hadamard_triple;

        };

// Each SigmoidPrecomputedValueNewMic struct holds preprocessing information needed to evaluate
// sigmoid securely on batch of inputs
        struct SigmoidPrecomputedValueNewMic {

            // ########################     Branching

            BranchingPrecomputedValueNewMic branching_precomp;


            // ########################     Spline gate

            std::vector<private_join_and_compute::applications::SplinePrecomputedValue> spline_precomp_pos;

            std::vector<private_join_and_compute::applications::SplinePrecomputedValue> spline_precomp_neg;


            // ########################     Exponentiation gate

            private_join_and_compute::MultToAddShare mta_pos;

            private_join_and_compute::MultToAddShare mta_neg;


            // ########################     Polynomial gate

            std::vector<std::vector<uint64_t>> powers_of_random_vector_pos;

            std::vector<std::vector<uint64_t>> powers_of_random_vector_neg;

            private_join_and_compute::PolynomialRandomOTPrecomputation rot_corr_pos;

            private_join_and_compute::PolynomialRandomOTPrecomputation rot_corr_neg;

            // Constants for Algorithm 3 and Algorithm 4
            std::vector<uint64_t> share_of_one, share_of_zero;


        };

        struct  RoundOneSigmoidNewMicState{

            // Secret share of actual input
            std::vector<uint64_t> shares_of_sigmoid_inputs;

            // Secure comparison round 1 state for each of the 6 intervals
            std::vector<RoundOneSecureComparisonState> secure_comparison_round_one_state;

            std::vector<std::vector<uint64_t>> first_bit_share;


            RoundOneSplineState round_one_spline_state_pos;

            RoundOneSplineState round_one_spline_state_neg;

            // TODO: secure_exponentiation.h contains 2 different state structs one for each party. But
            // we want a single to handle both of them in a single RoundOneSigmoidNewMicState struct.
            // Dirty temporary fix : Include both state structs in defining RoundOneSigmoidNewMicMessage and ensure that
            // only one (the relevant one depending on the party) is populated
            std::vector<private_join_and_compute::SecureExponentiationPartyZero::State> round_one_exp_state_party0_pos;

            std::vector<private_join_and_compute::SecureExponentiationPartyZero::State> round_one_exp_state_party0_neg;

            std::vector<private_join_and_compute::SecureExponentiationPartyOne::State> round_one_exp_state_party1_pos;

            std::vector<private_join_and_compute::SecureExponentiationPartyOne::State> round_one_exp_state_party1_neg;


        };

        struct  RoundTwoSigmoidNewMicState{


            RoundTwoSplineState round_two_spline_state_pos;

            RoundTwoSplineState round_two_spline_state_neg;

            PowersStateRoundOne round_one_polynomial_state_pos;

            PowersStateRoundOne round_one_polynomial_state_neg;

            //exp result state
            std::vector<FixedPointElement> exp_result_state_pos;

            std::vector<FixedPointElement> exp_result_state_neg;

            //Secure comparison Round 2 state for each of the 6 intervals
            std::vector<RoundTwoSecureComparisonState> secure_comparison_round_two_state;

            std::vector<std::vector<uint64_t>> first_bit_share;

        };

        struct  RoundThreeSigmoidNewMicState{

            std::vector<uint64_t> spline_res_in_secret_shared_form_pos;

            std::vector<uint64_t> spline_res_in_secret_shared_form_neg;

            PowersStateRoundOne round_one_polynomial_state_pos;

            PowersStateRoundOne round_one_polynomial_state_neg;

            PolynomialShareOfPolynomialShare round_two_polynomial_state_pos;

            PolynomialShareOfPolynomialShare round_two_polynomial_state_neg;

            //Secure comparison Round 3 state for each of the 6 intervals
            std::vector<RoundThreeSecureComparisonState> secure_comparison_round_three_state;

//            std::vector<private_join_and_compute::secure_comparison::ComparisonShortComparisonEquality>
//                    combination_input;

            std::vector<std::vector<uint64_t>> first_bit_share;

        };

        struct RoundThreePointFiveSigmoidNewMicState{
                // Outer vector is over batch of inputs, inner vector is over 6 intervals
            std::vector<std::vector<uint64_t>> branch_res_in_secret_shared_form;

            // Outer vector is over 7 intervals, inner vector is over batch of inputs - n
            std::vector<std::vector<uint64_t>> comparison_output_share;

            // Outer vector is over 7 intervals, inner vector is over batch of inputs - n
            std::vector<std::vector<uint64_t>> negated_comparison_output_share;


            // Add hadamard state
            std::vector<BatchedMultState> hadamard_state_for_ANDing_comparison_results;

        };

        struct  RoundFourSigmoidNewMicState{

            // Outer vector is over batch of inputs, inner vector is over 6 intervals
            std::vector<std::vector<uint64_t>> branch_res_in_secret_shared_form;

            // Outer vector is over 6 intervals, inner vector is over batch of inputs - n
        // Interval containment result shares
            std::vector<std::vector<uint64_t>> mic_output_share;


        };


        // Fill this up
        struct RoundFiveSigmoidNewMicState{

            // Outer vector is over 6 intervals, inner vector is over batch of inputs - n
            // Carryover from previous state to be consumed later in Final Result computation

            std::vector<std::vector<uint64_t>> mic_output_share;

            // Outer vector is over 6 intervals, inner vector is over batch of inputs - n
            std::vector<std::vector<uint64_t>> mux_randomness;

        };

        class SecureSigmoidNewMic {

        private:

            // A pointer to a Secure Spline gate.
            std::unique_ptr<SecureSpline> secure_spline_;

            // A pointer to a Secure Comparison gate.
            std::unique_ptr<ConstRoundSecureComparison> seccomp_;

            // Number of inputs in the batch.
            const uint64_t num_inputs_;


            // Private constructor, called by `Create`.
            SecureSigmoidNewMic(uint64_t num_inputs,
                          std::unique_ptr<ConstRoundSecureComparison> seccomp,
                          std::vector<uint64_t> intervals_uint64,
                          std::unique_ptr<SecureSpline> secure_spline,
                          std::unique_ptr<FixedPointElementFactory> fixed_point_factory,
                          SecureSigmoidNewMicParameters sigmoid_params,
                          std::unique_ptr<SecureExponentiationPartyZero> exp_party_zero,
                          std::unique_ptr<SecureExponentiationPartyOne> exp_party_one);

            // SecureSigmoid is neither copyable nor movable.
            SecureSigmoidNewMic(const SecureSigmoidNewMic &) = delete;

            SecureSigmoidNewMic &operator=(const SecureSigmoidNewMic &) = delete;

        public:


            //Intervals
        //     std::vector<uint64_t> intervals_lower_bound_uint64, intervals_upper_bound_uint64;
            std::vector<uint64_t> intervals_uint64;

            // Parameters for sigmoid specification.
            const SecureSigmoidNewMicParameters sigmoid_params_;

            // Pointer to parties for exponentiation protocol. Ideally, we should have
            // a single pointer but this is not possible with the way exponentiation
            // protocol is currently implemented.
            std::unique_ptr<SecureExponentiationPartyZero> exp_party_zero_;
            std::unique_ptr<SecureExponentiationPartyOne> exp_party_one_;


            // Pointer to Fixed Point factory for converting between double and
            // fixed point representation.
            std::unique_ptr<FixedPointElementFactory> fixed_point_factory_;

            // Factory method : creates and returns a SecureSigmoidNewMic initialized with
            // appropriate parameters.
            static absl::StatusOr<std::unique_ptr<SecureSigmoidNewMic>> Create(
                    uint64_t num_inputs, SecureSigmoidNewMicParameters sigmoid_params);

            // Performs precomputation stage of sigmoid and returns a pair of
            // n SigmoidPrecomputedValues - one for each party for each input x_i.
            StatusOr<std::pair<SigmoidPrecomputedValueNewMic, SigmoidPrecomputedValueNewMic>>
            PerformSigmoidPrecomputation();

            StatusOr<std::pair<RoundOneSigmoidNewMicState, RoundOneSigmoidNewMicMessage>>
            GenerateSigmoidRoundOneMessage(int partyid,
                                           SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                                           std::vector<uint64_t> &share_of_sigmoid_inputs);

            StatusOr<std::pair<RoundTwoSigmoidNewMicState, RoundTwoSigmoidNewMicMessage>>
            GenerateSigmoidRoundTwoMessage(
                    int partyid,
                    SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                    RoundOneSigmoidNewMicState round_one_state_this_party,
                    RoundOneSigmoidNewMicMessage round_one_msg_other_party);


            StatusOr<std::pair<RoundThreeSigmoidNewMicState, RoundThreeSigmoidNewMicMessage>>
            GenerateSigmoidRoundThreeMessage(
                    int partyid,
                    SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                    RoundTwoSigmoidNewMicState round_two_state_this_party,
                    RoundTwoSigmoidNewMicMessage round_two_msg_other_party);

                StatusOr<std::pair<RoundThreePointFiveSigmoidNewMicState, RoundThreePointFiveSigmoidNewMicMessage>>
        GenerateSigmoidRoundThreePointFiveMessage(
                int partyid,
                SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                RoundThreeSigmoidNewMicState round_three_state_this_party,
                RoundThreeSigmoidNewMicMessage round_three_msg_other_party);


            StatusOr<std::pair<RoundFourSigmoidNewMicState, RoundFourSigmoidNewMicMessage>>
            GenerateSigmoidRoundFourMessage(
                    int partyid,
                    SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                    RoundThreePointFiveSigmoidNewMicState round_three_point_five_state_this_party,
                    RoundThreePointFiveSigmoidNewMicMessage round_three_point_five_msg_other_party);

            StatusOr<std::pair<RoundFiveSigmoidNewMicState, RoundFiveSigmoidNewMicMessage>>
            GenerateSigmoidRoundFiveMessage(
                    int partyid,
                    SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                    RoundFourSigmoidNewMicState round_four_state_this_party,
                    RoundFourSigmoidNewMicMessage round_four_msg_other_party);


            StatusOr<std::vector<uint64_t>> GenerateSigmoidResult(
                    int partyid,
                    SigmoidPrecomputedValueNewMic &sigmoid_precomputed_value,
                    RoundFiveSigmoidNewMicState round_five_state_this_party,
                    RoundFiveSigmoidNewMicMessage round_five_msg_other_party);
        };
    }  // namespace applications
}  // namespace private_join_and_compute


#endif //GOOGLE_CODE_SECURE_SIGMOID_NEW_MIC_H


