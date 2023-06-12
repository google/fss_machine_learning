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

#ifndef GOOGLE_CODE_SECURE_SIGMOID_H
#define GOOGLE_CODE_SECURE_SIGMOID_H

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
#include "absl/numeric/int128.h"
#include "dcf/fss_gates/multiple_interval_containment.h"
#include "applications/secure_spline/secure_spline.h"
#include "secret_sharing_mpc/gates/polynomial.h"
#include "dcf/fss_gates/multiple_interval_containment.h"
#include "dcf/fss_gates/multiple_interval_containment.pb.h"
#include "secret_sharing_mpc/gates/vector_exponentiation.h"

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
// 1. MIC gate : Generate random input and output masks for MIC gate and call MIC.Gen,
// SampleBeaverTripleVector
// 2. Secure Spline
// 3. Exponentiation : SampleMultToAddSharesVector in vector_exponentiation.h
// 4. Polynomial : PolynomialSamplePowersOfRandomVector and PolynomialPreprocessRandomOTs in polynomial.h


struct SecureSigmoidParameters{
    // Input and output bit size of the ring that sigmoid will operate on.
    int log_group_size;

    private_join_and_compute::applications::SecureSplineParameters spline_params;

    // Number of bits of precision needed for fractional values.
    uint64_t num_fractional_bits;

    // Degree of taylor series polynomial approximation.
    uint64_t taylor_polynomial_degree;

    // Parameters needed for the exponentiation protocol.
    private_join_and_compute::ExponentiationParams exp_params;
};

// Each BranchingPrecomputedValue struct holds preprocessing information needed to evaluate
// all the branches of sigmoid flowchart in parallel and then combine the result. This is done
// using MIC gate (to execute the result of conditionals) and hadamard product (to retain the
// results of the relevant branches)
struct BranchingPrecomputedValue{

    // A key for evaluating Multiple Interval Containment Gate.
    distributed_point_functions::fss_gates::MicKey mic_key;

    // Beaver triples for executing the hadamard product.
    private_join_and_compute::BeaverTripleVector<uint64_t> hadamard_triple;

    // Secret share of input mask.
    uint64_t mic_input_mask_share;

    // Secret share of output mask. We need m different output mask shares - one
    // for each interval.
    std::vector<uint64_t> mic_output_mask_shares;

};

// Each SigmoidPrecomputedValue struct holds preprocessing information needed to evaluate
// sigmoid securely on batch of inputs
struct SigmoidPrecomputedValue {

    // ########################     Branching

    std::vector<BranchingPrecomputedValue> branching_precomp;


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

struct  RoundOneSigmoidState{

    // Secret share of actual input
    std::vector<uint64_t> shares_of_sigmoid_inputs;

    // Secret share of masked input. We will have n shares - one for each input
    // x_i in the batch.
    std::vector<uint64_t> shares_of_masked_input_for_branching;

    RoundOneSplineState round_one_spline_state_pos;

    RoundOneSplineState round_one_spline_state_neg;

    // TODO: secure_exponentiation.h contains 2 different state structs one for each party. But
    // we want a single to handle both of them in a single RoundOneSigmoidState struct.
    // Temporary fix : Include both state structs in defining RoundOneSigmoidMessage and ensure that
    // only one (the relevant one depending on the party) is populated
    std::vector<private_join_and_compute::SecureExponentiationPartyZero::State> round_one_exp_state_party0_pos;

    std::vector<private_join_and_compute::SecureExponentiationPartyZero::State> round_one_exp_state_party0_neg;

    std::vector<private_join_and_compute::SecureExponentiationPartyOne::State> round_one_exp_state_party1_pos;

    std::vector<private_join_and_compute::SecureExponentiationPartyOne::State> round_one_exp_state_party1_neg;


};

struct  RoundTwoSigmoidState{

    RoundTwoSplineState round_two_spline_state_pos;

    RoundTwoSplineState round_two_spline_state_neg;

    PowersStateRoundOne round_one_polynomial_state_pos;

    PowersStateRoundOne round_one_polynomial_state_neg;

    //exp result state
    std::vector<FixedPointElement> exp_result_state_pos;

    std::vector<FixedPointElement> exp_result_state_neg;

    //MIC result state
    std::vector<std::vector<uint64_t>> mic_gate_result_share;

};

struct  RoundThreeSigmoidState{

    std::vector<uint64_t> spline_res_in_secret_shared_form_pos;

    std::vector<uint64_t> spline_res_in_secret_shared_form_neg;

    PowersStateRoundOne round_one_polynomial_state_pos;

    PowersStateRoundOne round_one_polynomial_state_neg;

    PolynomialShareOfPolynomialShare round_two_polynomial_state_pos;

    PolynomialShareOfPolynomialShare round_two_polynomial_state_neg;

    //MIC result state
    std::vector<std::vector<uint64_t>> mic_gate_result_share;

};

struct  RoundFourSigmoidState{

    std::vector<BatchedMultState> hadamard_state_for_branching;

};

class SecureSigmoid {

private:

    // A pointer to a Multiple Interval Containment gate.
    const std::unique_ptr<MultipleIntervalContainmentGate> mic_gate_;

    // A pointer to a Secure Spline gate.
    std::unique_ptr<SecureSpline> secure_spline_;

    // Number of inputs in the batch.
    const uint64_t num_inputs_;

    // Private constructor, called by `Create`.
    SecureSigmoid(uint64_t num_inputs,
                  std::unique_ptr<MultipleIntervalContainmentGate> MicGate,
                  std::unique_ptr<SecureSpline> secure_spline,
                  std::unique_ptr<FixedPointElementFactory> fixed_point_factory,
                  SecureSigmoidParameters sigmoid_params,
                  std::unique_ptr<SecureExponentiationPartyZero> exp_party_zero,
                  std::unique_ptr<SecureExponentiationPartyOne> exp_party_one);

    // SecureSigmoid is neither copyable nor movable.
    SecureSigmoid(const SecureSigmoid &) = delete;

    SecureSigmoid &operator=(const SecureSigmoid &) = delete;

public:

    // Parameters for sigmoid specification.
    const SecureSigmoidParameters sigmoid_params_;

    // Pointer to parties for exponentiation protocol. Ideally, we should have
    // a single pointer but this is not possible with the way exponentiation
    // protocol is currently implemented.
    std::unique_ptr<SecureExponentiationPartyZero> exp_party_zero_;
    std::unique_ptr<SecureExponentiationPartyOne> exp_party_one_;


    // Pointer to Fixed Point factory for converting between double and
    // fixed point representation.
    std::unique_ptr<FixedPointElementFactory> fixed_point_factory_;

    // Factory method : creates and returns a SecureSigmoid initialized with
    // appropriate parameters.
    static absl::StatusOr<std::unique_ptr<SecureSigmoid>> Create(
            uint64_t num_inputs, SecureSigmoidParameters sigmoid_params);

    // Performs precomputation stage of sigmoid and returns a pair of
    // n SigmoidPrecomputedValues - one for each party for each input x_i.
    StatusOr<std::pair<SigmoidPrecomputedValue, SigmoidPrecomputedValue>>
    PerformSigmoidPrecomputation();

    StatusOr<std::pair<RoundOneSigmoidState, RoundOneSigmoidMessage>>
    GenerateSigmoidRoundOneMessage(int partyid,
            SigmoidPrecomputedValue sigmoid_precomputed_value,
            std::vector<uint64_t> &share_of_sigmoid_inputs);

    StatusOr<std::pair<RoundTwoSigmoidState, RoundTwoSigmoidMessage>>
    GenerateSigmoidRoundTwoMessage(
            int partyid,
            SigmoidPrecomputedValue sigmoid_precomputed_value,
            RoundOneSigmoidState round_one_state_this_party,
            RoundOneSigmoidMessage round_one_msg_other_party);


    StatusOr<std::pair<RoundThreeSigmoidState, RoundThreeSigmoidMessage>>
    GenerateSigmoidRoundThreeMessage(
            int partyid,
            SigmoidPrecomputedValue sigmoid_precomputed_value,
            RoundTwoSigmoidState round_two_state_this_party,
            RoundTwoSigmoidMessage round_two_msg_other_party);


    StatusOr<std::pair<RoundFourSigmoidState, RoundFourSigmoidMessage>>
    GenerateSigmoidRoundFourMessage(
            int partyid,
            SigmoidPrecomputedValue sigmoid_precomputed_value,
            RoundThreeSigmoidState round_three_state_this_party,
            RoundThreeSigmoidMessage round_three_msg_other_party);


    StatusOr<std::vector<uint64_t>> GenerateSigmoidResult(
            int partyid,
            SigmoidPrecomputedValue sigmoid_precomputed_value,
            RoundFourSigmoidState round_four_state_this_party,
            RoundFourSigmoidMessage round_four_msg_other_party);
};
    }  // namespace applications
}  // namespace private_join_and_compute


#endif //GOOGLE_CODE_SECURE_SIGMOID_H


