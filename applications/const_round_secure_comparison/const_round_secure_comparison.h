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

#ifndef GOOGLE_CODE_CONST_ROUND_SECURE_COMPARISON_H
#define GOOGLE_CODE_CONST_ROUND_SECURE_COMPARISON_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "applications/const_round_secure_comparison/const_round_secure_comparison.pb.h"
#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "absl/numeric/int128.h"
#include "fss_gates/comparison.h"
#include "fss_gates/equality.h"
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"



namespace private_join_and_compute {
    namespace applications {

// Steps:

// In our application, we will invoke comparison on a batch of inputs
// [ (x_1, y_1), ..... , (x_n, y_n)]

// Input : x_i, y_i

// Preprocessing


struct SecureComparisonParameters{
    uint64_t string_length;
    uint64_t num_pieces;
    uint64_t piece_length;
};


struct SecureComparisonPrecomputedValue{

    // Outer vector is of length = batch size
    // Inner vector is of length = no. of pieces

    std::vector<std::vector<distributed_point_functions::fss_gates::CmpKey>> fss_comparison_keys;
    std::vector<std::vector<uint64_t>> short_comparison_input_masks;
    std::vector<std::vector<uint64_t>> short_comparison_output_masks_share;

    std::vector<std::vector<distributed_point_functions::fss_gates::EqKey>> fss_equality_keys;
    std::vector<std::vector<uint64_t>> short_equality_input_masks;
    std::vector<std::vector<uint64_t>> short_equality_output_masks_share;

    std::vector<distributed_point_functions::DpfKey> idpf_keys;
    // Sharing of input mask for the idpf
//    std::vector<uint64_t> idpf_mask_share;
    std::vector<std::vector<uint64_t>> idpf_mask_bit_shares;

    std::vector<BeaverTripleVector<uint64_t>> beaver_vector_shares;
};

struct RoundOneSecureComparisonState{

    // Outer vector is of length = batch size
    // Inner vector is of length = no. of pieces
    std::vector<std::vector<uint64_t>> short_equality_masked_input_pieces;
    std::vector<std::vector<uint64_t>> short_comparison_masked_input_pieces;

};

struct RoundTwoSecureComparisonState{

    // Outer vector is of length = batch size
    // Inner vector is of length = no. of pieces
    std::vector<std::vector<uint64_t>> shares_of_lt;
//    std::vector<uint64_t> masked_shares_of_eq_integer;

    // Outer vector is of length = batch size
    // Inner vector is of length = no. of pieces - 1
    std::vector<std::vector<uint64_t>> masked_input_bit_shares_for_idpf;

};

struct RoundThreeSecureComparisonState{

    // Outer vector is of length = batch size
    // Inner vector is of length = no. of pieces
    std::vector<std::vector<uint64_t>> shares_of_prefixAND_of_eq;

    std::vector<BatchedMultState> hadamard_state;
};


class ConstRoundSecureComparison {

    private:

        // Number of inputs in the batch.
        const uint64_t num_inputs_;

        // A pointer to short Equality gate.
        const std::unique_ptr<distributed_point_functions::fss_gates::EqualityGate> eq_gate_;

        // A pointer to short Comparison gate.
        const std::unique_ptr<distributed_point_functions::fss_gates::ComparisonGate> lt_gate_;

        // A pointed to iDPF
        const std::unique_ptr<distributed_point_functions::DistributedPointFunction> idpf_;

        // Private constructor, called by `Create`.
        ConstRoundSecureComparison(uint64_t num_inputs,
                                   SecureComparisonParameters comparison_params,
                                   std::unique_ptr<distributed_point_functions::fss_gates::EqualityGate> eq_gate,
                                   std::unique_ptr<distributed_point_functions::fss_gates::ComparisonGate> lt_gate,
                                   std::unique_ptr<distributed_point_functions::DistributedPointFunction> idpf);

        // SecureSigmoid is neither copyable nor movable.
        ConstRoundSecureComparison(const ConstRoundSecureComparison &) = delete;

        ConstRoundSecureComparison &operator=(const ConstRoundSecureComparison &) = delete;

    public:

        // Parameters for sigmoid specification.
        const SecureComparisonParameters comparison_params_;

        // Factory method : creates and returns a ConstRoundSecureComparison initialized with
        // appropriate parameters.
        static absl::StatusOr<std::unique_ptr<ConstRoundSecureComparison>> Create(
                uint64_t num_inputs, SecureComparisonParameters comparison_params);

        // Performs precomputation stage of ConstRoundSecureComparison and returns a pair of
        // SecureComparisonPrecomputedValue - one for each party.
        StatusOr<std::pair<SecureComparisonPrecomputedValue, SecureComparisonPrecomputedValue>>
        PerformComparisonPrecomputation();

        StatusOr<std::pair<RoundOneSecureComparisonState, RoundOneSecureComparisonMessage>>
            GenerateComparisonRoundOneMessage(int partyid,
                                              SecureComparisonPrecomputedValue comparison_precomputed_value,
                                           std::vector<uint64_t> &comparison_private_inputs);

        StatusOr<std::pair<RoundTwoSecureComparisonState, RoundTwoSecureComparisonMessage>>
            GenerateComparisonRoundTwoMessage(
                    int partyid,
                    SecureComparisonPrecomputedValue comparison_precomputed_value,
                    RoundOneSecureComparisonState round_one_state_this_party,
                    RoundOneSecureComparisonMessage round_one_msg_other_party);

        StatusOr<std::vector<std::pair<RoundTwoSecureComparisonState, RoundTwoSecureComparisonMessage>>>
        BatchGenerateComparisonRoundTwoMessage(
            int partyid,
            size_t batch_size,
            std::vector<SecureComparisonPrecomputedValue> &comparison_precomputed_values,
            std::vector<RoundOneSecureComparisonState> &round_one_state_this_party,
            std::vector<RoundOneSecureComparisonMessage> &round_one_msgs_other_party);


        StatusOr<std::pair<RoundThreeSecureComparisonState, RoundThreeSecureComparisonMessage>>
            GenerateComparisonRoundThreeMessage(
                    int partyid,
                    SecureComparisonPrecomputedValue comparison_precomputed_value,
                    RoundTwoSecureComparisonState round_two_state_this_party,
                    RoundTwoSecureComparisonMessage round_two_msg_other_party);

        StatusOr<std::vector<uint64_t>> GenerateComparisonResult(
                int partyid,
                SecureComparisonPrecomputedValue comparison_precomputed_value,
                RoundThreeSecureComparisonState round_three_state_this_party,
                RoundThreeSecureComparisonMessage round_three_msg_other_party);
};


    }  // namespace applications
}  // namespace private_join_and_compute


#endif //GOOGLE_CODE_CONST_ROUND_SECURE_COMPARISON_H