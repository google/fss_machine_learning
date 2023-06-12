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


//
//  OLD PROTOCOL
//

#ifndef GOOGLE_CODE_APPLICATIONS_SECURE_COMPARISON_SECURE_COMPARISON_H_
#define GOOGLE_CODE_APPLICATIONS_SECURE_COMPARISON_SECURE_COMPARISON_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <tuple>
#include <vector>

#include "applications/secure_comparison/secure_comparison.pb.h"
#include "fss_gates/comparison.h"
#include "fss_gates/equality.h"
#include "poisson_regression/beaver_triple_utils.h"

namespace private_join_and_compute {
namespace secure_comparison {

// Secure comparison only where Q IS A POWER OF 2 AND m | l
// General case not handled

// Contains shares of short comparison / equality outputs
// Note the structure is different from e.g. in InputShortComparisonEqualityMasks
// This is to avoid unnecessary copying in the last part of the algorithms where shares are AND,XORed
// even_comparison_even_equality_output_shares (flattened to a single vector so that we can use HadamardProduct)
//     [x0 < y0] : for all comparisons 0 - batch_size-1
//     [x2 < y2] : for all comparisons 0 - batch_size-1
//     ...
//     [x_{batch_size-1} < y_{batch_size-1}] : for all comparisons 0 - batch_size-1
//     [x0 == y0] : for all equalities 0 - batch_size-1
//     [x2 == y2] : for all equalities 0 - batch_size-1
//     ...
//     [x_{batch_size-1} == y_{batch_size-1}] : for all equalities 0 - batch_size-1
//
// odd_equality_twice_copied_shares
//     [x1 == y1] : for all equalities 0 - batch_size-1
//     [x3 == y3] : for all equalities 0 - batch_size-1
//     ...
//     [x_{batch_size-1} == y_{batch_size-1}] : for all equalities 0 - batch_size-1
//     Copy the identical [x1==y1], [x3==y3], etc., once more (for vectorization of hadamard product)
// odd_comparison_appended_zeros_shares
//     [x1 < y1] : for all comparisons 0 - batch_size-1
//     [x3 < y3] : for all comparisons 0 - batch_size-1
//     ...
//     [x_{batch_size-1} < y_{batch_size-1}] : for all comparisons 0 - batch_size-1
//     Append zeros to double the current vector size (for vectorized xor)
struct ComparisonShortComparisonEquality {
  std::vector<uint64_t> even_comparison_even_equality_output_shares;
  std::vector<uint64_t> odd_equality_twice_copied_shares;
  std::vector<uint64_t> odd_comparison_appended_zeros_shares;
};

/*// Contains shares of short comparison / equality outputs
// For each large comparison input x=x1...xq,y=y1...yq (remember it is executed in batch)
// comparison_output_shares
//     Comparison 1: [x1 < y1] ... [xq < yq]
//     ...
//     Comparison n: [x1 < y1] ... [xq < yq]
// Same for equality_output_shares but uses '='
struct ComparisonShortComparisonEquality {
  std::vector<std::vector<uint64_t>> comparison_output_shares;
  std::vector<std::vector<uint64_t>> equality_output_shares;
};*/

// Contains FSS key for each comparison invocation of the
// short comparison gate: comparison_key_shares, and the
// short equality gate: equality_key_shares
// For x[i] = x1....xq, y[i] y1...yq (for each comparison input x[],y[]),
// the message is structured as follows:
// comparison_key_shares
//     Comparison 1: key: x_1 < y_1 ... key: x_q < y_q
//     ...
//     Comparison n: key: x_1 < y_1 ... key: x_q < y_q
// Symmetric for equality_key_shares,
struct ShortComparisonEqualityKeyShares {
  std::vector<std::vector<distributed_point_functions::fss_gates::CmpKey>> comparison_key_shares;
  std::vector<std::vector<distributed_point_functions::fss_gates::EqKey>> equality_key_shares;
};

// Contains input masks for the short equality and comparison gates
// For x[i] = x1....xq (for each comparison input x[]), the message is structured as follows:
// input_mask_short_comparison
//     Comparison 1: mask_1 ... mask_q
//     ...
//     Comparison n: mask_1 ... mask_q
// Symmetric for input_mask_short_equality
struct InputShortComparisonEqualityMasks {
  // Contains only last block size bits (likely 16, the remaining are zeroed out)
  // Kept in uint64_t only for convenience
  // TODO See if we can easily change to uint16_t in the future
  std::vector<std::vector<uint64_t>> input_mask_short_comparison;
  std::vector<std::vector<uint64_t>> input_mask_short_equality;
};

// Contains fss gates created during offline phase (FSS cmp/eq gate)
struct ComparisonEqualityGates {
  std::unique_ptr<distributed_point_functions::fss_gates::ComparisonGate> CmpGate;
  std::unique_ptr<distributed_point_functions::fss_gates::EqualityGate> EqGate;
};

// Contains all offline phase preprocessed values
// For FSS short equality and comparison: keys, masks
// For combining the FSS outputs: beaver_vector_shares
struct ComparisonPreprocessedValues {
  ShortComparisonEqualityKeyShares key_shares;
  InputShortComparisonEqualityMasks masks;
  // flattened so that it can be done in one round for a. all comparisons in a batch
  // and across all x_i, y_i
  // size log_2(num_splits), one for each round
  // Looks as follows:
  // Last round is FIRST in the vector, so that we can use pop_back() and back() in the implementation
  // round i .... j = 0 : for all comparisons 0 - batch_size-1
  //              ...
  //              j = (num_splits / 2^i) - 1: for all comparisons 0 - batch_size-1
  std::vector<BeaverTripleVector<uint64_t>> beaver_vector_shares;
};


// Contains masked inputs for the short equality and comparison gates
// For x[i] = x1....xq (for each comparison input x[]), the message is structured as follows:
// masked_input_short_comparison
//     Comparison 1: masked_x1 ... masked_xq
//     ...
//     Comparison n: masked_x1 ... masked_xq
// Symmetric for masked_input_short_equality
struct ComparisonStateRoundOne {
  // Contains only last block size bits (likely 16, the remaining are zeroed out)
  // Kept in uint64_t only for convenience
  // TODO See if we can easily change to uint16_t in the future
  std::vector<std::vector<uint64_t>> masked_input_short_comparison;
  std::vector<std::vector<uint64_t>> masked_input_short_equality;
};

namespace internal {

// Preprocessing phase for secure comparison
// Insecure - trusted dealer function
StatusOr<std::tuple<ComparisonEqualityGates,
                    ComparisonPreprocessedValues,
                    ComparisonPreprocessedValues>>
ComparisonPrecomputation(size_t batch_size, size_t block_length, size_t num_splits);

}  // namespace internal

// Invoke the first two function ONLY if you are doing secure comparison with secret shared inputs
// i.e. P0: x_0, y_0 and P1: x_1, y_1
// These two functions are not invoked if P0:x and P1: y
// This function is part of the reduction of secure comparison with secret-shared inputs
// to secure comparison where each party holds one input
// Returns 1: msb of the share y_i - x_i
//         2: inputs to the non-secret shared comparison
StatusOr<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
SecretSharedComparisonPrepareInputsPartyZero(
    const std::vector<uint64_t>& share_x,
    const std::vector<uint64_t>& share_y,
    size_t batch_size,
    uint64_t modulus,
    size_t num_ring_bits);

StatusOr<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
SecretSharedComparisonPrepareInputsPartyOne(
    const std::vector<uint64_t>& share_x,
    const std::vector<uint64_t>& share_y,
    size_t batch_size,
    uint64_t modulus,
    size_t num_ring_bits);

// Splits the input into blocks i.e. x = x1...xq
// Generates the round one message for the Comparison operation.
// I.e. masks x1...xq for both the FSS short equality and separately for FSS short comparison
// Since comparison is executed in batch, it generates the masked inputs in line above
// for each short comparison and equality
// See detailed description of the message in secure_comparison.proto
// TODO See if the mask_comparison/equality can be uint16_t
StatusOr<std::pair<ComparisonStateRoundOne, ComparisonMessageRoundOne>>
ComparisonGenerateRoundOneMessage(
    const std::vector<uint64_t>& inp,
    const ComparisonPreprocessedValues& preprocessed_values,
    size_t num_splits, size_t block_length);

// Computes short comparisons/equalities i.e. invokes the FSS gates for them
// For each input x=x1...xq,y=y1...yq (remember it is executed in batch)
// Computes and outputs x1=y1, x1<y1, ...., xq=yq,xq<yq
// Need separate function for each party due to 1. eval and 2. keep inputs in the right order
StatusOr<ComparisonShortComparisonEquality>
ComparisonComputeShortComparisonEqualityPartyZero(
    const ComparisonEqualityGates& short_gates,
    const ComparisonStateRoundOne& masked_inp,
    ComparisonMessageRoundOne other_party_masked_inp,
    const ComparisonPreprocessedValues& preprocessed_values,
    size_t num_splits, size_t block_length);

StatusOr<ComparisonShortComparisonEquality>
ComparisonComputeShortComparisonEqualityPartyOne(
    const ComparisonEqualityGates& short_gates,
    const ComparisonStateRoundOne& masked_inp,
    ComparisonMessageRoundOne other_party_masked_inp,
    const ComparisonPreprocessedValues& preprocessed_values,
    size_t num_splits, size_t block_length);

// The 2 functions below are invoked log_2 num_splits times and result in log_2 num_splits rounds
// They are parameterized by i = 1 .... log_2 q (where q is num_splits)
// After each invocation, there remain q/2^i terms
// i.e. the number of terms halves until there is only 1 term left
// Final invocation returns the shares of the comparison output

// The following function returns message to do hadamard product on shares output by ComparisonComputeShortComparisonEquality
// Only calls GenerateHadamardProductMessage
StatusOr<std::pair<BatchedMultState, MultiplicationGateMessage>>
ComparisonGenerateNextRoundMessage(
    const ComparisonShortComparisonEquality& shares,
    const ComparisonPreprocessedValues& preprocessed_values);

// One more function ComparisonProcessNextRoundMessage
// This function finishes the Hadamard product and does some addition mod 2
// to get the shares for the next round of Hadamard product (or comparison output if last invocation)
// There are two separate functions as Hadamard product is not symmetric for the two parties
// (since Beaver multiplication is not symmetric)
StatusOr<ComparisonShortComparisonEquality>
ComparisonProcessNextRoundMessagePartyZero(
    const ComparisonShortComparisonEquality& shares, // for XORs
    BatchedMultState state,
    ComparisonPreprocessedValues& preprocessed_values,
    MultiplicationGateMessage other_party_message,
    size_t num_splits_after_this_round);  // halves each round

StatusOr<ComparisonShortComparisonEquality>
ComparisonProcessNextRoundMessagePartyOne(
    const ComparisonShortComparisonEquality& shares, // for XORs
    BatchedMultState state,
    ComparisonPreprocessedValues& preprocessed_values,
    MultiplicationGateMessage other_party_message,
    size_t num_splits_after_this_round);  // halves each round

// For secret-shared comparison only
// finish the reduction by adding first bit to the comparison output
StatusOr<std::vector<uint64_t>>
SecretSharedComparisonFinishReduction(
    const std::vector<uint64_t>& share_first_bit,
    const std::vector<uint64_t>& share_comparison_output,
    size_t batch_size);

}  // namespace secure_comparison
}  // namespace private_join_and_compute

#endif //GOOGLE_CODE_APPLICATIONS_SECURE_COMPARISON_SECURE_COMPARISON_H_
