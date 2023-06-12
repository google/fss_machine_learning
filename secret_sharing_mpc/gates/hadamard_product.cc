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

#include "secret_sharing_mpc/gates/hadamard_product.h"

namespace private_join_and_compute {

StatusOr<std::pair<BeaverTripleVector<uint64_t>, BeaverTripleVector<uint64_t>>>
SampleBeaverTripleVector(size_t length, uint64_t modulus) {
  return internal::SampleBeaverVectorShareWithPrng(length, modulus);
}

StatusOr<std::pair<BatchedMultState, MultiplicationGateMessage>>
GenerateHadamardProductMessage(
    const std::vector<uint64_t>& share_x,
    const std::vector<uint64_t>& share_y,
    const BeaverTripleVector<uint64_t>& beaver_vector_share,
    uint64_t modulus) {
  return GenerateBatchedMultiplicationGateMessage(share_x, share_y,
                                                 beaver_vector_share,
                                                 modulus);
}

StatusOr<std::vector<uint64_t>> HadamardProductPartyZero(
    BatchedMultState state,
    const BeaverTripleVector<uint64_t>& beaver_vector_share,
    MultiplicationGateMessage other_party_message, uint8_t num_fractional_bits,
    uint64_t modulus) {
  // Each party computes its share of output from its input, Beaver triple
  // matrix, and other party's message.
  // The product includes an additional term 2^num_fractional_bits, which needs
  // to be divided. Thus, the result needs to be truncated which introduces
  // an error of at most 2^-lf (lf = num_fractional_bits).
  // The truncation happens only once at the end for each element of the
  // vector/matrix.
  std::vector<uint64_t> share_xy;
    ASSIGN_OR_RETURN(
        share_xy,
        GenerateBatchedMultiplicationOutputPartyZero(
            state, beaver_vector_share, other_party_message, modulus));
    ASSIGN_OR_RETURN(share_xy,
                     TruncateSharePartyZero(
                         share_xy, (1ULL << num_fractional_bits), modulus));
    return share_xy;
}

StatusOr<std::vector<uint64_t>> HadamardProductPartyOne(
    BatchedMultState state,
    const BeaverTripleVector<uint64_t>& beaver_vector_share,
    MultiplicationGateMessage other_party_message, uint8_t num_fractional_bits,
    uint64_t modulus) {
  // Each party computes its share of output from its input, Beaver triple
  // matrix, and other party's message.
  // The product includes an additional term 2^num_fractional_bits, which needs
  // to be divided. Thus, the result needs to be truncated which introduces
  // an error of at most 2^-lf (lf = num_fractional_bits).
  // The truncation happens only once at the end for each element of the
  // vector/matrix.
  std::vector<uint64_t> share_xy;
  ASSIGN_OR_RETURN(
      share_xy, GenerateBatchedMultiplicationOutputPartyOne(
                    state, beaver_vector_share, other_party_message, modulus));
  ASSIGN_OR_RETURN(
      share_xy,
      TruncateSharePartyOne(share_xy, (1ULL << num_fractional_bits), modulus));
  return share_xy;
}

}  // namespace private_join_and_compute
