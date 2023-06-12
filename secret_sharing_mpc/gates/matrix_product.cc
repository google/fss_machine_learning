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

#include "secret_sharing_mpc/gates/matrix_product.h"

namespace private_join_and_compute {

StatusOr<std::pair<BeaverTripleMatrix<uint64_t>, BeaverTripleMatrix<uint64_t>>>
SampleBeaverTripleMatrix(size_t dim1, size_t dim2, size_t dim3,
                         uint64_t modulus) {
  return internal::SampleBeaverMatrixShareWithPrng(dim1, dim2, dim3, modulus);
}

StatusOr<std::pair<MatrixMultState, MatrixMultiplicationGateMessage>>
GenerateMatrixProductMessage(
    const std::vector<uint64_t>& share_x,
    const std::vector<uint64_t>& share_y,
    const BeaverTripleMatrix<uint64_t>& beaver_matrix_share,
    uint64_t modulus)  {
  return GenerateMatrixMultiplicationGateMessage(share_x, share_y,
                                                 beaver_matrix_share,
                                                 modulus);
}

StatusOr<std::vector<uint64_t>> MatrixProductPartyZero(
    MatrixMultState state,
    const BeaverTripleMatrix<uint64_t>& beaver_matrix_share,
    MatrixMultiplicationGateMessage other_party_message,
    uint8_t num_fractional_bits, uint64_t modulus) {
  // Party 0 computes its share of the output from its input, Beaver triple
  // matrix, and other party's message.
  // The product includes an additional term 2^num_fractional_bits, which needs
  // to be divided. Thus, the result needs to be truncated which introduces
  // an error of at most 2^-lf (lf = num_fractional_bits).
  // The truncation happens only once at the end for each element of the matrix.
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_xy,
      GenerateMatrixMultiplicationOutputPartyZero(
          state, beaver_matrix_share, other_party_message, modulus));
  ASSIGN_OR_RETURN(
      share_xy,
      TruncateSharePartyZero(share_xy, (1ULL << num_fractional_bits), modulus));
  return share_xy;
}

StatusOr<std::vector<uint64_t>> MatrixProductPartyOne(
    MatrixMultState state,
    const BeaverTripleMatrix<uint64_t>& beaver_matrix_share,
    MatrixMultiplicationGateMessage other_party_message,
    uint8_t num_fractional_bits, uint64_t modulus) {
  // Party 1 computes its share of the output from its input, Beaver triple
  // matrix, and other party's message.
  // The product includes an additional term 2^num_fractional_bits, which needs
  // to be divided. Thus, the result needs to be truncated which introduces
  // an error of at most 2^-lf (lf = num_fractional_bits).
  // The truncation happens only once at the end for each element of the matrix.
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_xy,
      GenerateMatrixMultiplicationOutputPartyOne(state, beaver_matrix_share,
                                                 other_party_message, modulus));
  ASSIGN_OR_RETURN(
      share_xy,
      TruncateSharePartyOne(share_xy, (1ULL << num_fractional_bits), modulus));
  return share_xy;
}

}  // namespace private_join_and_compute
