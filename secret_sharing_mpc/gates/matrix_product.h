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

#ifndef PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_MATRIX_PRODUCT_H_
#define PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_MATRIX_PRODUCT_H_

#include <utility>
#include <vector>

#include "poisson_regression/beaver_triple_utils.h"

namespace private_join_and_compute {


// The functions in this file multiply two matrices.
// Two players P_0, P_1
// Input: [X], [Y] (shares of matrices X, Y)
// Output: [Z] = [X] * [Y]
// Due to fixed point representation, the outputs are truncated at the end
// and have an error of at most 2^-lf (where lf = number of fractional bits)
// Matrix dimensions: X (dim1 x dim2), Y (dim2 x dim3), Z (dim1 x dim3)

// Preprocessing phase for product with Beaver triples.
// Input: Matrix dimensions for matrix A (dim1 x dim2) and B (dim2 x dim3)
// Output: [C] = [A] * [B], where [A], [B] are shares of RANDOM matrices A, B
// The current solution is a trusted dealer solution and returns shares for
// both parties.
StatusOr<std::pair<BeaverTripleMatrix<uint64_t>, BeaverTripleMatrix<uint64_t>>>
SampleBeaverTripleMatrix(size_t dim1, size_t dim2, size_t dim3,
                         uint64_t modulus);

// Helper function to generate the message for Matrix Product Gate.
// As part of the evaluation phase, each party computes a share of [X - A]
// and [Y - B] and broadcasts it to all other parties. Consequently,
// each party can reconstruct X - A and Y - B.
// X, Y are the matrices to multiply while A, B are the Beaver triples
StatusOr<std::pair<MatrixMultState, MatrixMultiplicationGateMessage>>
GenerateMatrixProductMessage(
    const std::vector<uint64_t>& share_x, const std::vector<uint64_t>& share_y,
    const BeaverTripleMatrix<uint64_t>& beaver_matrix_share, uint64_t modulus);

// Evaluation phase for product with Beaver triples.
// There is one function for each party to execute the matrix product.
// Input: state: party's share of [X - A] and [Y - B]
//        other_party_message: other party's share of [X - A] and [Y - B]
//        beaver_matrix_share: Beaver Triple from preprocessing, [A], [B], [C]
//        modulus: we are doing multiplication mod modulus
// Output: [Z] = [X] * [Y], where [X], [Y] are shares of matrices X, Y
// i.e. [X] (dim1 x dim2) and [Y] (dim2 x dim3)
// * represents matrix product mod modulus.
StatusOr<std::vector<uint64_t>> MatrixProductPartyZero(
    MatrixMultState state,
    const BeaverTripleMatrix<uint64_t>& beaver_matrix_share,
    MatrixMultiplicationGateMessage other_party_message,
    uint8_t num_fractional_bits, uint64_t modulus);

StatusOr<std::vector<uint64_t>> MatrixProductPartyOne(
    MatrixMultState state,
    const BeaverTripleMatrix<uint64_t>& beaver_matrix_share,
    MatrixMultiplicationGateMessage other_party_message,
    uint8_t num_fractional_bits, uint64_t modulus);

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_MATRIX_PRODUCT_H_
