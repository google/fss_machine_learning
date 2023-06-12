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

#ifndef PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_CORRELATED_MATRIX_PRODUCT_H_
#define PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_CORRELATED_MATRIX_PRODUCT_H_

#include <tuple>
#include <utility>
#include <vector>

#include "poisson_regression/beaver_triple_utils.h"
#include "secret_sharing_mpc/gates/correlated_beaver_triple_messages.pb.h"

namespace private_join_and_compute {

// The functions in this file multiply two matrices.
// Unlike in matrix_product.h, this gate should be used when
// one of these matrices is used across multiple products.
// I.e. This gate should be used when
// multiplying [X] * ([Y_0], [Y_1], ..., [Y_n])
// (The gate multiplies a single X * Y but should be used only when there are
// multiple Y_0, ..., Y_n)
// Two players P_0, P_1
// Input: [X], [Y] (shares of matrices X, Y)
// Output: [Z] = [X] * [Y]
// Due to fixed point representation, the outputs are truncated at the end
// and have an error of at most 2^-lf (where lf = number of fractional bits)
// Matrix dimensions: X (dim1 x dim2), Y (dim2 x dim3), Z (dim1 x dim3)

// In order to multiply [X] * ([Y_0], [Y_1], ..., [Y_n]), the functions should
// be invoked in the following order:
// GenerateMatrixXminusAProductMessage (once, exchange masked X between parties)
// GenerateMatrixYminusBProductMessage (for each Y_i, masked Y exchanged
// separately)
// CorrelatedMatrixProductParty{Zero,One} (once for each Y_i, using the fixed
// X-A message and the Y-B message specific to Y_i)

// Preprocessing phase for product with correlated Beaver triples.
// 2 functions
//
// SampleBeaverTripleMatrixA generates random mask A for matrix X, which will be
// reused across multiplications of different matrices Y. The outputs is a
// tuple constituting P_0's share of A, P_1's share of A, and A (reconstructed)
//
// SampleBeaverTripleMatrixBandC generates masks B, C (rest of the beaver
// triple). This function is called once for each Y that X multiplies.
// B is drawn at random, C is computed as C = A * B
// The function takes as input A output by SampleBeaverTripleMatrixA.
// The output is a pair of [B] and [C], a share for each party P_0 and P_1.

StatusOr<std::tuple<std::vector<uint64_t>, std::vector<uint64_t>,
                    std::vector<uint64_t>>>
SampleBeaverTripleMatrixA(size_t dim1, size_t dim2, uint64_t modulus);

StatusOr<std::pair<
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>>,   // P_0's [B], [C]
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>>  // P_1's [B], [C]
SampleBeaverTripleMatrixBandC(
    std::tuple<std::vector<uint64_t>, std::vector<uint64_t>,
               std::vector<uint64_t>>& vector_a,
    size_t dim1, size_t dim2, size_t dim3, uint64_t modulus);

// Helper functions to generate the messages for Correlated Matrix Product Gate.
// As in preprocessing, there are two functions.
//
// GenerateMatrixXminusAProductMessage generates the X masked by A i.e. [X - A]
// which will be sent only once, reconstructed, and reused across multiples Ys.
// Hence, the communication costs only 1 round across all multiplications
//
// GenerateMatrixYminusBProductMessage computes [Y - B] shares, one for each Y
// and reconstructed to Y - B for each Y_i.
// This function costs 1 communication round per product
//
// X, Y are the matrices to multiply while A, B are the Beaver triples

// Takes share_x and the beaver triple matrix as input and
// returns a pair: 1. state share_x_minus_a and 2. message X-A to send to the
// other party
StatusOr<std::pair<std::vector<uint64_t>, MatrixXminusAProductMessage>>
GenerateMatrixXminusAProductMessage(
    const std::vector<uint64_t>& share_x,
    const std::vector<uint64_t>& beaver_matrix_share_a, uint64_t modulus);

// Takes the share_x_minus_a output by GenerateMatrixXminusAProductMessage
// as input and is used alongside share_y_minus_b constructed in this function
// to get 1. MatrixMultState, also computes the Y - B message to the other
// party.
StatusOr<std::pair<MatrixMultState, MatrixYminusBProductMessage>>
GenerateMatrixYminusBProductMessage(
    const std::vector<uint64_t>& share_y,
    const std::vector<uint64_t>& share_x_minus_a,
    const std::vector<uint64_t>& beaver_matrix_share_b, uint64_t modulus);

// Evaluation phase for product with correlated Beaver triples.
// There is one function for each party to execute the matrix product.
// The two different functions are needed as the 1. beaver product
// protocol is asymmetric and the 2. truncation behavior is asymmetric.

// Input: state: party's share of [X - A] and [Y - B]
//        other_party_x_minus_a_message: other party's share of [X - A]
//        other_party_y_minus_b_message: other party's share of [Y - B]
//        beaver_matrix_share{a,b,c}:
//          Beaver Triple from preprocessing, [A], [B], [C]
//        dim{1,2,3}: dimensions A(dim1 x dim2), B(dim2 x dim3), C(dim1 x dim3)
//        modulus: we are doing multiplication mod modulus
// Output: [Z] = [X] * [Y], where [X], [Y] are shares of matrices X, Y
// i.e. [X] (dim1 x dim2) and [Y] (dim2 x dim3)
// * represents matrix product mod modulus.
StatusOr<std::vector<uint64_t>> CorrelatedMatrixProductPartyZero(
    MatrixMultState state, const std::vector<uint64_t>& beaver_matrix_share_a,
    const std::vector<uint64_t>& beaver_matrix_share_b,
    const std::vector<uint64_t>& beaver_matrix_share_c,
    MatrixXminusAProductMessage other_party_x_minus_a_message,
    MatrixYminusBProductMessage other_party_y_minus_b_message, size_t dim1,
    size_t dim2, size_t dim3, uint8_t num_fractional_bits, uint64_t modulus);

StatusOr<std::vector<uint64_t>> CorrelatedMatrixProductPartyOne(
    MatrixMultState state, const std::vector<uint64_t>& beaver_matrix_share_a,
    const std::vector<uint64_t>& beaver_matrix_share_b,
    const std::vector<uint64_t>& beaver_matrix_share_c,
    MatrixXminusAProductMessage other_party_x_minus_a_message,
    MatrixYminusBProductMessage other_party_y_minus_b_message, size_t dim1,
    size_t dim2, size_t dim3, uint8_t num_fractional_bits, uint64_t modulus);

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_CORRELATED_MATRIX_PRODUCT_H_
