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

#ifndef PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_BEAVER_TRIPLE_UTILS_H_
#define PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_BEAVER_TRIPLE_UTILS_H_

#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_messages.pb.h"
#include "poisson_regression/prng/basic_rng.h"

namespace private_join_and_compute {
// The data type to hold correlated randomness used in secure exponentiation.
// P0 holds mult_to_add_share_0 = (a_0, b_0).
// P1 holds mult_to_add_share_1 = (a_1, b_1).
// a_0[i]*a_1[i] + b_0[i]*b_1[i] = 1 mod (modulus).
typedef std::pair<std::vector<uint64_t>, std::vector<uint64_t>> MultToAddShare;

struct BatchedMultState {
  std::vector<uint64_t> share_x_minus_a;
  std::vector<uint64_t> share_y_minus_b;
};

struct MatrixMultState {
  std::vector<uint64_t> share_x_minus_a;
  std::vector<uint64_t> share_y_minus_b;
};

// Helper function to sample a random vector that contains elements of type
// uint64_t. The function takes 3 inputs: length of the desired output,
// a pseudorandom generator PRNG, and the modulus of the ring.
StatusOr<std::vector<uint64_t>> SampleVectorFromPrng(
    size_t length, uint64_t modulus, SecurePrng* prng);

// The following functions are helper functions that are used for secure
// multiplication via the use of Beaver triples.

// Helper function to generate the message for Multiplication Gate.
// P0 and P1 each holds shares [X], [Y], and beaver shares ([A], [B], [C]).
// The parties want to compute [Z] = Fmult([X], [Y]) where z[i] = x[i]*y[i],
// where * represents regular multiplication mod modulus.
// P0 and P1 needs to reconstruct (X-A) and (Y-B).
// Each sends the message containing [X-A] and [Y-B] to the other party.
StatusOr<std::pair<BatchedMultState, MultiplicationGateMessage>>
GenerateBatchedMultiplicationGateMessage(
    const std::vector<uint64_t>& share_x, const std::vector<uint64_t>& share_y,
    const BeaverTripleVector<uint64_t>& beaver_vector_share, uint64_t modulus);

// Functions to generate the output of the batched multiplication gate.
// Each party outputs shares of X*Y, where * represents component-wises
// multiplication mod modulus (x[i]*y[i] mod modulus).
// The behavior of party zero and party one is asymmetric.
// P0: [A]_0*(Y-B) + [B]_0*(X-A) + [C]_0 + (X-A)*(Y-B).
// P1: [A]_1*(Y-B) + [B]_1*(X-A) + [C]_1.
// Where ([A]_i, [B]_i, [C]_i) is Beaver triple vector held by party P_i.
StatusOr<std::vector<uint64_t>> GenerateBatchedMultiplicationOutputPartyZero(
    BatchedMultState state,
    const BeaverTripleVector<uint64_t>& beaver_vector_share,
    MultiplicationGateMessage other_party_message,
    uint64_t modulus);

StatusOr<std::vector<uint64_t>> GenerateBatchedMultiplicationOutputPartyOne(
    BatchedMultState state,
    const BeaverTripleVector<uint64_t>& beaver_vector_share,
    MultiplicationGateMessage other_party_message,
    uint64_t modulus);

// Helper function to generate the message for Matrix Multiplication Gate.
// P0 and P1 each holds shares [X], [Y], and beaver shares ([A], [B], [C]).
// The parties want to compute [Z] = Fmult([X], [Y]) where Z = X*Y,
// where * represents matrix multiplication mod modulus.
// P0 and P1 needs to reconstruct (X-A) and (Y-B).
// Each sends the message containing [X-A] and [Y-B] to the other party.
StatusOr<std::pair<MatrixMultState, MatrixMultiplicationGateMessage>>
GenerateMatrixMultiplicationGateMessage(
    const std::vector<uint64_t>& share_x, const std::vector<uint64_t>& share_y,
    const BeaverTripleMatrix<uint64_t>& beaver_matrix_share, uint64_t modulus);

// Functions to generate the output of the matrix multiplication gate.
// Each party outputs shares of X*Y, where * represents matrix multiplication
// mod modulus.
// The behavior of party zero and party one is asymmetric.
// P0: [A]_0*(Y-B) + (X-A)*[B]_0 + [C]_0 + (X-A)*(Y-B).
// P1: [A]_1*(Y-B) + (X-A)*[B]_1 + [C]_1.
// Where ([A]_i, [B]_i, [C]_i) is Beaver triple matrix held by party P_i.
StatusOr<std::vector<uint64_t>> GenerateMatrixMultiplicationOutputPartyZero(
    MatrixMultState state,
    const BeaverTripleMatrix<uint64_t>& beaver_matrix_share,
    MatrixMultiplicationGateMessage other_party_message,
    uint64_t modulus);

StatusOr<std::vector<uint64_t>> GenerateMatrixMultiplicationOutputPartyOne(
    MatrixMultState state,
    const BeaverTripleMatrix<uint64_t>& beaver_matrix_share,
    MatrixMultiplicationGateMessage other_party_message,
    uint64_t modulus);

// Helper functions to perform truncation of shares in batched setting.
// P0 and P1 each has share of a vector X, and want to compute the truncation
// for each of the element of X. The function is not symmetric and each parties
// perform different operation to do the truncation on shares.
// Let share_x = share_x_0 + share_x_1, then the truncation is done as:
// P0: share_x_0 --> floor(share_x_0/divisor}).
// P1: share_x_1 -->
// modulus - floor((modulus - share_x_1)/divisor).
// Each party outputs a share of (floor(share_x/divisor) + error), where
// error is -1, 0, or 1.
StatusOr<std::vector<uint64_t>>  TruncateSharePartyZero(
    const std::vector<uint64_t>& shares,
    uint64_t divisor, uint64_t modulus);

StatusOr<std::vector<uint64_t>>  TruncateSharePartyOne(
    const std::vector<uint64_t>& shares,
    uint64_t divisor, uint64_t modulus);

// These functions are only for testing purpose.
// The functions generate shares of BeaverTripleVector and BeaverTripleMatrix
// insecurely.
namespace internal {

// Helper function to generate random shares of a Beaver triple vector in the
// form of a pair ((A_1, B_1, C_1), (A_2, B_2, C_2)) from a PRNG such that:
// (A, B, C) = (A_1 + A_2 % modulus, B_1 + B_2 % modulus, C_1 + C_2 % modulus)
// and C[idx] = A[idx]*B[idx] % modulus. * represents regular multiplication.
StatusOr<std::pair<BeaverTripleVector<uint64_t>, BeaverTripleVector<uint64_t>>>
    SampleBeaverVectorShareWithPrng(size_t length, uint64_t modulus);

// Helper function to generate random shares of a Beaver triple matrix in the
// form of a pair ((A_1, B_1, C_1), (A_2, B_2, C_2)) from a PRNG such that:
// (A, B, C) = (A_1 + A_2 % modulus, B_1 + B_2 % modulus, C_1 + C_2 % modulus)
// and C = A*B % modulus. * represents matrix multiplication.
// (dim1, dim2, dim3) define the size of the matrices A, B, and C.
// #row x #col of A: dim1 x dim2.
// #row x #col of B: dim2 x dim3.
// #row x #col of C: dim1 x dim3.
StatusOr<std::pair<BeaverTripleMatrix<uint64_t>, BeaverTripleMatrix<uint64_t>>>
    SampleBeaverMatrixShareWithPrng(
        size_t dim1, size_t dim2, size_t dim3, uint64_t modulus);

// Helper function to generate tuples of random values (a_0, b_0, a_1, b_1)
// using a PRNG such that a_0*a_1 + b_0*b_1 = 1 mod modulus.
// The number of tuples must be at least 1, else the function returns
// INVALID_ARGUMENT.
// The tuples are used to compute secure exponentiation.
StatusOr<std::pair<MultToAddShare, MultToAddShare>>
    SampleMultToAddSharesWithPrng(size_t length, uint64_t modulus);


// Helper function to generate share of zero. The output is a pair of random
// vectors A and B such that A[i] + B[i] = 0 mod modulus.
StatusOr<std::pair<std::vector<uint64_t>,
                   std::vector<uint64_t>>> SampleShareOfZero(
    size_t length, uint64_t modulus);

}  // namespace internal
}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_BEAVER_TRIPLE_UTILS_H_
