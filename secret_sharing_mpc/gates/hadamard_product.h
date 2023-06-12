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

#ifndef PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_HADAMARD_PRODUCT_H_
#define PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_HADAMARD_PRODUCT_H_

#include <utility>
#include <vector>

#include "poisson_regression/beaver_triple_utils.h"

namespace private_join_and_compute {


// The functions in this file element-wise multiply two vectors.
// Two players P_0, P_1
// Input: [X], [Y] (shares of vectors or flattened matrices X, Y)
// Output: [Z] = [X] * [Y] where * is Hadamard product
// Matrix dimensions of X, Y, Z are equal

// Preprocessing phase for batched hadamard product with Beaver triples.
// Input: Vector length (can represent a plain vector or a flattened matrix)
//        modulus: computes hadamard product modulo modulus
// Output: [C] = [A] * [B], where [A], [B] are shares of RANDOM vectors A, B
// of size 'length' and C is their hadamard product also of size 'length'
// The current solution is a trusted dealer solution and returns shares for
// both parties.
StatusOr<std::pair<BeaverTripleVector<uint64_t>, BeaverTripleVector<uint64_t>>>
SampleBeaverTripleVector(size_t length, uint64_t modulus);

// Helper function to generate the message for Hadamard Product.
// As part of the evaluation phase, each party computes a share of [X - A]
// and [Y - B] and broadcasts it to all other parties. Consequently,
// each party can reconstruct X - A and Y - B.
// X, Y are the vectors (or flattened matrices) to element-wise multiply while
// A, B are the Beaver triples
StatusOr<std::pair<BatchedMultState, MultiplicationGateMessage>>
GenerateHadamardProductMessage(
    const std::vector<uint64_t>& share_x, const std::vector<uint64_t>& share_y,
    const BeaverTripleVector<uint64_t>& beaver_vector_share, uint64_t modulus);

// Evaluation phase for element-wise hadamard product with Beaver triples.
// There is one function for each party to execute the matrix product.
// Input: state: party's share of [X - A] and [Y - B]
//        other_party_message: other party's share of [X - A] and [Y - B]
//        beaver_vector_share: Beaver Triple vectors from preprocessing,
//          [A], [B], [C] where the dimensions of X, Y, A, B, C are equal
//        modulus: we are doing multiplication mod modulus
// Output: [Z] = [X] * [Y], where [X], [Y] are shares of vectors X, Y
// Due to fixed point representation, the outputs are truncated at the end
// and have an error of at most 2^-lf (where lf = number of fractional bits)
// * represents hadamard product mod modulus.
StatusOr<std::vector<uint64_t>> HadamardProductPartyZero(
    BatchedMultState state,
    const BeaverTripleVector<uint64_t>& beaver_vector_share,
    MultiplicationGateMessage other_party_message, uint8_t num_fractional_bits,
    uint64_t modulus);

StatusOr<std::vector<uint64_t>> HadamardProductPartyOne(
    BatchedMultState state,
    const BeaverTripleVector<uint64_t>& beaver_vector_share,
    MultiplicationGateMessage other_party_message, uint8_t num_fractional_bits,
    uint64_t modulus);

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_HADAMARD_PRODUCT_H_
