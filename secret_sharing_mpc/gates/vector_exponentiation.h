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

#ifndef PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_VECTOR_EXPONENTIATION_H_
#define PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_VECTOR_EXPONENTIATION_H_

#include <memory>
#include <utility>
#include <vector>

#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/secure_exponentiation.h"
#include "poisson_regression/secure_exponentiation.pb.h"

namespace private_join_and_compute {


// This file implements the two-party secure exponentiation protocol.
// The two parties, denoted by P_0 and P_1, are given additive shares
// x_0 and x_1 respectively, of an exponent x, represented as a
// FixedPointElement in ring R.
// The interactive secure exponentation protocol results in P_0 having output
// y_0 and P_1 having output y_1 such that (y_0, y_1) is an approximate
// additive sharing (in ring R) of the FixedPointElement that represents e^x.
//
// This operation is a batched operation i.e. x is a vector and the output is
// e^x[i] for all indices i
//
// For details on secure exponentiation of a single element see
// 'poisson_regression/secure_exponentiation.cc'

// Preprocessing phase for the exponentiation protocol.
// The current solution is a trusted dealer solution and returns shares for
// both parties.
// Trusted dealer generates tuples of random values (a_0, a_1, b_0, b_1)
// using a PRNG such that a_0*a_1 + b_0*b_1 = 1 mod modulus.
// The number of tuples must be at least 1, else the function returns
// INVALID_ARGUMENT.
// Trusted dealer distributes (a_0, b_0)s to P_0 and (a_1, b_1)s to P_1
StatusOr<std::pair<MultToAddShare, MultToAddShare>> SampleMultToAddSharesVector(
    size_t length, uint64_t modulus);

// Generates the protocol message to send to party P_1 as well as state
// required for the second part of the computation.
// This message is for the subprotocol that converts multiplicative shares
// in \Z_{prime_q} to additive shares in \Z_{prime_q}.
// The function takes as input a FixedPointElement (fpe_shares_zero)
// representing P_0's share of the exponent vector x and a pre-processed tuple
// of vectors (alpha_zero, beta_zero).
// The protocol has undefined behavior if the pre-processed input
// is not correct.
StatusOr<std::pair<BatchedExponentiationPartyZeroMultToAddMessage,
                   std::vector<private_join_and_compute::SecureExponentiationPartyZero::State>>>
GenerateVectorMultToAddMessagePartyZero(
    std::unique_ptr<SecureExponentiationPartyZero>& party_zero_,
    const std::vector<FixedPointElement>& fpe_shares_zero,
    const std::vector<uint64_t>& alpha_zero,
    const std::vector<uint64_t>& beta_zero);

StatusOr<std::pair<BatchedExponentiationPartyOneMultToAddMessage,
                   std::vector<private_join_and_compute::SecureExponentiationPartyOne::State>>>
GenerateVectorMultToAddMessagePartyOne(
    std::unique_ptr<SecureExponentiationPartyOne>& party_one_,
    const std::vector<FixedPointElement>& fpe_shares_one,
    const std::vector<uint64_t>& alpha_one,
    const std::vector<uint64_t>& beta_one);

// Takes as input the protocol message from P_1 and SecureExpPartyZero
// struct msg containing the state from GenerateVectorMultToAddMessage and
// the preprocessed tuple and outputs the final protocol result which is an
// additive share of e^x in the ring \Z_{fpe_params_.primary_ring_modulus}.
// Returns an INVALID_ARGUMENT error code if the message from P_1
// is malformed.
// The pre-processed tuple of vectors (alpha_zero, beta_zero) should be
// the same as the one input to GenerateMultToAddMessage that created the
// interaction message.
//
// The protocol has undefined behavior if the two messages were not created
// from FixedPointElements in the same ring.
// The protocol also has undefined behavior if the pre-processed tuple is
// incorrect.
//
// The function for PartyOne has symmetric input/output parameters.
StatusOr<std::vector<FixedPointElement>> VectorExponentiationPartyZero(
    std::unique_ptr<SecureExponentiationPartyZero>& party_zero_,
    const BatchedExponentiationPartyOneMultToAddMessage& party_one_msg,
    const std::vector<private_join_and_compute::SecureExponentiationPartyZero::State>& state);

StatusOr<std::vector<FixedPointElement>> VectorExponentiationPartyOne(
    std::unique_ptr<SecureExponentiationPartyOne>& party_one_,
    const BatchedExponentiationPartyZeroMultToAddMessage& party_zero_msg,
    const std::vector<private_join_and_compute::SecureExponentiationPartyOne::State>& state);

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_VECTOR_EXPONENTIATION_H_
