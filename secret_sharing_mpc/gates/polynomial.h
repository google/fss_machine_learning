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

#ifndef PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_POLYNOMIAL_H_
#define PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_POLYNOMIAL_H_

#include <memory>
#include <utility>
#include <vector>

#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "secret_sharing_mpc/gates/polynomial_messages.pb.h"
#include "secret_sharing_mpc/gates/powers.h"
#include "secret_sharing_mpc/gates/powers_messages.pb.h"

namespace private_join_and_compute {

// The following functions are used for secure
// polynomial evaluation. The protocol is based on the Powers protocol (see powers.h).
// Coefficients are public input: a_0, a_1, ...., a_k
// (they are of type double - not in our fixed point ring)
// Output a share: [a_0 + a_1 * m + a_2 * m^2 + ... + a_k * m^k]

struct PolynomialRandomOTCorrelationSenderMessage {
  uint64_t sender_msg0;
  uint64_t sender_msg1;
};

struct PolynomialRandomOTCorrelationReceiverMessage {
  bool receiver_choice;
  uint64_t receiver_msg;
};


// pair < PolynomialRandomOTPrecomputation, PolynomialRandomOTPrecomputation >
// { sender_msgs = ( x_0, x_1), receiver_msgs = (c, y_c) }
// { sender_msgs = (y_0, y_1), receiver_msgs = (b, x_b) }

struct PolynomialRandomOTPrecomputation {
  std::vector<PolynomialRandomOTCorrelationSenderMessage> sender_msgs;
  std::vector<PolynomialRandomOTCorrelationReceiverMessage> receiver_msgs;
};

struct PolynomialCoefficients {
  // a_0, a_1, ..., a_k from (a_0 + a_1 * m + a_2 * m^2 + ... + a_k * m^k)
  std::vector<double> coefficients;
};

struct PolynomialShareOfPolynomialShare {
  // Each OT outputs a share to each party of each party's share of polynomials
  // 1. One of these shares is output by OT
  // 2. The other share is sampled uniformly and helps to form the OT inputs.
  std::vector<uint64_t> polynomial_share;
};

namespace internal {

// Preprocessing phase for the Polynomials operation

// This preprocessing function is identical to that of Powers
// See detailed description in powers.h
// This function only invokes the 'SamplePowersOfRandomVector'implemented in Powers
// and is included here only for convenience
StatusOr<std::pair<std::vector<std::vector<uint64_t>>,
                   std::vector<std::vector<uint64_t>>>>
PolynomialSamplePowersOfRandomVector (
    size_t k, size_t n,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_,
    uint64_t modulus);

// Random OT Preprocessing
// This function generates random OTs, which are transformed into 1-2 OTs
// with a trick by Beaver (Extending Oblivious Transfers) in the online phase
// This function is insecure and relies on a trusted dealer
StatusOr<std::pair<PolynomialRandomOTPrecomputation, PolynomialRandomOTPrecomputation>>
PolynomialPreprocessRandomOTs (
    size_t n,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_);

}  // namespace internal

// Helper function to generate the round one message for the Polynomial operation.
// This message generation function is identical to that of Powers
// See detailed description in powers_messages.proto
// This function uses structs defined in powers.h, which are currently not used
// in the powers protocol (powers protocol uses at the moment trusted dealer for the OTs)
StatusOr<std::pair<PowersStateRoundOne, PowersMessageRoundOne>>
PolynomialGenerateRoundOneMessage(
    const std::vector<uint64_t>& share_m,
    const std::vector<std::vector<uint64_t>>& random_powers_share,
    PolynomialRandomOTPrecomputation precomputed_ots,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_,
    uint64_t modulus);

// This function 1. Computes a single share of a share of polynomial output
// by uniformly sampling each element (1 for each polynomial invocation)
// and 2. generates inputs to two OTs
// for a single party P{0,1} (once as a receiver and once as a sender). The OT
// inputs will be used in the next step of the protocol to compute the second
// share of a share of polynomial, which will enable to reconstruct the output of
// polynomial.
// This function invokes two honey badger protocols as a subprocedure.
// Input: polynomial_coefficients: a_0, a_1, ..., a_k, public polynomial coefficients
//        state: party's share of [m - b] where [m] and [b]
//          are vectors of the same size and both in [0,2^num_fractional_bits).
//          The state includes the truncated [m-b]^low as well as the full [m-b]
//        other_party_message: other party's share of [m - b]^low
//          with only the last num_fractional_bits filled
//        random_powers_share (result of preprocessing from the offline phase):
//          1st-kth powers of a random vector b,
//          i.e. [b], [b^2], ..., [b^k]. Outer vector is of size k, inner
//          vector depends on application (for logreg it is of size equal
//          to the number of logreg training examples)
//        modulus: we are computing polynomial mod modulus
// Output: 1. a single share of a share of polynomial output
//         2. Inputs to 2 OTs that will enable to compute the second share
//            of a share of polynomial.
// There is one function per party. This is because
// the protocol is not completely symmetric in that P0 and P1 need to set
// a sharing of 1 in an internal data structure (2^num_fractional_bits in the
// ring), hence there are two separate functions, one for each party.
StatusOr<std::pair<PolynomialShareOfPolynomialShare, PolynomialMessageRoundTwo>>
PolynomialGenerateRoundTwoMessagePartyZero(
    PolynomialCoefficients polynomial_coefficients,
    PowersStateRoundOne state,
    const std::vector<std::vector<uint64_t>>& random_powers_share,
    PolynomialRandomOTPrecomputation precomputed_ots,
    PowersMessageRoundOne other_party_message,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_, uint64_t modulus);

StatusOr<std::pair<PolynomialShareOfPolynomialShare, PolynomialMessageRoundTwo>>
PolynomialGenerateRoundTwoMessagePartyOne(
    PolynomialCoefficients polynomial_coefficients,
    PowersStateRoundOne state,
    const std::vector<std::vector<uint64_t>>& random_powers_share,
    PolynomialRandomOTPrecomputation precomputed_ots,
    PowersMessageRoundOne other_party_message,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_, uint64_t modulus);

// The function below is an insecure test function.
// This function takes inputs to both OTs from both parties
// and computes the OT output for each party.
// Output: OT Output for each P_0, P_1. The OT output represents a share of a
//         share of polynomial output from the 'other' party
// Each OT input contains a receiver bit and the two sender messages.
// Given inputs from both parties, computing the OT output is immediate.

// PolynomialOutput computes and outputs a sharing
// [a_0 + a_1 * m + a_2 * m^2 + ... + a_k * m^k] where m: [0,1)
// Inputs: share 0, share 1 are vectors of size n (n invocations of polynomial).
// Both inputs share 0 and share 1 belong to 1 party (they are share of a share of
// polynomial [a_0 + a_1 * m + ... + a_k * m^k]:
// i.e. [[a_0 + a_1 * m + ... + a_k * m^k]_P{0,1}])
// One of these shares was drawn uniformly while the other was received via OT
// in the previous steps.
// This function completes the OT protocol, which provides 1 share of share
// and then adds together both shares of shares to get a single share of polynomial
// Output: [a_0 + a_1 * m + ... + a_k * m^k] = share 0[i] + share 1[i] for all i: [0,n)
// share 0 : randomly generated in previous steps
// share 1 : received via OT
StatusOr<std::vector<uint64_t>>
PolynomialOutput(
    PolynomialMessageRoundTwo other_party_message,  // sender messages (OT - produces 1 share)
    PolynomialRandomOTPrecomputation precomputed_ots, // random ot choice bit, receiver mask
    PolynomialShareOfPolynomialShare random_share,
    PowersStateRoundOne state, // p or q
    uint64_t modulus);

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_POLYNOMIAL_H_
