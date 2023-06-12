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
#ifndef PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_POWERS_H_
#define PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_POWERS_H_

#include <memory>
#include <utility>
#include <vector>

#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "secret_sharing_mpc/gates/powers_messages.pb.h"

namespace private_join_and_compute {

// The current implementation of powers is NOT secure
// It needs to be transformed in a similar way as polynomial.cc
// to get security

// Contains shares of m-b, which is revealed during the powers protocol.
// Also contains the xor of the 1-2 OT receiver bit with the choice bit
// from the ROT, which is necessary for part 2 of the ROT -> 1-2 OT protocol
struct PowersStateRoundOne {
  // p (P_0) or q (P_1) (bit) in algorithm definition
  // This is the bit at [m-b]^num_fractional_bits for party i's
  // not truncated share
  std::vector<bool> ot_receiver_bit;
  // Contains only last num_fractional_bits (the remaining are zeroed out)
  std::vector<uint64_t> share_m_minus_b_fractional;
};

// Contains shares of m-b, which is revealed during the powers protocol.
struct PowersStateMminusB {
  std::vector<uint64_t> share_m_minus_b_full;
  // Contains only last num_fractional_bits (the remaining are zeroed out)
  std::vector<uint64_t> share_m_minus_b_fractional;
};

struct PowersStateOTInputs {
  // p or q (bit) in algorithm definition
  // This is the bit at [m-b]^num_fractional_bits for party i's
  // not truncated share
  std::vector<bool> receiver_bit;
  // powers_input_x are ordered according to p/q for party P_0 and P_1
  // respectively and carry bit c
  // Result of using truncated [m-b]^0 (when m >= b)
  // Result of using truncated [m-b]^1 (when m < b)
  std::vector<std::vector<uint64_t>> powers_input_0;
  std::vector<std::vector<uint64_t>> powers_input_1;
};

struct PowersShareOfPowersShare {
  // Each OT outputs a share to each party of each party's share of powers
  // 1. One of these shares is output by OT
  // 2. The other share is sampled uniformly and helps to form the OT inputs.
  std::vector<std::vector<uint64_t>> powers_share;
};

// The following functions are used for secure
// powers computation via a protocol based on Figure 7 of HoneyBadgerMPC.
// The paper can be found at: https://eprint.iacr.org/2019/883.pdf
// Our protocol is adapted to our fixed-point ring

namespace internal {

// Preprocessing phase for the Powers operation invoked in batch of n
// and for maximum power k.
// Input: k, n
// Output: [b], [b^2], ..., [b^k] where b is a random vector of size n
// and in range [0,1) i.e. 1 => 2^num_fractional_bits
// The current solution is a trusted dealer solution
// I.e. this function is only for testing purposes.
// The function generates shares of [b], [b^2], ..., [b^k] insecurely.
// Each power of b introduces an error of at most 2^-(num_fractional_bits).
// This error is a result of truncating in fixed point multiplication
// Elements of b are sampled uniformly in the appropriate range:
// [0-2^num_fractional_bits) i.e. [0-1)
// Note that this differs from the plain honey-badger power approach as:
// 1. b^k must be small enough not to overflow uint64_t
// 2. When nesting multiplication, error can grow significantly if b is large
// i.e. the error depends on b itself
// This is an insecure, trusted dealer function.;p
StatusOr<std::pair<std::vector<std::vector<uint64_t>>,
                   std::vector<std::vector<uint64_t>>>>
SamplePowersOfRandomVector(
    size_t k, size_t n,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_,
    uint64_t modulus);

// Helper function to compute powers
// (not invoked by the client but by PowersGenerateOTInputsPartyOne)
// This function directly computes the powers protocol from HoneyBadgerMPC
// This function should not be invoked by the client, and hence is in internal namespace
// It is secure.
StatusOr<std::vector<std::vector<uint64_t>>>
HoneyBadgerPowersPartyOne(size_t k, size_t n,
                          const std::vector<std::vector<uint64_t>>& random_powers_share,
                          std::vector<FixedPointElement>& m_minus_b_option,
                          std::unique_ptr<FixedPointElementFactory>& fp_factory_,
                          uint64_t modulus);

// Helper function to compute powers
// (not invoked by the client but by PowersGenerateOTInputsPartyZero)
// This function directly computes the powers protocol from HoneyBadgerMPC
// This function should not be invoked by the client, and hence is in internal namespace
// It is secure.
StatusOr<std::vector<std::vector<uint64_t>>>
HoneyBadgerPowersPartyZero(size_t k, size_t n,
                           const std::vector<std::vector<uint64_t>>& random_powers_share,
                           std::vector<FixedPointElement>& m_minus_b_option,
                           std::unique_ptr<FixedPointElementFactory>& fp_factory_,
                           uint64_t modulus);

}  // namespace internal

// Helper function to generate the message for the Powers operation.
// P0 and P1 each hold shares [m], and a random [b] (both in [0,2^lf)).
// [m] and [b] are vectors of the same size
// The parties want to compute [m], [m^2], [m^3], ..., [m^k],
// for all elements in m where ^ represents power mod modulus.
// P0 and P1 need to reconstruct (m-b) with only the last num_fractional_bits
// bits -> (m-b)_low
// Each sends the message containing [m-b]_low to the other party.
// Each party also stores its untruncated [m-b], as the bits in
// num_fractional_bits position will be used in the next step.
// We pass the complete random_powers_share from the preprocessing phase.
// This is only for client convenience, only the first power 'b' will be used.
StatusOr<std::pair<PowersStateMminusB, PowersMessageMminusB>>
GenerateBatchedPowersMessage(
    const std::vector<uint64_t>& share_m,
    const std::vector<std::vector<uint64_t>>& random_powers_share,
    uint64_t num_fractional_bits,
    uint64_t modulus);

// This function 1. Computes a single share of a share of powers output
// by uniformly sampling each element and 2. generates inputs to two OTs
// for a single party P{0,1} (once as a receiver and once as a sender). The OT
// inputs will be used in the next step of the protocol to compute the second
// share of a share of powers, which will enable to reconstruct the output of
// powers.
// This function invokes two honey badger protocols as a subprocedure.
// Input: state: party's share of [m - b] where [m] and [b]
//          are vectors of the same size and both in [0,2^num_fractional_bits).
//          The state includes the truncated [m-b]^low as well as the full [m-b]
//        other_party_message: other party's share of [m - b]^low
//          with only the last num_fractional_bits filled
//        random_powers_share (result of preprocessing from the offline phase):
//          1st-kth powers of a random vector b,
//          i.e. [b], [b^2], ..., [b^k]. Outer vector is of size k, inner
//          vector depends on application (for logreg it is of size equal
//          to the number of logreg training examples)
//        modulus: we are computing powers mod modulus
// Output: 1. a single share of a share of powers output
//         2. Inputs to 2 OTs that will enable to compute the second share
//            of a share of powers.
// There is one function per party. This is because
// the protocol is not completely symmetric in that P0 and P1 need to set
// a sharing of 1 in an internal data structure (2^num_fractional_bits in the
// ring), hence there are two separate functions, one for each party.
StatusOr<std::pair<PowersStateOTInputs, PowersShareOfPowersShare>>
PowersGenerateOTInputsPartyZero(
    PowersStateMminusB state,
    const std::vector<std::vector<uint64_t>>& random_powers_share,
    PowersMessageMminusB other_party_message,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_, uint64_t modulus);

StatusOr<std::pair<PowersStateOTInputs, PowersShareOfPowersShare>>
PowersGenerateOTInputsPartyOne(
    PowersStateMminusB state,
    const std::vector<std::vector<uint64_t>>& random_powers_share,
    PowersMessageMminusB other_party_message,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_, uint64_t modulus);

// The function below is an insecure test function.
// This function takes inputs to both OTs from both parties
// and computes the OT output for each party.
// Output: OT Output for each P_0, P_1. The OT output represents a share of a
//         share of powers output from the 'other' party
// Each OT input contains a receiver bit and the two sender messages.
// Given inputs from both parties, computing the OT output is immediate.
StatusOr<std::pair<PowersShareOfPowersShare, PowersShareOfPowersShare>>
PowersGenerateOTOutputForTesting(
    PowersStateOTInputs p_0_inputs,
    PowersStateOTInputs p_1_inputs,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_, uint64_t modulus);

// PowersOutput computes and outputs a sharing [m, m^2, ..., m^k] where m: [0,1)
// Inputs: share0, share1 are vectors of size n (n invocations of powers),
// each constituting a share for all k powers.
// Both inputs share0 and share1 belong to 1 party (they are share of a share of
// powers [m, m^2, ..., m^k] i.e. [[m, m^2, ..., m^k]_P{0,1}])
// One of these shares was drawn uniformly while the other was received via OT
// in the previous steps.
// Output: [m, m^2, ..., m^k] = share0[i] + share1[i] for all i: [0,n)
StatusOr<std::vector<std::vector<uint64_t>>>
PowersOutput(
    PowersShareOfPowersShare share0,
    PowersShareOfPowersShare share1,
    uint64_t modulus);

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_POWERS_H_
