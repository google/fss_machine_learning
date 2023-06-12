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

#ifndef PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_SECURE_EXPONENTIATION_H_
#define PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_SECURE_EXPONENTIATION_H_

// This file implements the two-party secure exponentiation protocol.
// The two parties, denoted by P_0 and P_1, are given additive shares
// x_0 and x_1 respectively, of an exponent x, represented as a
// FixedPointElement in ring R.
// The interactive secure exponentation protocol results in P_0 having output
// y_0 and P_1 having output y_1 such that (y_0, y_1) is an approximate
// additive sharing (in ring R) of the FixedPointElement that represents e^x.
//
// P_0 and P_1 are also provided pre-processed input (alpha_zero, beta_zero),
// and (alpha_one, beta_one) such that
// (alpha_zero * alpha_one + beta_zero * beta_one = 1).
//
// In the protocol, each party can locally compute to end up with multiplicative
// shares of the result (e^x) along with some additional multipliers
// in \Z_{\prime_q}. Each party will now send exactly one message in the
// interactive part of the protocol that converts these shares into additive
// shares of the result in \Z_{\prime_q}. Finally, the parties can divide by
// the previous multipliers and then change their shares to be in the original
// ring R through a local computation.
//
// R and prime_q should be sufficiently large such that x and e^x do not wrap
// around R and \Z_{prime_q}.
//
// The first stage of the protocol firse converts the base e exponent into a
// base 2 exponent. Then, it exponentiates the integer and fractional
// parts separately and combines them to get multiplicative shares. For the
// integer part to work correctly, we require that the overall base 2 exponent
// be >= 1. Since the minimum (base e) exponent is -exponent_bound, the
// protocol first adds a constant to the exponent to make sure all (base 2)
// exponents end up >= 1. This multiplier is then removed in the second stage
// of the protocol.
//
// Suppose that the exponent x is bounded as -A < x < A.
// The multiplicative shares output by the first stage of the computation
// contain two multipliers: (1) fractional_multiplier = 2^{num_fractional_bits};
// (2) 2^{ceil(A log_2(e)) + 1}.
// Therefore, R and prime_q should be sufficiently large such that the maximum
// intermediate computation 2^{2*ceil(A log_2(e)) + 1 + num_fractional_bits}
// should not wrap around R and \Z_{\prime_q}.
// Otherwise, the protocol has undefined behavior.
//
// If x > A, then the protocol should still work if the ring is large enough.
// If x < -A, then in some cases, the protocol might sometimes still return
// the correct output but in most cases, the protocol is expected to fail.
// In general, choosing an exponent_bound A such that (-A < x < A) holds is
// crucial. Otherwise, the protocol may have undefined behavior.
//
// The protocol also has a small failure probability if the sharing of a
// positive FixedPointElement exponent doesn't wrap around the ring modulus,
// or if the sharing of a negative FixedPointElement exponent does wrap around.
// The failure probability can be made smaller by using a larger ring or
// exponents with a smaller absolute value.
//

#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/secure_exponentiation.pb.h"
#include "private_join_and_compute/util/status.inc"

namespace private_join_and_compute {

// Secure exponentiation parameters.
// exponent_bound is a bound such that -exponent_bound < x < exponent_bound.
// prime_q is a prime number that is large enough to fit the result of the
// exponentiation without wrapping around.
struct ExponentiationParams {
  uint64_t exponent_bound;
  uint64_t prime_q;
};

// Class representing the party P_0 in the secure exponentiation protocol.
class SecureExponentiationPartyZero {
 public:
  // State struct for party P_0 created in the first part of the computation
  // (GenerateMultToAddMessage) and input to the second part (OutputResult).
  struct State {
    uint64_t mult_share_zero;  // Mult share resulting from local exponentiation
    // (alpha_zero, beta_zero) is the pre-processed tuple.
    uint64_t alpha_zero;
    uint64_t beta_zero;
  };

  // Returns a std::unique_ptr to a SecureExponentiationPartyZero, which
  // represents the party P_0 in the secure exponentiation protocol.
  //
  // The Create function takes input:
  // (1) A FixedPointElementFactory::Params (fpe_params) for the
  // FixedPointElements that will be exponentiated.
  // (2) An ExponentiationParams (exp_params) which includes a large prime
  // number (prime_q) and a bound (exponent_bound) such that
  // -exponent_bound < x < exponent_bound
  //
  // Returns an INVALID_ARGUMENT error code if prime_q is not a prime.
  // The protocol has undefined behavior if the prime is not large enough
  // to hold the result of the exponentiation or if primes input to the two
  // parties are different.
  static StatusOr<std::unique_ptr<SecureExponentiationPartyZero>> Create(
      const FixedPointElementFactory::Params& fpe_params,
      const ExponentiationParams& exp_params);

  // Generates the protocol message to send to party P_1 as well as state
  // required for the second part of the computation.
  // This message is for the subprotocol that converts multiplicative shares
  // in \Z_{prime_q} to additive shares in \Z_{prime_q}.
  // The function takes as input a FixedPointElement (fpe_share_zero)
  // representing P_0's share of the exponent x and a pre-processed tuple
  // (alpha_zero, beta_zero).
  // The protocol has undefined behavior if the pre-processed input
  // is not correct.
  StatusOr<std::pair<ExponentiationPartyZeroMultToAddMessage, State>>
  GenerateMultToAddMessage(const FixedPointElement& fpe_share_zero,
                           uint64_t alpha_zero, uint64_t beta_zero);

  // This function achieves the same functionality as
  // 'GenerateMultToAddMessage' but for a vector of exponents i.e. when
  // computing e^x where x is a vector.
  StatusOr<std::pair<BatchedExponentiationPartyZeroMultToAddMessage,
                     std::vector<State>>>
  GenerateBatchedMultToAddMessage(
      const std::vector<FixedPointElement>& fpe_shares_zero,
      const std::vector<uint64_t>& alpha_zero,
      const std::vector<uint64_t>& beta_zero);

  // Takes as input, the protocol message from P_1, and SecureExpPartyZero
  // struct message containing the state from GenerateMultToAddMessage and the
  // preprocessed tuple and outputs the final protocol resut which is an
  // additive share of e^x in the ring \Z_{fpe_params_.primary_ring_modulus}.
  // Returns an INVALID_ARGUMENT error code if the message from P_1
  // is malformed.
  // The pre-processed tuple (alpha_zero, beta_zero) should be the same as the
  // one input to GenerateMultToAddMessage that created the
  // interaction message.
  //
  // The protocol has undefined behavior if the two messages were not created
  // from FixedPointElements in the same ring.
  // The protocol also has undefined behavior if the pre-processed tuple is
  // incorrect.
  StatusOr<FixedPointElement> OutputResult(
      const ExponentiationPartyOneMultToAddMessage& party_one_msg,
      const State& self_state);

  // This function achieves the same functionality as 'OutputResult'
  // but for a vector of exponents i.e. when computing e^x where x is a vector
  StatusOr<std::vector<FixedPointElement>> BatchedOutputResult(
      const BatchedExponentiationPartyOneMultToAddMessage& party_one_msg,
      const std::vector<State>& self_state);

 private:
  explicit SecureExponentiationPartyZero(
      std::unique_ptr<FixedPointElementFactory> fpe_factory,
      const FixedPointElementFactory::Params* fpe_params,
      const ExponentiationParams* exp_params,
      std::unique_ptr<FixedPointElement> logbase2_e_fpe,
      std::unique_ptr<FixedPointElement> exp_bound_adder);

  std::unique_ptr<FixedPointElementFactory> fpe_factory_;
  const FixedPointElementFactory::Params* fpe_params_;
  const ExponentiationParams* exp_params_;

  // Useful computations that need to be done only once.
  // The FixedPointElement representation of log_2(e).
  std::unique_ptr<FixedPointElement> logbase2_e_fpe_;
  std::unique_ptr<FixedPointElement> exp_bound_adder_;
  uint64_t two_power_base2_bound_;
};

// Class representing the party P_1 in the secure exponentiation protocol.
class SecureExponentiationPartyOne {
 public:
  // State struct for party P_1 created in the first part of the computation
  // (GenerateMultToAddMessage) and input to the second part (OutputResult).
  struct State {
    uint64_t mult_share_one;  // Mult share resulting from local exponentiation.
    // (alpha_one, beta_one) is the pre-processed tuple.
    uint64_t alpha_one;
    uint64_t beta_one;
  };

  // Returns a std::unique_ptr to a SecureExponentiationPartyOne, which
  // represents the party P_1 in the secure exponentiation protocol.
  //
  // The Create function takes input:
  // (1) A FixedPointElementFactory::Params (fpe_params) for the
  // FixedPointElements that will be exponentiated.
  // (2) An ExponentiationParams (exp_params) which includes a large prime
  // number (prime_q) and a bound (exponent_bound) such that
  // -exponent_bound < x < exponent_bound
  //
  // The protocol has undefined behavior if prime_q is not prime, or if
  // prime_q is not large enough to hold the result of the exponentiation or if
  // primes input to the two parties are different.
  static StatusOr<std::unique_ptr<SecureExponentiationPartyOne>> Create(
      const FixedPointElementFactory::Params& fpe_params,
      const ExponentiationParams& exp_params);

  // Generates the protocol message to send to party P_0 as well as state
  // required for the second part of the computation.
  // This message is for the subprotocol that converts multiplicative shares
  // in \Z_{prime_q} to additive shares in \Z_{prime_q}
  // The function takes as input a FixedPointElement (fpe_share_one)
  // representing P_1's share of the exponent x and a pre-processed tuple
  // (alpha_one, beta_one).
  // The protocol has undefined behavior if the pre-processed input
  // is not correct.
  StatusOr<std::pair<ExponentiationPartyOneMultToAddMessage,
                     State>> GenerateMultToAddMessage(
      const FixedPointElement& fpe_share_one,
      uint64_t alpha_one, uint64_t beta_one);

  // This function achieves the same functionality as
  // 'GenerateMultToAddMessage' but for a vector of exponents i.e. when
  // computing e^x where x is a vector
  StatusOr<std::pair<BatchedExponentiationPartyOneMultToAddMessage,
                     std::vector<State>>>
  GenerateBatchedMultToAddMessage(
      const std::vector<FixedPointElement>& fpe_shares_one,
      const std::vector<uint64_t>& alpha_one,
      const std::vector<uint64_t>& beta_one);

  // Takes as input, the protocol message from P_1, and SecureExpPartyZero
  // struct message containing the state from GenerateMultToAddMessage and the
  // preprocessed tuple and outputs the final protocol resut which is an
  // additive share of e^x in the ring \Z_{fpe_params_.primary_ring_modulus}.
  // Returns an INVALID_ARGUMENT error code if the message from P_0
  // is malformed.
  // The pre-processed tuple (alpha_one, beta_one) should be the same as the
  // one input to GenerateMultToAddMessage that created the
  // interaction message.
  //
  // The protocol has undefined behavior if the two messages were not created
  // from FixedPointElements in the same ring.
  // The protocol also has undefined behavior if the pre-processed tuple is
  // incorrect.
  StatusOr<FixedPointElement> OutputResult(
      const ExponentiationPartyZeroMultToAddMessage& party_zero_msg,
      const State& self_state);

  // This function achieves the same functionality as 'OutputResult'
  // but for a vector of exponents i.e. when computing e^x where x is a vector
  StatusOr<std::vector<FixedPointElement>> BatchedOutputResult(
      const BatchedExponentiationPartyZeroMultToAddMessage& party_zero_msg,
      const std::vector<State>& self_state);

 private:
  explicit SecureExponentiationPartyOne(
      std::unique_ptr<FixedPointElementFactory> fpe_factory,
      const FixedPointElementFactory::Params* fpe_params,
      const ExponentiationParams* exp_params,
      std::unique_ptr<FixedPointElement> logbase2_e_fpe);

  std::unique_ptr<FixedPointElementFactory> fpe_factory_;
  const FixedPointElementFactory::Params* fpe_params_;
  const ExponentiationParams* exp_params_;

  // Useful computations that need to be done only once.
  // The FixedPointElement representation of log_2(e).
  std::unique_ptr<FixedPointElement> logbase2_e_fpe_;
  uint64_t two_power_base2_bound_;
};

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_SECURE_EXPONENTIATION_H_
