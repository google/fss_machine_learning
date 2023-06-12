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

#ifndef PRIVATE_JOIN_AND_COMPUTE_APPLICATIONS_SECURE_SPLINE_SECURE_SPLINE_H_
#define PRIVATE_JOIN_AND_COMPUTE_APPLICATIONS_SECURE_SPLINE_SECURE_SPLINE_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "applications/secure_spline/secure_spline.pb.h"
#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "absl/numeric/int128.h"
#include "dcf/fss_gates/multiple_interval_containment.h"
#include "dcf/fss_gates/multiple_interval_containment.pb.h"

namespace private_join_and_compute {
namespace applications {

// Implements the Secure Spline functionality between 2 parties using a
// combination of Multiple Interval Gate (from Function Secret Sharing paradigm)
// and Hadamard Product (from Secret Sharing based MPC). A spline is a function
// defined piecewise by polynomials. Formally, consider an interval [p, q)
// which is split into m contiguous intervals: { [p_1, p_2), [p_2, p_3) ....
// [p_m, p_(m+1)) } where p_1 = p and p_(m+1) = q. Now consider m different
// d-degree polynomials {f_1, ..., f_m}. Then the spline functionality on input
// x returns the evaluation of polynomial f_i(x) where x lies in the interval
// [p_i, p_(i+1)).

// In the context of secure computation using 2 parties, party P0 and P1 hold
// a secret share x_0 and x_1 of the actual input x respectively. Using secure
// spline functionality, they can interact and compute secret shares y_0 and y_1
// of the actual output y  = f(x) where f is the output of spline functionality
// on the input x. Note that the specification of spline is public which
// includes the intervals and coefficients of polynomials. It is only the input
// and output that is kept in secret (shared form).

// Note that this file implements a special case of spline functionality where
// all piece-wise polynomials are restricted to degree 1 polynomials. In other
// words, each polynomial p_i is a line and can be specified using 2 parameters:
// slope (a_i) and yIntercept (b_i).


// Also note that this file implements the batched version of spline where
// the spline is evaluated on a vector of inputs X = [x_1, ...., x_n] in
// parallel and produces a vector of outputs Y = [y_1, ... ,y_n].

using ::distributed_point_functions::fss_gates::MultipleIntervalContainmentGate;

// In our actual application, invoke sigmoid on a bunch of inputs
// [x_1, .... x_n].

// This in turn will require spline on [x_1, .... x_n].

// Input : x

// Public a = [a_1, a_2, ..... a_m]
// Secret x_i
// Public b = [b_1, b_2, ..... b_m]

// Secret t = [t_1, t_2, ..... t_m] -> Output of MIC gate

// Compute a.x_i + b = [a_1.x_i + b_1, a_2.x_i + b_2, .... a_m.x_i + b_m]

// Compute t.(a.x_i + b) = [t_1. (a_1.x_i + b_1), t_2. (a_2.x_i + b_2), .... t_m
// . (a_m.x_i + b_m)]

// Precomputations(split into Create and Precompute):

// Create:
// - Generate/Hardcode spline parameters (i.e a, b and intervals).
// - Create MIC gate using intervals.

// Precompute :
// - Create input and output masks for MIC gate (using SampleVectorFromPrng in
// beaver_triple_utils).
// - Invoke MIC.Gen using input r_in and output masks r_out.
// - Generate Beaver triples using SampleBeaverTripleVector in
// hadamard_product.h (to be used for Step 6)

// [1 round] Step 1 : Create masked input (i.e. x + r_in)
// - Add [x] + [r_in] for both players (using VectorAdd in vector_addition).
// - Create proto for storing the [x] + [r_in].
// - Use BatchedModAdd in ring_arithmetic_utils to reconstruct shares for each
// party.

// [Local] Step 2 : Invoke MIC.Eval on the masked input and get masked shares of
// t (i.e. [t + r_out]) for each party.

// [Local] Step 3 : Subtract [r_out] from [t + r_out] for each party (using
// VectorSubtract in vector_subtraction).

// [Local] Step 4 : Compute Secret scalar and Public vector product between s
// and a -> call scalar_vector_product n times (with vector of size = 1).

//  [Local] Step 5 : Compute a_t where t is the active interval with scalar-vector products (where vector is size 1)


//  [Local] Step 6: Similarly compute b_t

// [1 round] Step 7 : Compute a_t * x.

// [Local] Step 8: Compute a_t * x + b_t

// A single SplinePrecomputedValues will be used for a single input x_i
// Since we have `n` different inputs in a batch - [x_1,...., x_n] - we will
// need to generate n SplinePrecomputedValues for each party.
struct SplinePrecomputedValue {
  // A key for evaluating Multiple Interval Containment Gate.
  distributed_point_functions::fss_gates::MicKey mic_key;

  // Beaver triples (vector of size 1).
  private_join_and_compute::BeaverTripleVector<uint64_t> beaver_triple;

  // Secret share of input mask.
  uint64_t mic_input_mask_share;

  // Secret share of output mask. We need m different output mask shares - one
  // for each interval.
  std::vector<uint64_t> mic_output_mask_shares;
};


// RoundOneSplineState is the local state of party P_i after executing Round
// 1 of the Secure Spline protocol.
struct RoundOneSplineState {
  // Secret share of masked input. We will have n shares - one for each input
  // x_i in the batch.
  std::vector<uint64_t> shares_of_masked_input;

  // Secret share of the actual input x_i. We will have n shares - one for
  // each input x_i in the batch.
  std::vector<uint64_t> share_of_spline_inputs;
};

// RoundTwoSplineState is the local state of party P_i after executing Round
// 2 of the Secure Spline protocol.
struct RoundTwoSplineState {
  // hadamard_state is a BatchedMultState wrapper.
  // BatchedMultState stores a pair
  // of vector differences share (which is basically a vectorized version of
  // what each party reveals during a normal beaver multiplication)
  std::vector<BatchedMultState> hadamard_state;
  std::vector<uint64_t> share_active_degree_0_coeff; // b_t * [d_t], where t is active interval and d is spline output
};

// Contains information needed to fully specify a spline restricted to
// degree-1 polynomials
struct SecureSplineParameters {
  // Input and output bit size of the ring that spline will operate on.
  int log_group_size;

  // Number of intervals.
  uint64_t interval_count;

  // Number of bits of precision needed for fractional values.
  int64_t num_fractional_bits;

  // A vector of length = interval_count storing the lower boundaries for
  // all the intervals
  std::vector<double> lower_bounds_double;

  // A vector of length = interval_count storing the upper boundaries for
  // all the intervals.
  std::vector<double> upper_bounds_double;

  // A vector of length = interval_count storing the slope for
  // all the intervals.
  std::vector<double> slope_double;

  // A vector of length = interval_count storing the y Intercept for
  // all the intervals.
  std::vector<double> y_intercept_double;
};

class SecureSpline {
 public:
  // Factory method : creates and returns a SecureSpline initialized with
  // appropriate parameters.
  static absl::StatusOr<std::unique_ptr<SecureSpline>> Create(
      uint64_t num_inputs, SecureSplineParameters spline_params);

  // Performs precomputation stage of spline and returns a pair of
  // n SplinePrecomputedValues - one for each party for each input x_i.
  StatusOr<std::pair<std::vector<SplinePrecomputedValue>,
                     std::vector<SplinePrecomputedValue>>>
  PerformSplinePrecomputation();

  // Performs all the local computations between precomputation and Round 1
  // and finally outputs a local state RoundOneSplineState and
  // a message RoundOneSplineMessage for party P_i.
  StatusOr<std::pair<RoundOneSplineState, RoundOneSplineMessage>>
  GenerateSplineRoundOneMessage(
      std::vector<SplinePrecomputedValue> &spline_precomputed_values,
      std::vector<uint64_t> &share_of_spline_inputs);

  // Performs all the local computations between Round 1 and Round 2 using
  // RoundOneSplineState of party P_i and RoundOneSplineMessage of party P_(1-i)
  // , and finally outputs a local state RoundTwoSplineState and
  // a message RoundTwoSplineMessage for party P_i.
  StatusOr<std::pair<RoundTwoSplineState, RoundTwoSplineMessage>>
  GenerateSplineRoundTwoMessage(
      int partyid,
      std::vector<SplinePrecomputedValue> &spline_precomputed_values,
      RoundOneSplineState round_one_state_this_party,
      RoundOneSplineMessage round_one_msg_other_party);

  // Performs all the local computations needed after Round 2, using
  // RoundTwoSplineState of party P_i and RoundTwoSplineMessage for party
  // P_(1-i)
  //, and finally outputs a secret share of the final spline output for
  // party P_i. Since we are operating in a batched setting, the output is
  // actually a n-length vector - one output corresponding to each input in
  // the batch.
  StatusOr<std::vector<uint64_t>> GenerateSplineResult(
      int partyid,
      std::vector<SplinePrecomputedValue> &spline_precomputed_values,
      RoundTwoSplineState round_two_state_this_party,
      RoundTwoSplineMessage round_two_msg_other_party);

  // Pointer to Fixed Point factory for converting between double and
  // fixed point representation.
  std::unique_ptr<FixedPointElementFactory> fixed_point_factory_;

 private:
  // Private constructor, called by `Create`.
  SecureSpline(std::unique_ptr<MultipleIntervalContainmentGate> MicGate,
               uint64_t num_inputs,
               std::unique_ptr<FixedPointElementFactory> fixed_point_factory,
               SecureSplineParameters spline_params);

  // SecureSpline is neither copyable nor movable.
  SecureSpline(const SecureSpline&) = delete;
  SecureSpline& operator=(const SecureSpline&) = delete;

  // A pointer to a Multiple Interval Containment gate.
  const std::unique_ptr<MultipleIntervalContainmentGate> mic_gate_;

  // Number of inputs in the batch.
  const uint64_t num_inputs_;

  // Parameters for spline specification.
  const SecureSplineParameters spline_params_;
};

}  // namespace applications
}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_APPLICATIONS_SECURE_SPLINE_SECURE_SPLINE_H_
