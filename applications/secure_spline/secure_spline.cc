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

#include "applications/secure_spline/secure_spline.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "applications/secure_spline/secure_spline.pb.h"
#include "poisson_regression/beaver_triple_messages.pb.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status.inc"
#include "secret_sharing_mpc/gates/hadamard_product.h"
#include "secret_sharing_mpc/gates/scalar_vector_product.h"
#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "dcf/fss_gates/multiple_interval_containment.h"
#include "dcf/fss_gates/multiple_interval_containment.pb.h"
#include "dpf/status_macros.h"

namespace private_join_and_compute {
namespace applications {

using ::private_join_and_compute::FixedPointElement;
using ::private_join_and_compute::FixedPointElementFactory;
using ::distributed_point_functions::fss_gates::MultipleIntervalContainmentGate;

absl::StatusOr<std::unique_ptr<SecureSpline>> SecureSpline::Create(
    uint64_t num_inputs, SecureSplineParameters spline_params) {
  distributed_point_functions::fss_gates::MicParameters mic_parameters;

  // Sanity check of spline params
  //  1. Intervals should be contiguous i.e.
  //  they should be of the form [a, b), [b, c) ... [y, z)
  //  2. Size of lower_bounds_double, upper_bounds_double, slope_double and
  //  yIntercept_double should be exactly same as interval_count

  // Checking whether the size of vector containing lower interval
  // boundaries matches the interval_count
  if (spline_params.lower_bounds_double.size() !=
      spline_params.interval_count) {
    return InvalidArgumentError(
        "Size of lower bounds vector should match interval_count");
  }

  // Checking whether the size of vector containing upper interval
  // boundaries matches the interval_count
  if (spline_params.upper_bounds_double.size() !=
      spline_params.interval_count) {
    return InvalidArgumentError(
        "Size of upper bounds vector should match interval_count");
  }

  // Checking whether the size of vector containing slope values matches
  // the interval_count
  if (spline_params.slope_double.size() != spline_params.interval_count) {
    return InvalidArgumentError(
        "Size of slope vector should match interval_count");
  }

  // Checking whether the size of vector containing yIntercept values matches
  // the interval_count
  if (spline_params.y_intercept_double.size() != spline_params.interval_count) {
    return InvalidArgumentError(
        "Size of yIntercept vector should match interval_count");
  }

  // Checking whether intervals are contiguous
  for (int i = 0; i < spline_params.interval_count - 1; i++) {
    if (spline_params.upper_bounds_double[i] !=
        spline_params.lower_bounds_double[i + 1]) {
      return InvalidArgumentError(
          "Secure Spline intervals should be contiguous");
    }
  }

  // Setting input and output group
  mic_parameters.set_log_group_size(spline_params.log_group_size);

  ASSIGN_OR_RETURN(
      FixedPointElementFactory fixed_point_factory,
      FixedPointElementFactory::Create(spline_params.num_fractional_bits,
                                       spline_params.log_group_size));

  // Converting the spline parameters into fixed point representation
  std::vector<FixedPointElement> lower_bounds_fxp;
  std::vector<FixedPointElement> upper_bounds_fxp;
  lower_bounds_fxp.reserve(spline_params.interval_count);
  upper_bounds_fxp.reserve(spline_params.interval_count);

  for (int i = 0; i < spline_params.interval_count; ++i) {
    ASSIGN_OR_RETURN(FixedPointElement lb,
                     fixed_point_factory.CreateFixedPointElementFromDouble(
                         spline_params.lower_bounds_double[i]));
    ASSIGN_OR_RETURN(FixedPointElement ub,
                     fixed_point_factory.CreateFixedPointElementFromDouble(
                         spline_params.upper_bounds_double[i]));

    lower_bounds_fxp.push_back(lb);
    upper_bounds_fxp.push_back(ub);
  }

  for (int i = 0; i < spline_params.interval_count; ++i) {
    distributed_point_functions::fss_gates::Interval* interval =
        mic_parameters.add_intervals();

    interval->mutable_lower_bound()->mutable_value_uint128()->set_low(
        lower_bounds_fxp[i].ExportToUint64());

    // Since the Multiple interval containment gate checks containment with
    // inclusive boundaries, we subtract one from the upper bound
    interval->mutable_upper_bound()->mutable_value_uint128()->set_low(
        upper_bounds_fxp[i].ExportToUint64() - 1);
  }

  // Creating a MIC gate
  DPF_ASSIGN_OR_RETURN(std::unique_ptr<MultipleIntervalContainmentGate> MicGate,
                       MultipleIntervalContainmentGate::Create(mic_parameters));

  return absl::WrapUnique(
      new SecureSpline(std::move(MicGate), num_inputs,
      absl::make_unique<FixedPointElementFactory>(fixed_point_factory),
      spline_params));
}

SecureSpline::SecureSpline(
    std::unique_ptr<MultipleIntervalContainmentGate> MicGate,
    uint64_t num_inputs,
    std::unique_ptr<FixedPointElementFactory> fixed_point_factory,
    SecureSplineParameters spline_params)
    : fixed_point_factory_(std::move(fixed_point_factory)),
      mic_gate_(std::move(MicGate)),
      num_inputs_(num_inputs),
      spline_params_(spline_params) {}

StatusOr<std::pair<std::vector<SplinePrecomputedValue>,
                   std::vector<SplinePrecomputedValue>>>
SecureSpline::PerformSplinePrecomputation() {

  // The modulus is needed to ensure that all the arithmetic operations happen
  // over the primary ring which might be of size < 2^64
  uint64_t modulus = fixed_point_factory_->GetParams().primary_ring_modulus;

  std::vector<SplinePrecomputedValue> precomputation_party_0;
  std::vector<SplinePrecomputedValue> precomputation_party_1;

  precomputation_party_0.reserve(num_inputs_);
  precomputation_party_1.reserve(num_inputs_);

  for (int i = 0; i < num_inputs_; i++) {
    distributed_point_functions::fss_gates::MicKey key_0, key_1;

    // Initializing the input and output masks uniformly at random;
    const absl::string_view kSampleSeed = absl::string_view();
    ASSIGN_OR_RETURN(auto rng, private_join_and_compute::BasicRng::Create(kSampleSeed));

    uint64_t r_in;

    ASSIGN_OR_RETURN(r_in, rng->Rand64());
    r_in = r_in % modulus;

    std::vector<absl::uint128> r_outs;
    std::vector<uint64_t> r_outs_uint64;

    for (int j = 0; j < spline_params_.interval_count; j++) {
      ASSIGN_OR_RETURN(uint64_t r_out, rng->Rand64());
      r_out = r_out % modulus;
      r_outs.push_back(r_out);
      r_outs_uint64.push_back(r_out);
    }

    // Generating MIC gate keys
    DPF_ASSIGN_OR_RETURN(std::tie(key_0, key_1), mic_gate_->Gen(r_in, r_outs));

    // Generate Beaver triple vector for P0 and P1.
    ASSIGN_OR_RETURN(auto beaver_vector_shares,
                     private_join_and_compute::SampleBeaverTripleVector(
                         1, modulus));
    auto beaver_vector_share_0 = beaver_vector_shares.first;
    auto beaver_vector_share_1 = beaver_vector_shares.second;

    ASSIGN_OR_RETURN(uint64_t mic_input_mask_share_party_0, rng->Rand64());
    mic_input_mask_share_party_0 = mic_input_mask_share_party_0 % modulus;
    uint64_t mic_input_mask_share_party_1 =
        private_join_and_compute::ModSub(r_in, mic_input_mask_share_party_0, modulus);

    // Generate random shares of input and output masks
    ASSIGN_OR_RETURN(auto mic_output_mask_share_party_0,
                     SampleVectorFromPrng(spline_params_.interval_count,
                                          modulus, rng.get()));
    ASSIGN_OR_RETURN(
        auto mic_output_mask_share_party_1,
        private_join_and_compute::BatchedModSub(r_outs_uint64, mic_output_mask_share_party_0,
                                modulus));

    SplinePrecomputedValue party_0{key_0, beaver_vector_share_0,
                                   mic_input_mask_share_party_0,
                                   mic_output_mask_share_party_0};
    SplinePrecomputedValue party_1{key_1, beaver_vector_share_1,
                                   mic_input_mask_share_party_1,
                                   mic_output_mask_share_party_1};

    precomputation_party_0.push_back(party_0);
    precomputation_party_1.push_back(party_1);
  }

  return std::make_pair(std::move(precomputation_party_0),
                        std::move(precomputation_party_1));
}

StatusOr<std::pair<RoundOneSplineState, RoundOneSplineMessage>>
SecureSpline::GenerateSplineRoundOneMessage(
    std::vector<SplinePrecomputedValue> &spline_precomputed_values,
    std::vector<uint64_t> &share_of_spline_inputs) {
  // In round 1, the state and message hold the same content i.e. for Party P_i,
  // state and message both are shares of masked input i.e. [x]_i + [r]_i
  RoundOneSplineMessage round_one_spline_message;
  RoundOneSplineState round_one_spline_state;

  round_one_spline_state.share_of_spline_inputs = share_of_spline_inputs;

  uint64_t modulus = fixed_point_factory_->GetParams().primary_ring_modulus;

  for (int i = 0; i < num_inputs_; i++) {
    // uint64 share_of_spline_input_uint64 =
    // share_of_spline_inputs[i].ExportToUint64();
    uint64_t share_of_masked_input = private_join_and_compute::ModAdd(
        share_of_spline_inputs[i],
        spline_precomputed_values[i].mic_input_mask_share, modulus);

    round_one_spline_state.shares_of_masked_input.push_back(
        share_of_masked_input);
    round_one_spline_message.add_shares_of_masked_input(share_of_masked_input);
  }
  return std::make_pair(round_one_spline_state, round_one_spline_message);
}

StatusOr<std::pair<RoundTwoSplineState, RoundTwoSplineMessage>>
SecureSpline::GenerateSplineRoundTwoMessage(
    int partyid, std::vector<SplinePrecomputedValue> &spline_precomputed_values,
    RoundOneSplineState round_one_state_this_party,
    RoundOneSplineMessage round_one_msg_other_party) {
  uint64_t modulus = fixed_point_factory_->GetParams().primary_ring_modulus;

  // Reconstruct the actual masked input from the shares;
  std::vector<absl::uint128> masked_inputs;
  masked_inputs.reserve(num_inputs_);

  for (int i = 0; i < num_inputs_; i++) {
    uint64_t masked_input = private_join_and_compute::ModAdd(
        round_one_state_this_party.shares_of_masked_input[i],
        round_one_msg_other_party.shares_of_masked_input(i), modulus);
    masked_inputs.push_back(masked_input);
  }

  // Invoke MIC.Eval on the masked inputs to get secret shares of the masked
  // output. Note that the outer vector indexes into one out of the n different
  // outputs (corresponding to each input) whereas the inner vector indexes into
  // one out of the m different outputs (corresponding to each interval for a
  // fixed input). Therefore the outer vector is of length n and the inner
  // vector is of length m.
  std::vector<std::vector<uint64_t>> shares_of_masked_outputs;
  shares_of_masked_outputs.reserve(num_inputs_);

  // new

  // refactor spline precomputed value to avoid this unnecessary copy
  std::vector<const distributed_point_functions::fss_gates::MicKey *> keys(num_inputs_);
  for (size_t idx = 0; idx < num_inputs_; idx++) {
      keys[idx] = &spline_precomputed_values[idx].mic_key;
  }

  // call eval one by one version (much slower)
  /*
  std::vector<absl::uint128> share_of_masked_output;
  for (size_t i = 0; i < num_inputs_; i++) {
      DPF_ASSIGN_OR_RETURN(std::vector<absl::uint128> share_of_masked_output_temp,
                           mic_gate_->Eval(keys[i],
                                           masked_inputs[i]));
      for (size_t j = 0; j < spline_params_.interval_count; j++) {
          share_of_masked_output.push_back(share_of_masked_output_temp[j]);
      }
  }*/

  // batched version
  DPF_ASSIGN_OR_RETURN(std::vector<absl::uint128> share_of_masked_output,
                       mic_gate_->BatchEval(keys, masked_inputs));

  size_t offset = 0;
  for (size_t i = 0; i < num_inputs_; i++) {
        // Static casting uint128 (the output type of MIC gate) into uint64 (the
        // input type of Secret Sharing MPC codebase).
        std::vector<uint64_t> share_of_masked_output_uint64;
        share_of_masked_output_uint64.reserve(spline_params_.interval_count);
        for (int j = 0; j < spline_params_.interval_count; j++)
            share_of_masked_output_uint64.push_back(
                    static_cast<uint64_t>(share_of_masked_output[offset + j]));

        shares_of_masked_outputs.push_back(share_of_masked_output_uint64);
        offset += spline_params_.interval_count;
  }


//  old (not batched)
//  for (int i = 0; i < num_inputs_; i++) {
//    DPF_ASSIGN_OR_RETURN(std::vector<absl::uint128> share_of_masked_output,
//                         mic_gate_->Eval(spline_precomputed_values[i].mic_key,
//                                         masked_inputs[i]));
//
//    // Static casting uint128 (the output type of MIC gate) into uint64 (the
//    // input type of Secret Sharing MPC codebase).
//    std::vector<uint64_t> share_of_masked_output_uint64;
//    share_of_masked_output_uint64.reserve(spline_params_.interval_count);
//    for (int j = 0; j < spline_params_.interval_count; j++)
//      share_of_masked_output_uint64.push_back(
//          static_cast<uint64_t>(share_of_masked_output[j]));
//
//    shares_of_masked_outputs.push_back(share_of_masked_output_uint64);
//  }

  // Converting shares of masked output to shares of actual output.
  std::vector<std::vector<uint64_t>> shares_of_actual_outputs;
  shares_of_actual_outputs.reserve(num_inputs_);

  for (int i = 0; i < num_inputs_; i++) {
    ASSIGN_OR_RETURN(
        std::vector<uint64_t> shares_of_actual_output,
        private_join_and_compute::BatchedModSub(
            shares_of_masked_outputs[i],
            spline_precomputed_values[i].mic_output_mask_shares, modulus));
    shares_of_actual_outputs.push_back(shares_of_actual_output);
  }

  // Performing [a_1, .... a_m] . x_i for all inputs [x_1, ....  x_n].
  /*std::vector<std::vector<uint64_t>> degree_1_coeff_times_inputs;
  degree_1_coeff_times_inputs.reserve(num_inputs_);

  for (int i = 0; i < num_inputs_; i++) {
    std::vector<uint64_t> degree_1_coeff_times_input_i;
    degree_1_coeff_times_input_i.reserve(spline_params_.interval_count);

    for (int j = 0; j < spline_params_.interval_count; j++) {
      std::vector<uint64_t> degree_1_coeff_interval_j_times_input_i;
      if (partyid == 0) {
        // Share of party 0.
        ASSIGN_OR_RETURN(
            degree_1_coeff_interval_j_times_input_i,
            private_join_and_compute::ScalarVectorProductPartyZero(
                spline_params_.slope_double[j],
                std::vector<uint64_t>{
                    round_one_state_this_party.share_of_spline_inputs[i]},
                fixed_point_factory_, modulus));
      } else {
        // Share of party 1.
        ASSIGN_OR_RETURN(
            degree_1_coeff_interval_j_times_input_i,
            private_join_and_compute::ScalarVectorProductPartyOne(
                spline_params_.slope_double[j],
                std::vector<uint64_t>{
                    round_one_state_this_party.share_of_spline_inputs[i]},
                fixed_point_factory_, modulus));
      }
      degree_1_coeff_times_input_i.push_back(
          degree_1_coeff_interval_j_times_input_i[0]);
    }
    degree_1_coeff_times_inputs.push_back(degree_1_coeff_times_input_i);
  }*/

  // Converting degree 0 coeff into Ring elements.
  /*std::vector<uint64_t> yIntercept_ring;
  for (int i = 0; i < spline_params_.interval_count; ++i) {
    ASSIGN_OR_RETURN(FixedPointElement yI,
                     fixed_point_factory_->CreateFixedPointElementFromDouble(
                         spline_params_.y_intercept_double[i]));
    yIntercept_ring.push_back(yI.ExportToUint64());
  }*/

  /*// Adding degree 0 coefficients i.e. Performing [a_1. x_i, ... a_m. x_i] +
  // [b_1, ..., b_m] for all inputs [x_1, ... x_n].
  std::vector<std::vector<uint64_t>>
      degree_1_coeff_times_inputs_plus_degree_0_coeff;

  for (int i = 0; i < num_inputs_; i++) {
    std::vector<uint64_t> degree_1_coeff_times_input_i_plus_degree_0_coeff;
    if (partyid == 0) {
      // Party 0 adds the degree 0 coefficients locally to its share.
      ASSIGN_OR_RETURN(degree_1_coeff_times_input_i_plus_degree_0_coeff,
                       private_join_and_compute::BatchedModAdd(degree_1_coeff_times_inputs[i],
                                               yIntercept_ring, modulus));
    } else {
      // Party 1 adds nothing to its share.
      ASSIGN_OR_RETURN(
          degree_1_coeff_times_input_i_plus_degree_0_coeff,
          private_join_and_compute::BatchedModAdd(
              degree_1_coeff_times_inputs[i],
              std::vector<uint64_t>(spline_params_.interval_count, 0),
              modulus));
    }
    degree_1_coeff_times_inputs_plus_degree_0_coeff.push_back(
        degree_1_coeff_times_input_i_plus_degree_0_coeff);
  }*/

  // Secret x which is some integer (either 0 or 1).

  // Secret y which is a floating point number (e.g. y = 3.456).

  // Compute the product of x * y given that
  // P0 has x_0 and y_0, P1 has x_1 and y_1 s.t
  // (x_0 + x_1) = x (mod 2^l)
  // (y_0 + y_1) = y * 2^(l_f) (mod 2^l)

  // Multiply x_0 and x_1 by 2^l_f
  // i.e. x_0' = (x_0 * 2^l_f) mod (2^l)
  // and x_1' = (x_1 * 2^l_f) mod (2^l)
  // So now x_0' and x_1' are secret shares of the fixed point reprentation of
  // x. i.e. x_0 + x_1 = x * 2^l_f (mod 2^l)

  // Convert shares of actual output of spline into shares of
  // output in FixedPoint representation (by multiplying the share value
  // locally by 2^lf).
  std::vector<std::vector<uint64_t>> shares_of_actual_spline_outputs_in_fxp;
  shares_of_actual_spline_outputs_in_fxp.reserve(num_inputs_);

  std::vector<uint64_t> fractional_multiplier_vector(
      spline_params_.interval_count,
      fixed_point_factory_->GetParams().fractional_multiplier);
  for (int i = 0; i < num_inputs_; i++) {
    ASSIGN_OR_RETURN(
        std::vector<uint64_t> shares_of_actual_outputs_in_fxp_i,
        private_join_and_compute::BatchedModMul(shares_of_actual_outputs[i],
                                fractional_multiplier_vector, modulus));
    shares_of_actual_spline_outputs_in_fxp.push_back(
        shares_of_actual_outputs_in_fxp_i);
  }

  // Let t be the idx where the spline output is 1
  // Let [d] be shares_of_actual_spline_outputs_in_fxp for a single example

  // Compute [a_t] <- a_1[d_1] + ... + a_n [d_n]

  // we can do as n scalar-vector products where the vectors are of size 1
  // and add them (repeat for each example)
  std::vector<uint64_t> share_active_degree_1_coeff (num_inputs_);

  for (int i = 0; i < num_inputs_; i++) {
    for (int j = 0; j < spline_params_.interval_count; j++) {
      std::vector<uint64_t> degree_1_coeff_times_spline_output_interval_j_input_i;
      if (partyid == 0) {
        // Share of party 0.
        ASSIGN_OR_RETURN(
            degree_1_coeff_times_spline_output_interval_j_input_i,
            private_join_and_compute::ScalarVectorProductPartyZero(
                spline_params_.slope_double[j],
                std::vector<uint64_t>{
                    shares_of_actual_spline_outputs_in_fxp[i][j]},
                fixed_point_factory_, modulus));
      } else {
        // Share of party 1.
        ASSIGN_OR_RETURN(
            degree_1_coeff_times_spline_output_interval_j_input_i,
            private_join_and_compute::ScalarVectorProductPartyOne(
                spline_params_.slope_double[j],
                std::vector<uint64_t>{
                    shares_of_actual_spline_outputs_in_fxp[i][j]},
                fixed_point_factory_, modulus));
      }

      share_active_degree_1_coeff[i] = private_join_and_compute::ModAdd(
          share_active_degree_1_coeff[i], degree_1_coeff_times_spline_output_interval_j_input_i[0],
          fixed_point_factory_->GetParams().primary_ring_modulus);
    }
  }

  // Compute [b_t] <- b_1[d_1] + ... + b_n [d_n]
  std::vector<uint64_t> share_active_degree_0_coeff (num_inputs_);

  for (size_t i = 0; i < num_inputs_; i++) {
    for (size_t j = 0; j < spline_params_.interval_count; j++) {
      std::vector<uint64_t> degree_0_coeff_times_spline_output_interval_j_input_i;
      if (partyid == 0) {
        // Share of party 0.
        ASSIGN_OR_RETURN(
            degree_0_coeff_times_spline_output_interval_j_input_i,
            private_join_and_compute::ScalarVectorProductPartyZero(
                spline_params_.y_intercept_double[j],
                std::vector<uint64_t>{
                    shares_of_actual_spline_outputs_in_fxp[i][j]},
                fixed_point_factory_, modulus));
      } else {
        // Share of party 1.
        ASSIGN_OR_RETURN(
            degree_0_coeff_times_spline_output_interval_j_input_i,
            private_join_and_compute::ScalarVectorProductPartyOne(
                spline_params_.y_intercept_double[j],
                std::vector<uint64_t>{
                    shares_of_actual_spline_outputs_in_fxp[i][j]},
                fixed_point_factory_, modulus));
      }

      share_active_degree_0_coeff[i] = private_join_and_compute::ModAdd(
          share_active_degree_0_coeff[i], degree_0_coeff_times_spline_output_interval_j_input_i[0],
          fixed_point_factory_->GetParams().primary_ring_modulus);
    }
  }

  // Now we have a vector of a_t's
  // and a vector of x's (spline inputs)
  // Compute a hadamard product to get a_t * x
  // (I.e. prepare state and message and compute in the next round)
  RoundTwoSplineMessage round_two_spline_msg;
  std::vector<private_join_and_compute::BatchedMultState> hadamard_state;

  for (size_t idx = 0; idx < num_inputs_; idx++) {
    // Each party generates its batched multiplication message.
    std::pair<private_join_and_compute::BatchedMultState, private_join_and_compute::MultiplicationGateMessage>
        hadamard_state_plus_msg;
    ASSIGN_OR_RETURN(
        hadamard_state_plus_msg,
        GenerateHadamardProductMessage(
            std::vector<uint64_t>{share_active_degree_1_coeff[idx]},
            std::vector<uint64_t>{round_one_state_this_party.share_of_spline_inputs[idx]},
            spline_precomputed_values[idx].beaver_triple, modulus));
    hadamard_state.push_back(
        hadamard_state_plus_msg.first);
    *(round_two_spline_msg.add_hadamard_product()) =
        hadamard_state_plus_msg.second;
  }

  RoundTwoSplineState round_two_spline_state = {
      .hadamard_state = std::move(hadamard_state),
      .share_active_degree_0_coeff = std::move(share_active_degree_0_coeff)
  };


  // Computing hadamard product between the output of spline and evaluation of
  // single degree polynomial on the inputs in all intervals.
  /*RoundTwoSplineState round_two_spline_state;
  RoundTwoSplineMessage round_two_spline_msg;

  for (int i = 0; i < num_inputs_; i++) {
    // Hadamard product between
    // degree_1_coeff_times_inputs_plus_degree_0_coeff[i] and
    // shares_of_actual_spline_outputs_in_fxp[i].

    // Each party generates its batched multiplication message.
    std::pair<private_join_and_compute::BatchedMultState, private_join_and_compute::MultiplicationGateMessage>
        hadamard_state_plus_msg;
    ASSIGN_OR_RETURN(
        hadamard_state_plus_msg,
        GenerateHadamardProductMessage(
            degree_1_coeff_times_inputs_plus_degree_0_coeff[i],
            shares_of_actual_spline_outputs_in_fxp[i],
            spline_precomputed_values[i].hadamard_triple, modulus));
    round_two_spline_state.hadamard_state.push_back(
        hadamard_state_plus_msg.first);
    *(round_two_spline_msg.add_hadamard_product()) =
        hadamard_state_plus_msg.second;
  }*/

  return std::make_pair(round_two_spline_state, round_two_spline_msg);
}

StatusOr<std::vector<uint64_t>> SecureSpline::GenerateSplineResult(
    int partyid, std::vector<SplinePrecomputedValue> &spline_precomputed_values,
    RoundTwoSplineState round_two_state_this_party,
    RoundTwoSplineMessage round_two_msg_other_party) {
  std::vector<uint64_t> partial_spline_outputs_share (num_inputs_);
  for (size_t idx = 0; idx < num_inputs_; idx++) {
    if (partyid == 0) {
      // Execute HadamardProductPartyZero.
      ASSIGN_OR_RETURN(
          auto partial_spline_outputs_share_temp,
          private_join_and_compute::HadamardProductPartyZero(
              round_two_state_this_party.hadamard_state[idx],
              spline_precomputed_values[idx].beaver_triple,
              round_two_msg_other_party.hadamard_product(idx),
              fixed_point_factory_->GetParams().num_fractional_bits,
              fixed_point_factory_->GetParams().primary_ring_modulus));
      partial_spline_outputs_share[idx] = partial_spline_outputs_share_temp[0];
    } else {
      // Execute HadamardProductPartyOne.
      ASSIGN_OR_RETURN(
          auto partial_spline_outputs_share_temp,
          private_join_and_compute::HadamardProductPartyOne(
              round_two_state_this_party.hadamard_state[idx],
              spline_precomputed_values[idx].beaver_triple,
              round_two_msg_other_party.hadamard_product(idx),
              fixed_point_factory_->GetParams().num_fractional_bits,
              fixed_point_factory_->GetParams().primary_ring_modulus));
      partial_spline_outputs_share[idx] = partial_spline_outputs_share_temp[0];
    }
  }

  // Add in b_t (active degree 0 coeff)

  std::vector<uint64_t> final_spline_outputs_share (num_inputs_);
  for (size_t i = 0; i < num_inputs_; i++) {
    uint64_t output_i_share = private_join_and_compute::ModAdd(
          round_two_state_this_party.share_active_degree_0_coeff[i], partial_spline_outputs_share[i],
          fixed_point_factory_->GetParams().primary_ring_modulus);
    final_spline_outputs_share[i] = output_i_share;
  }

  /*
  for (int i = 0; i < num_inputs_; i++) {
    std::vector<uint64_t> final_spline_output_i_share;
    final_spline_output_i_share.reserve(spline_params_.interval_count);

    if (partyid == 0) {
      // Execute HadamardProductPartyZero.
      ASSIGN_OR_RETURN(
          final_spline_output_i_share,
          private_join_and_compute::HadamardProductPartyZero(
              round_two_state_this_party.hadamard_state[i],
              spline_precomputed_values.hadamard_triple,
              round_two_msg_other_party.hadamard_product(i),
              fixed_point_factory_->GetParams().num_fractional_bits,
              fixed_point_factory_->GetParams().primary_ring_modulus));
    } else {
      // Execute HadamardProductPartyOne.
      ASSIGN_OR_RETURN(
          final_spline_output_i_share,
          private_join_and_compute::HadamardProductPartyOne(
              round_two_state_this_party.hadamard_state[i],
              spline_precomputed_values.hadamard_triple,
              round_two_msg_other_party.hadamard_product(i),
              fixed_point_factory_->GetParams().num_fractional_bits,
              fixed_point_factory_->GetParams().primary_ring_modulus));
    }
    final_spline_outputs_share.push_back(final_spline_output_i_share);
  }

  std::vector<uint64_t> res;
  res.reserve(num_inputs_);

  for (int i = 0; i < num_inputs_; i++) {
    uint64_t output_i_share = 0;
    for (int j = 0; j < spline_params_.interval_count; j++) {
      output_i_share = private_join_and_compute::ModAdd(
          output_i_share, final_spline_outputs_share[i][j],
          fixed_point_factory_->GetParams().primary_ring_modulus);
    }
    res.push_back(output_i_share);
  }*/

  return final_spline_outputs_share;
}

}  // namespace applications
}  // namespace private_join_and_compute
