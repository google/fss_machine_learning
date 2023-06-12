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

#include "secure_comparison.h"

#include "secret_sharing_mpc/gates/hadamard_product.h"
#include "secret_sharing_mpc/gates/vector_addition.h"
#include "secret_sharing_mpc/gates/vector_subtraction.h"

#include <cmath>
#include <iostream>

namespace private_join_and_compute {
namespace secure_comparison {

namespace internal {

// Preprocessing phase for secure comparison
// Insecure - trusted dealer function
StatusOr<std::tuple<ComparisonEqualityGates,
                   ComparisonPreprocessedValues,
                   ComparisonPreprocessedValues>>
ComparisonPrecomputation(size_t batch_size, size_t block_length, size_t num_splits) {

  if (block_length * num_splits != 64) {
    return InvalidArgumentError("block_length * num_splits must equal 64.");
  }

  // Generate Beaver triples for the log_2 num_splits rounds of multiplication mod 2
  // log 2 q (where q is num_splits) rounds, round i:
  // round i: q/2^i * batch_size triples needed
  // the number of triples needed halves at each round
  size_t num_rounds = log2(num_splits);
  std::vector<BeaverTripleVector<uint64_t>> beaver_vector_shares_p0;
  std::vector<BeaverTripleVector<uint64_t>> beaver_vector_shares_p1;
  for (size_t idx = num_rounds; idx >= 1; idx--) {
    // Size of the Beaver triple at this round
    size_t num_triples = (num_splits / pow(2,idx)) * batch_size;
    // The last round requires only one half of triples so handle separately
    if (idx != num_rounds) {
      num_triples *= 2;
    }
    // Generate the Beaver triple
    ASSIGN_OR_RETURN(auto beaver_vector_shares,
        SampleBeaverTripleVector(num_triples, 2));

    beaver_vector_shares_p0.push_back(beaver_vector_shares.first);
    beaver_vector_shares_p1.push_back(beaver_vector_shares.second);
  }

  // Generate input masks for the FSS gates and secret share them between parties

  std::vector<std::vector<uint64_t>> input_mask_short_comparison_p0 (
      batch_size,
      std::vector<uint64_t> (num_splits));
  std::vector<std::vector<uint64_t>> input_mask_short_equality_p0 (
      batch_size,
      std::vector<uint64_t> (num_splits));

  std::vector<std::vector<uint64_t>> input_mask_short_comparison_p1 (
      batch_size,
      std::vector<uint64_t> (num_splits));
  std::vector<std::vector<uint64_t>> input_mask_short_equality_p1 (
      batch_size,
      std::vector<uint64_t> (num_splits));

  // Now fill in the above
  // Initializing the input masks uniformly at random;
  const absl::string_view kSampleSeed = absl::string_view();
  ASSIGN_OR_RETURN(auto rng, private_join_and_compute::BasicRng::Create(kSampleSeed));
  size_t num_variables = 4; // 4 masks to fill for each comparison in batch and for each split
  size_t inp_modulus_cmp = (1ULL << (block_length + 1));
  size_t inp_modulus_eq = (1ULL << block_length);
  for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
    for (size_t idx_split = 0; idx_split < num_splits; idx_split++) {
      std::vector<uint64_t> randomness (num_variables);
      for (size_t idx = 0; idx < num_variables; idx++) {
        ASSIGN_OR_RETURN(randomness[idx], rng->Rand64());
      }
      input_mask_short_comparison_p0[idx_batch][idx_split] = randomness[0] % inp_modulus_cmp;
      input_mask_short_equality_p0[idx_batch][idx_split] = randomness[1] % inp_modulus_eq;
      input_mask_short_comparison_p1[idx_batch][idx_split] = randomness[2] % inp_modulus_cmp;
      input_mask_short_equality_p1[idx_batch][idx_split] = randomness[3] % inp_modulus_eq;
    }
  }

  InputShortComparisonEqualityMasks masks_p0 = {
      .input_mask_short_comparison = std::move(input_mask_short_comparison_p0),
      .input_mask_short_equality = std::move(input_mask_short_equality_p0)
  };

  InputShortComparisonEqualityMasks masks_p1 = {
      .input_mask_short_comparison = std::move(input_mask_short_comparison_p1),
      .input_mask_short_equality = std::move(input_mask_short_equality_p1)
  };

  // Initialize cmp and eq gates parameters
  distributed_point_functions::fss_gates::CmpParameters cmp_parameters;
  cmp_parameters.set_log_group_size(block_length + 1); // gate domain must be 1 bit larger

  // Creating a Comparison gate
  DPF_ASSIGN_OR_RETURN(
      std::unique_ptr<distributed_point_functions::fss_gates::ComparisonGate> CmpGate,
      distributed_point_functions::fss_gates::ComparisonGate::Create(cmp_parameters));

  distributed_point_functions::fss_gates::EqParameters eq_parameters;
  eq_parameters.set_log_group_size(block_length);

  // Creating an Equality gate
  DPF_ASSIGN_OR_RETURN(
      std::unique_ptr<distributed_point_functions::fss_gates::EqualityGate> EqGate,
      distributed_point_functions::fss_gates::EqualityGate::Create(eq_parameters));

  // TODO Do when FSS gates are implemented, invoke Gen

  std::vector<std::vector<distributed_point_functions::fss_gates::CmpKey>> comparison_key_shares_p0 (batch_size);
  std::vector<std::vector<distributed_point_functions::fss_gates::CmpKey>> comparison_key_shares_p1 (batch_size);
  std::vector<std::vector<distributed_point_functions::fss_gates::EqKey>> equality_key_shares_p0 (batch_size);
  std::vector<std::vector<distributed_point_functions::fss_gates::EqKey>> equality_key_shares_p1 (batch_size);

  size_t out_modulus = 2;

  for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
    for (size_t idx_split = 0; idx_split < num_splits; idx_split++) {
      // Generate Comparison output mask
      ASSIGN_OR_RETURN(uint64_t r_out_cmp, rng->Rand64());
      r_out_cmp = r_out_cmp % out_modulus;

      // Generating Comparison gate keys
      distributed_point_functions::fss_gates::CmpKey cmp_key_0, cmp_key_1;
      DPF_ASSIGN_OR_RETURN(std::tie(cmp_key_0, cmp_key_1),
                               CmpGate->Gen(masks_p0.input_mask_short_comparison[idx_batch][idx_split],
                                            masks_p1.input_mask_short_comparison[idx_batch][idx_split],
                                            r_out_cmp));

      comparison_key_shares_p0[idx_batch].push_back(cmp_key_0);
      comparison_key_shares_p1[idx_batch].push_back(cmp_key_1);

      // Generate Equality output mask
      ASSIGN_OR_RETURN(uint64_t r_out_eq, rng->Rand64());
      r_out_eq = r_out_eq % out_modulus;

      // Generating Equality gate keys
      distributed_point_functions::fss_gates::EqKey eq_key_0, eq_key_1;
      DPF_ASSIGN_OR_RETURN(std::tie(eq_key_0, eq_key_1),
                               EqGate->Gen(masks_p0.input_mask_short_equality[idx_batch][idx_split],
                                            masks_p1.input_mask_short_equality[idx_batch][idx_split],
                                            r_out_eq));
      equality_key_shares_p0[idx_batch].push_back(eq_key_0);
      equality_key_shares_p1[idx_batch].push_back(eq_key_1);
    }
  }

  ShortComparisonEqualityKeyShares key_shares_p0 = {
      .comparison_key_shares = std::move(comparison_key_shares_p0),
      .equality_key_shares = std::move(equality_key_shares_p0)
  };
  ShortComparisonEqualityKeyShares key_shares_p1 = {
      .comparison_key_shares = std::move(comparison_key_shares_p1),
      .equality_key_shares = std::move(equality_key_shares_p1)
  };

  ComparisonEqualityGates short_gates = {
      .CmpGate = std::move(CmpGate),
      .EqGate = std::move(EqGate)
  };
  ComparisonPreprocessedValues preprocessed_values_p0 = {
      .key_shares = std::move(key_shares_p0),
      .masks = std::move(masks_p0),
      .beaver_vector_shares = std::move(beaver_vector_shares_p0)
  };
  ComparisonPreprocessedValues preprocessed_values_p1 = {
      .key_shares = std::move(key_shares_p1),
      .masks = std::move(masks_p1),
      .beaver_vector_shares = std::move(beaver_vector_shares_p1)
  };
  return std::make_tuple(
      std::move(short_gates),
      std::move(preprocessed_values_p0),
      std::move(preprocessed_values_p1));
}

}  // namespace internal

// In the paper: q=num_splits, x=inp
// Parses x = x_{q-1} || ... || x_0
StatusOr<std::pair<uint64_t, uint64_t>> parse_shared_input(uint64_t inp,
    uint64_t modulus, size_t num_ring_bits) {
  uint64_t first_bit, remaining_block;
  // remaining_block = ((1ULL << 63) - 1) & inp;
  // inp >>= 63;

  // inp is l bits = 1 + num_ring_bits, we want to chop it off into 1 bit MSB, rest l - 1 bits

  // 1st bit MSB : inp >> (l - 1)

  // rest l-1 bits : inp & (0001111) =  2 ^ (l - 1) - 1 : l-1 ones LSB and rest 0s

  remaining_block = (modulus / 2 - 1) & inp;
  inp >>= (num_ring_bits - 1);
  first_bit = inp;
  return std::make_pair(first_bit, remaining_block);
}

StatusOr<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
SecretSharedComparisonPrepareInputsPartyZero(
    const std::vector<uint64_t>& share_x,
    const std::vector<uint64_t>& share_y,
    size_t batch_size,
    uint64_t modulus,
    size_t num_ring_bits) {
  // Check dimensions of inputs
  if (share_x.empty() || share_y.empty() ||
      share_x.size() != batch_size ||
      share_y.size() != batch_size) {
    return InvalidArgumentError("Parameters have invalid dimensions.");
  }

  // P_i locally computes y_i - x_i
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_y_minus_x,
      VectorSubtract(share_y, share_x, modulus));

  // Parse the share of y_i - x_i
  std::vector<uint64_t> first_bit_shares (batch_size);
  std::vector<uint64_t> remaining_block_shares (batch_size);
  for (size_t idx = 0; idx < batch_size; idx++) {
    ASSIGN_OR_RETURN(auto parsed_share_y_minus_x,
                     parse_shared_input(share_y_minus_x[idx], modulus, num_ring_bits));
    first_bit_shares[idx] = parsed_share_y_minus_x.first;
    remaining_block_shares[idx] = parsed_share_y_minus_x.second;
  }

  // Compute a new share that will be used to compute the carry of remaining_block
  // This value will be input to non-secret shared comparison
  std::vector<uint64_t> offset(batch_size, ((modulus / 2) - 1));

//  std::vector<uint64_t> comparison_input (batch_size);
//  for (size_t idx = 0; idx < batch_size; idx++) {
//    comparison_input[idx] = offset[idx] - remaining_block_shares[idx];
//  }
  ASSIGN_OR_RETURN(std::vector<uint64_t> comparison_input,
                   VectorSubtract(offset, remaining_block_shares, (modulus / 2)));

//  std::cerr << "comp input" << std::endl;
//  for (const auto & e : comparison_input) {
//    std::cerr << e << " ";
//  }
//  std::cerr << std::endl;

  return std::make_pair(first_bit_shares, comparison_input);
}

StatusOr<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
SecretSharedComparisonPrepareInputsPartyOne(
    const std::vector<uint64_t>& share_x,
    const std::vector<uint64_t>& share_y,
    size_t batch_size,
    uint64_t modulus,
    size_t num_ring_bits) {
  // Check dimensions of inputs
  if (share_x.empty() || share_y.empty() ||
      share_x.size() != batch_size ||
      share_y.size() != batch_size) {
    return InvalidArgumentError("Parameters have invalid dimensions.");
  }

  // P_i locally computes y_i - x_i
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_y_minus_x,
                   VectorSubtract(share_y, share_x, modulus));

  // Parse the share of y_i - x_i
  std::vector<uint64_t> first_bit_shares (batch_size);
  std::vector<uint64_t> remaining_block_shares (batch_size);
  for (size_t idx = 0; idx < batch_size; idx++) {
    ASSIGN_OR_RETURN(auto parsed_share_y_minus_x,
                     parse_shared_input(share_y_minus_x[idx], modulus, num_ring_bits));
    first_bit_shares[idx] = parsed_share_y_minus_x.first;
    remaining_block_shares[idx] = parsed_share_y_minus_x.second;
  }

  // remaining_block_shares is a new share that will be used to compute the carry of remaining_block
  // This value will be input to non-secret shared comparison

  return std::make_pair(first_bit_shares, remaining_block_shares);
}

// In the paper: q=num_splits, x=inp
// Parses x = x_{q-1} || ... || x_0
StatusOr<std::vector<uint64_t>> parse_input(uint64_t inp, size_t num_splits, size_t block_length) {
  // check input params are correct
  if (block_length * num_splits != 64) {
    return InvalidArgumentError("block_length * num_splits must equal 64.");
  }
  std::vector<uint64_t> out (num_splits);
  for (size_t idx = 0; idx < num_splits; idx++) {
    out[idx] = ((1ULL << block_length) - 1) & inp;
    // std::cerr << "block idx: " << idx << " block: " << out[idx] << std::endl;
    inp >>= block_length;
  }
  return out;
}

StatusOr<std::pair<ComparisonStateRoundOne, ComparisonMessageRoundOne>>
ComparisonGenerateRoundOneMessage(
    const std::vector<uint64_t>& inp,
    const ComparisonPreprocessedValues& preprocessed_values,
    size_t num_splits, size_t block_length) {
  size_t batch_size = inp.size();
  if (inp.empty() ||
      preprocessed_values.masks.input_mask_short_comparison.size() != batch_size ||
      preprocessed_values.masks.input_mask_short_equality.size() != batch_size) {
    return InvalidArgumentError("Parameters have invalid dimensions.");
  }
  for (size_t idx = 0; idx < batch_size; idx++) {
    if (preprocessed_values.masks.input_mask_short_comparison[idx].size() != num_splits ||
        preprocessed_values.masks.input_mask_short_equality[idx].size() != num_splits) {
      return InvalidArgumentError("Parameters have invalid dimensions.");
    }
  }
  // Parse the inp into num_splits blocks of size block_length
  std::vector<std::vector<uint64_t>> parsed_inp;
  for (size_t idx = 0; idx < batch_size; idx++) {
    ASSIGN_OR_RETURN(std::vector<uint64_t> parsed,
                     parse_input(inp[idx], num_splits, block_length));
    parsed_inp.push_back(parsed);
  }
  // Add the masks on top of the parsed_inp

  // masked inputs for the short comparison and short equality
  std::vector<std::vector<uint64_t>> masked_input_short_comparison (batch_size,
      std::vector<uint64_t> (num_splits, 0));
  std::vector<std::vector<uint64_t>> masked_input_short_equality (batch_size,
      std::vector<uint64_t> (num_splits, 0));
  for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
    for (size_t idx_split = 0; idx_split < num_splits; idx_split++) {
      masked_input_short_comparison[idx_batch][idx_split] =
          ModAdd(parsed_inp[idx_batch][idx_split],
                 preprocessed_values.masks.input_mask_short_comparison[idx_batch][idx_split], (1ULL << (block_length + 1)));
      masked_input_short_equality[idx_batch][idx_split] =
          ModAdd(parsed_inp[idx_batch][idx_split],
                 preprocessed_values.masks.input_mask_short_equality[idx_batch][idx_split], (1ULL << block_length));
    }
  }

  // Construct the round message sent to the other party
  ComparisonMessageRoundOne round_one_msg;
  for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
    for (size_t idx_split = 0; idx_split < num_splits; idx_split++) {
      round_one_msg.add_masked_input_short_comparison(masked_input_short_comparison[idx_batch][idx_split]);
      round_one_msg.add_masked_input_short_equality(masked_input_short_equality[idx_batch][idx_split]);
    }
  }
  // Add these masked inputs into ComparisonStateRoundOne
  ComparisonStateRoundOne round_one_state = {
      .masked_input_short_comparison = std::move(masked_input_short_comparison),
      .masked_input_short_equality = std::move(masked_input_short_equality)
  };
  return std::make_pair(std::move(round_one_state), std::move(round_one_msg));
}

StatusOr<ComparisonShortComparisonEquality>
ComparisonComputeShortComparisonEqualityPartyZero(
    const ComparisonEqualityGates& short_gates,
    const ComparisonStateRoundOne& masked_inp,
    ComparisonMessageRoundOne other_party_masked_inp,
    const ComparisonPreprocessedValues& preprocessed_values,
    size_t num_splits, size_t block_length) {
  size_t batch_size = masked_inp.masked_input_short_comparison.size();

  if (masked_inp.masked_input_short_equality.size() != batch_size ||
      preprocessed_values.key_shares.comparison_key_shares.size() != batch_size ||
      preprocessed_values.key_shares.equality_key_shares.size() != batch_size) {
    return InvalidArgumentError("Parameters have invalid dimensions.");
  }

  if (static_cast<size_t>(
      other_party_masked_inp.masked_input_short_comparison_size()) != (batch_size * num_splits) ||
      static_cast<size_t>(
          other_party_masked_inp.masked_input_short_equality_size()) != (batch_size * num_splits)) {
    return InvalidArgumentError("Message has invalid dimensions.");
  }

  for (size_t idx = 0; idx < batch_size; idx++) {
    if (masked_inp.masked_input_short_equality[idx].size() != num_splits ||
        masked_inp.masked_input_short_comparison[idx].size() != num_splits ||
        preprocessed_values.key_shares.comparison_key_shares[idx].size() != num_splits ||
        preprocessed_values.key_shares.equality_key_shares[idx].size() != num_splits) {
      return InvalidArgumentError("Parameters have invalid dimensions.");
    }
  }

  size_t num_splits_next = num_splits / 2;
  size_t batch_size_next = num_splits_next * batch_size; // number of even equalities, odd equalities
                                                         // even comparisons, odd comparisons
  size_t length_to_vectorize_and = 2 * batch_size_next;

  std::vector<uint64_t> even_comparison_even_equality_output_shares (length_to_vectorize_and);
  std::vector<uint64_t> odd_equality_twice_copied_shares (length_to_vectorize_and);
  std::vector<uint64_t> odd_comparison_appended_zeros_shares (length_to_vectorize_and);

  // Invoke the short cmp/eq gates
  size_t offset = 0;
  for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
    for (size_t idx_split = 0; idx_split < num_splits; idx_split++) {

      //
      // Comparison first
      //

//      // For now unmask inputs
//      uint64_t p0_inp_cmp = ModSub(
//          masked_inp.masked_input_short_comparison[idx_batch][idx_split],
//          preprocessed_values.masks.input_mask_short_comparison[idx_batch][idx_split],
//          (1ULL << block_length));
//      uint64_t p1_inp_cmp = ModSub(
//          other_party_masked_inp.masked_input_short_comparison(offset),
//          other_party_preprocessed_values.masks.input_mask_short_comparison[idx_batch][idx_split],
//          (1ULL << block_length));
//      uint64_t res_cmp_rec = (p0_inp_cmp < p1_inp_cmp);
//
//      std::cerr << "p0 func: x cmp: " << p0_inp_cmp << std::endl;
//      std::cerr << "p1 func: y cmp: " << p1_inp_cmp << std::endl;
//      std::cerr << "expected x < y: " << res_cmp_rec << std::endl;

      absl::uint128 res_cmp;

      // Evaluating Comparison gate key_partyidx on masked input x + r_in0, y + r_in1
      DPF_ASSIGN_OR_RETURN(res_cmp,
          short_gates.CmpGate->Eval(
              absl::uint128(0), preprocessed_values.key_shares.comparison_key_shares[idx_batch][idx_split],
              masked_inp.masked_input_short_comparison[idx_batch][idx_split],
              other_party_masked_inp.masked_input_short_comparison(offset)));

      // Retrieve output mask for cmp
      absl::uint128 r_out_cmp =
          absl::MakeUint128(
              preprocessed_values.key_shares.comparison_key_shares[idx_batch][idx_split].output_mask_share().value_uint128().high(),
              preprocessed_values.key_shares.comparison_key_shares[idx_batch][idx_split].output_mask_share().value_uint128().low());

      // Subtracting out the output mask r_out_cmp
      uint64_t res_cmp_uint64t = uint64_t(res_cmp ^ r_out_cmp);

      // std::cerr << "obtained x < y : " << res_cmp_uint64t << std::endl;

      // Save the results
      size_t output_offset = (idx_split / 2) * batch_size + idx_batch;
      if (idx_split % 2 == 0) {
        // Even
        even_comparison_even_equality_output_shares[output_offset] = res_cmp_uint64t;
      } else {
        // Odd
        odd_comparison_appended_zeros_shares[output_offset] = res_cmp_uint64t;
        odd_comparison_appended_zeros_shares[batch_size_next + output_offset] = 0;
      }

      //
      // Equality next
      //

      absl::uint128 res_eq;

      // Evaluating Equality gate key_partyidx on masked input x + r_in0, y + r_in1
      DPF_ASSIGN_OR_RETURN(res_eq,
          short_gates.EqGate->Eval(
                                   preprocessed_values.key_shares.equality_key_shares[idx_batch][idx_split],
                                   masked_inp.masked_input_short_equality[idx_batch][idx_split],
                                   other_party_masked_inp.masked_input_short_equality(offset)));

      // Retrieve output mask for eq
      absl::uint128 r_out_eq =
          absl::MakeUint128(
              preprocessed_values.key_shares.equality_key_shares[idx_batch][idx_split].output_mask_share().value_uint128().high(),
              preprocessed_values.key_shares.equality_key_shares[idx_batch][idx_split].output_mask_share().value_uint128().low());

      // Subtracting out the output mask r_out_eq
      uint64_t res_eq_uint64t = uint64_t(res_eq ^ r_out_eq);

      // Save the results

      if (idx_split % 2 == 0) {
        // Even
        even_comparison_even_equality_output_shares[batch_size_next + output_offset] = res_eq_uint64t;
      } else {
        // Odd
        odd_equality_twice_copied_shares[output_offset] = res_eq_uint64t;
        odd_equality_twice_copied_shares[batch_size_next + output_offset] = res_eq_uint64t;
      }

      offset++;
    }
  }
  // Add these resulting cmp/eq shares into ComparisonShortComparisonEquality
  ComparisonShortComparisonEquality output_shares = {
      .even_comparison_even_equality_output_shares = std::move(even_comparison_even_equality_output_shares),
      .odd_equality_twice_copied_shares = std::move(odd_equality_twice_copied_shares),
      .odd_comparison_appended_zeros_shares = std::move(odd_comparison_appended_zeros_shares)
  };
  return output_shares;
}

StatusOr<ComparisonShortComparisonEquality>
ComparisonComputeShortComparisonEqualityPartyOne(
    const ComparisonEqualityGates& short_gates,
    const ComparisonStateRoundOne& masked_inp,
    ComparisonMessageRoundOne other_party_masked_inp,
    const ComparisonPreprocessedValues& preprocessed_values,
    size_t num_splits, size_t block_length) {
  size_t batch_size = masked_inp.masked_input_short_comparison.size();

  if (masked_inp.masked_input_short_equality.size() != batch_size ||
      preprocessed_values.key_shares.comparison_key_shares.size() != batch_size ||
      preprocessed_values.key_shares.equality_key_shares.size() != batch_size) {
    return InvalidArgumentError("Parameters have invalid dimensions.");
  }

  if (static_cast<size_t>(
      other_party_masked_inp.masked_input_short_comparison_size()) != (batch_size * num_splits) ||
      static_cast<size_t>(
          other_party_masked_inp.masked_input_short_equality_size()) != (batch_size * num_splits)) {
    return InvalidArgumentError("Message has invalid dimensions.");
  }

  for (size_t idx = 0; idx < batch_size; idx++) {
    if (masked_inp.masked_input_short_equality[idx].size() != num_splits ||
        masked_inp.masked_input_short_comparison[idx].size() != num_splits ||
        preprocessed_values.key_shares.comparison_key_shares[idx].size() != num_splits ||
        preprocessed_values.key_shares.equality_key_shares[idx].size() != num_splits) {
      return InvalidArgumentError("Parameters have invalid dimensions.");
    }
  }

  size_t num_splits_next = num_splits / 2;
  size_t batch_size_next = num_splits_next * batch_size; // number of even equalities, odd equalities
  // even comparisons, odd comparisons
  size_t length_to_vectorize_and = 2 * batch_size_next;

  std::vector<uint64_t> even_comparison_even_equality_output_shares (length_to_vectorize_and);
  std::vector<uint64_t> odd_equality_twice_copied_shares (length_to_vectorize_and);
  std::vector<uint64_t> odd_comparison_appended_zeros_shares (length_to_vectorize_and);

  // Invoke the short cmp/eq gates
  size_t offset = 0;
  for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
    for (size_t idx_split = 0; idx_split < num_splits; idx_split++) {

      //
      // Comparison first
      //

//      // For now unmask inputs
//      uint64_t p1_inp_cmp = ModSub(
//          masked_inp.masked_input_short_comparison[idx_batch][idx_split],
//          preprocessed_values.masks.input_mask_short_comparison[idx_batch][idx_split],
//          (1ULL << block_length));
//      uint64_t p0_inp_cmp = ModSub(
//          other_party_masked_inp.masked_input_short_comparison(offset),
//          other_party_preprocessed_values.masks.input_mask_short_comparison[idx_batch][idx_split],
//          (1ULL << block_length));
//      uint64_t res_cmp_rec = (p0_inp_cmp < p1_inp_cmp);
//
//      std::cerr << "p1 func: x cmp: " << p0_inp_cmp << std::endl;
//      std::cerr << "p1 func: y cmp: " << p1_inp_cmp << std::endl;
//      std::cerr << "x < y: " << res_cmp_rec << std::endl;

      absl::uint128 res_cmp;

      // Evaluating Comparison gate key_partyidx on masked input x + r_in0, y + r_in1
      DPF_ASSIGN_OR_RETURN(res_cmp,
                           short_gates.CmpGate->Eval(
                               absl::uint128(1), preprocessed_values.key_shares.comparison_key_shares[idx_batch][idx_split],
                               other_party_masked_inp.masked_input_short_comparison(offset),
                               masked_inp.masked_input_short_comparison[idx_batch][idx_split]));

      // Retrieve output mask for cmp
      absl::uint128 r_out_cmp =
          absl::MakeUint128(
              preprocessed_values.key_shares.comparison_key_shares[idx_batch][idx_split].output_mask_share().value_uint128().high(),
              preprocessed_values.key_shares.comparison_key_shares[idx_batch][idx_split].output_mask_share().value_uint128().low());

      // Subtracting out the output mask r_out_cmp
      uint64_t res_cmp_uint64t = uint64_t(res_cmp ^ r_out_cmp);

      // std::cerr << "obtained x < y : " << res_cmp_uint64t << std::endl;

      // Save the results
      size_t output_offset = (idx_split / 2) * batch_size + idx_batch;
      if (idx_split % 2 == 0) {
        // Even
        even_comparison_even_equality_output_shares[output_offset] = res_cmp_uint64t;
      } else {
        // Odd
        odd_comparison_appended_zeros_shares[output_offset] = res_cmp_uint64t;
        odd_comparison_appended_zeros_shares[batch_size_next + output_offset] = 0;
      }

      //
      // Equality next
      //

      absl::uint128 res_eq;

      // Evaluating Equality gate key_partyidx on masked input x + r_in0, y + r_in1
      DPF_ASSIGN_OR_RETURN(res_eq,
                           short_gates.EqGate->Eval(
                               preprocessed_values.key_shares.equality_key_shares[idx_batch][idx_split],
                               other_party_masked_inp.masked_input_short_equality(offset),
                               masked_inp.masked_input_short_equality[idx_batch][idx_split]));

      // Retrieve output mask for eq
      absl::uint128 r_out_eq =
          absl::MakeUint128(
              preprocessed_values.key_shares.equality_key_shares[idx_batch][idx_split].output_mask_share().value_uint128().high(),
              preprocessed_values.key_shares.equality_key_shares[idx_batch][idx_split].output_mask_share().value_uint128().low());

      // Subtracting out the output mask r_out_eq
      uint64_t res_eq_uint64t = uint64_t(res_eq ^ r_out_eq);

      // Save the results

      if (idx_split % 2 == 0) {
        // Even
        even_comparison_even_equality_output_shares[batch_size_next + output_offset] = res_eq_uint64t;
      } else {
        // Odd
        odd_equality_twice_copied_shares[output_offset] = res_eq_uint64t;
        odd_equality_twice_copied_shares[batch_size_next + output_offset] = res_eq_uint64t;
      }

      offset++;
    }
  }
  // Add these resulting cmp/eq shares into ComparisonShortComparisonEquality
  ComparisonShortComparisonEquality output_shares = {
      .even_comparison_even_equality_output_shares = std::move(even_comparison_even_equality_output_shares),
      .odd_equality_twice_copied_shares = std::move(odd_equality_twice_copied_shares),
      .odd_comparison_appended_zeros_shares = std::move(odd_comparison_appended_zeros_shares)
  };
  return output_shares;
}

StatusOr<std::pair<BatchedMultState, MultiplicationGateMessage>>
ComparisonGenerateNextRoundMessage(
    const ComparisonShortComparisonEquality& shares,
    const ComparisonPreprocessedValues& preprocessed_values) {
  return GenerateHadamardProductMessage(
      shares.even_comparison_even_equality_output_shares,
      shares.odd_equality_twice_copied_shares,
      preprocessed_values.beaver_vector_shares.back(),
      2);
}

StatusOr<ComparisonShortComparisonEquality>
ComparisonProcessNextRoundMessagePartyZero(
    const ComparisonShortComparisonEquality& shares, // for XORs
    BatchedMultState state,
    ComparisonPreprocessedValues& preprocessed_values,
    MultiplicationGateMessage other_party_message,
    size_t num_splits_after_this_round) {
  size_t batch_size = preprocessed_values.masks.input_mask_short_comparison.size();
  // Check dimensions (dependent on round number)
  if (num_splits_after_this_round == 1) {
    // last round
    if (batch_size != (shares.even_comparison_even_equality_output_shares.size() / num_splits_after_this_round)) {
      return InvalidArgumentError("Inputs have invalid dimensions.");
    }
    if (preprocessed_values.beaver_vector_shares.back().GetA().size() !=
        (num_splits_after_this_round * batch_size)) {
      return InvalidArgumentError("Beaver triples have invalid dimensions.");
    }
  } else {
    // not last round
    if (batch_size != ((shares.even_comparison_even_equality_output_shares.size() / num_splits_after_this_round) / 2)) {
      return InvalidArgumentError("Inputs have invalid dimensions.");
    }
    if (preprocessed_values.beaver_vector_shares.back().GetA().size() !=
        (num_splits_after_this_round * batch_size * 2)) {
      return InvalidArgumentError("Beaver triples have invalid dimensions.");
    }
  }

  // First finish the AND gate (multiply mod 2)
  // Compute temp in the cryptflow paper, and
  // AND even 'equalities' with odd 'equalities' to get new 'equalities' for the next round
  ASSIGN_OR_RETURN(std::vector<uint64_t> hadamard_share,
                   HadamardProductPartyZero(
                       state,
                       preprocessed_values.beaver_vector_shares.back(),
                       other_party_message,
                       0, // num_fractional_bits
                       2)); // modulus

  preprocessed_values.beaver_vector_shares.pop_back();

  // XOR now with odd 'less than's to get 'less thans' and 'equalities' for the next round
  ASSIGN_OR_RETURN(std::vector<uint64_t> next_round_lessthan_equality_share,
                   VectorAdd(
                       hadamard_share,
                       shares.odd_comparison_appended_zeros_shares,
                       2)); // modulus

  // Place into a suitable format into new ComparisonShortComparisonEquality
  // so that we can use in the next round
  // if batch_size_next is 0, this is the last round (needs careful handling)
  size_t batch_size_next = (num_splits_after_this_round / 2) * batch_size;

  // even comparisons, odd comparisons
  size_t length_to_vectorize_and = 2 * batch_size_next;

  std::vector<uint64_t> even_comparison_even_equality_output_shares (length_to_vectorize_and);
  std::vector<uint64_t> odd_equality_twice_copied_shares (length_to_vectorize_and);
  std::vector<uint64_t> odd_comparison_appended_zeros_shares (length_to_vectorize_and);

  if (num_splits_after_this_round == 1) {
    // Last round is slightly different as there is no odd block, only even (0)
    // The output vector was initialized to double size above
    even_comparison_even_equality_output_shares.resize(batch_size);
    for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
      even_comparison_even_equality_output_shares[idx_batch] = next_round_lessthan_equality_share[idx_batch];
    }
  } else if (num_splits_after_this_round == 2) {
    // Second to Last round is slightly different as we do not want to multiply equalities in the last round
    // std::cerr << "second to last round" << std::endl;
    // no need for the even equality part
    even_comparison_even_equality_output_shares.resize(batch_size);
    // no need to copy odd equality twice
    odd_equality_twice_copied_shares.resize(batch_size);
    // no need to append zeros
    odd_comparison_appended_zeros_shares.resize(batch_size);

    size_t offset = 0;
    for (size_t idx_split = 0; idx_split < num_splits_after_this_round; idx_split++) {
      size_t output_offset = (idx_split / 2) * batch_size;
      if (idx_split % 2 == 0) {
        // Even
        for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
          even_comparison_even_equality_output_shares[output_offset + idx_batch] =
              next_round_lessthan_equality_share[offset + idx_batch];
        }
      } else {
        // Odd
        for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
          odd_comparison_appended_zeros_shares[output_offset + idx_batch] =
              next_round_lessthan_equality_share[offset + idx_batch];
          odd_equality_twice_copied_shares[output_offset + idx_batch] =
              next_round_lessthan_equality_share[length_to_vectorize_and + offset + idx_batch];
        }
      }
      offset += batch_size;
    }
  } else {
    size_t offset = 0;
    for (size_t idx_split = 0; idx_split < num_splits_after_this_round; idx_split++) {
      size_t output_offset = (idx_split / 2) * batch_size;
      if (idx_split % 2 == 0) {
        // Even
        for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
          even_comparison_even_equality_output_shares[output_offset + idx_batch] =
              next_round_lessthan_equality_share[offset + idx_batch];
          even_comparison_even_equality_output_shares[batch_size_next + output_offset + idx_batch] =
              next_round_lessthan_equality_share[length_to_vectorize_and + offset + idx_batch];
        }
      } else {
        // Odd
        for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
          odd_comparison_appended_zeros_shares[output_offset + idx_batch] =
              next_round_lessthan_equality_share[offset + idx_batch];
          odd_comparison_appended_zeros_shares[batch_size_next + output_offset + idx_batch] = 0;
          odd_equality_twice_copied_shares[output_offset + idx_batch] =
              next_round_lessthan_equality_share[length_to_vectorize_and + offset + idx_batch];
          odd_equality_twice_copied_shares[batch_size_next + output_offset + idx_batch] =
              next_round_lessthan_equality_share[length_to_vectorize_and + offset + idx_batch];
        }
      }
      offset += batch_size;
    }
  }

  ComparisonShortComparisonEquality output_shares = {
      .even_comparison_even_equality_output_shares = std::move(even_comparison_even_equality_output_shares),
      .odd_equality_twice_copied_shares = std::move(odd_equality_twice_copied_shares),
      .odd_comparison_appended_zeros_shares = std::move(odd_comparison_appended_zeros_shares)
  };

  return output_shares;
}

StatusOr<ComparisonShortComparisonEquality>
ComparisonProcessNextRoundMessagePartyOne(
    const ComparisonShortComparisonEquality& shares, // for XORs
    BatchedMultState state,
    ComparisonPreprocessedValues& preprocessed_values,
    MultiplicationGateMessage other_party_message,
    size_t num_splits_after_this_round) {
  size_t batch_size = preprocessed_values.masks.input_mask_short_comparison.size();
  // Check dimensions (dependent on round number)
  if (num_splits_after_this_round == 1) {
    // last round
    if (batch_size != (shares.even_comparison_even_equality_output_shares.size() / num_splits_after_this_round)) {
      return InvalidArgumentError("Inputs have invalid dimensions.");
    }
    if (preprocessed_values.beaver_vector_shares.back().GetA().size() !=
        (num_splits_after_this_round * batch_size)) {
      return InvalidArgumentError("Beaver triples have invalid dimensions.");
    }
  } else {
    // not last round
    if (batch_size != ((shares.even_comparison_even_equality_output_shares.size() / num_splits_after_this_round) / 2)) {
      return InvalidArgumentError("Inputs have invalid dimensions.");
    }
    if (preprocessed_values.beaver_vector_shares.back().GetA().size() !=
        (num_splits_after_this_round * batch_size * 2)) {
      return InvalidArgumentError("Beaver triples have invalid dimensions.");
    }
  }

  // First finish the AND gate (multiply mod 2)
  // Compute temp in the cryptflow paper, and
  // AND even 'equalities' with odd 'equalities' to get new 'equalities' for the next round
  ASSIGN_OR_RETURN(std::vector<uint64_t> hadamard_share,
                   HadamardProductPartyOne(
                       state,
                       preprocessed_values.beaver_vector_shares.back(),
                       other_party_message,
                       0, // num_fractional_bits
                       2)); // modulus

  preprocessed_values.beaver_vector_shares.pop_back();

  // XOR now with odd 'less than's to get 'less thans' and 'equalities' for the next round
  ASSIGN_OR_RETURN(std::vector<uint64_t> next_round_lessthan_equality_share,
                   VectorAdd(
                       hadamard_share,
                       shares.odd_comparison_appended_zeros_shares,
                       2)); // modulus

  // Place into a suitable format into new ComparisonShortComparisonEquality
  // so that we can use in the next round
  // if batch_size_next is 0, this is the last round (needs careful handling)
  size_t batch_size_next = (num_splits_after_this_round / 2) * batch_size;

  // even comparisons, odd comparisons
  size_t length_to_vectorize_and = 2 * batch_size_next;

  std::vector<uint64_t> even_comparison_even_equality_output_shares (length_to_vectorize_and);
  std::vector<uint64_t> odd_equality_twice_copied_shares (length_to_vectorize_and);
  std::vector<uint64_t> odd_comparison_appended_zeros_shares (length_to_vectorize_and);

  if (num_splits_after_this_round == 1) {
    // Last round is slightly different as there is no odd block, only even (0)
    // std::cerr << "last round" << std::endl;
    // The output vector was initialized to double size above
    even_comparison_even_equality_output_shares.resize(batch_size);
    for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
      even_comparison_even_equality_output_shares[idx_batch] = next_round_lessthan_equality_share[idx_batch];
    }
  } else if (num_splits_after_this_round == 2) {
    // Second to Last round is slightly different as we do not want to multiply equalities in the last round
    // std::cerr << "second to last round" << std::endl;
    // no need for the even equality part
    even_comparison_even_equality_output_shares.resize(batch_size);
    // no need to copy odd equality twice
    odd_equality_twice_copied_shares.resize(batch_size);
    // no need to append zeros
    odd_comparison_appended_zeros_shares.resize(batch_size);

    size_t offset = 0;
    for (size_t idx_split = 0; idx_split < num_splits_after_this_round; idx_split++) {
      size_t output_offset = (idx_split / 2) * batch_size;
      if (idx_split % 2 == 0) {
        // Even
        for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
          even_comparison_even_equality_output_shares[output_offset + idx_batch] =
              next_round_lessthan_equality_share[offset + idx_batch];
        }
      } else {
        // Odd
        for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
          odd_comparison_appended_zeros_shares[output_offset + idx_batch] =
              next_round_lessthan_equality_share[offset + idx_batch];
          odd_equality_twice_copied_shares[output_offset + idx_batch] =
              next_round_lessthan_equality_share[length_to_vectorize_and + offset + idx_batch];
        }
      }
      offset += batch_size;
    }
  } else {
    size_t offset = 0;
    for (size_t idx_split = 0; idx_split < num_splits_after_this_round; idx_split++) {
      size_t output_offset = (idx_split / 2) * batch_size;
      if (idx_split % 2 == 0) {
        // Even
        for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
          even_comparison_even_equality_output_shares[output_offset + idx_batch] =
              next_round_lessthan_equality_share[offset + idx_batch];
          even_comparison_even_equality_output_shares[batch_size_next + output_offset + idx_batch] =
              next_round_lessthan_equality_share[length_to_vectorize_and + offset + idx_batch];
        }
      } else {
        // Odd
        for (size_t idx_batch = 0; idx_batch < batch_size; idx_batch++) {
          odd_comparison_appended_zeros_shares[output_offset + idx_batch] =
              next_round_lessthan_equality_share[offset + idx_batch];
          odd_comparison_appended_zeros_shares[batch_size_next + output_offset + idx_batch] = 0;
          odd_equality_twice_copied_shares[output_offset + idx_batch] =
              next_round_lessthan_equality_share[length_to_vectorize_and + offset + idx_batch];
          odd_equality_twice_copied_shares[batch_size_next + output_offset + idx_batch] =
              next_round_lessthan_equality_share[length_to_vectorize_and + offset + idx_batch];
        }
      }
      offset += batch_size;
    }
  }

  ComparisonShortComparisonEquality output_shares = {
      .even_comparison_even_equality_output_shares = std::move(even_comparison_even_equality_output_shares),
      .odd_equality_twice_copied_shares = std::move(odd_equality_twice_copied_shares),
      .odd_comparison_appended_zeros_shares = std::move(odd_comparison_appended_zeros_shares)
  };
  return output_shares;
}

StatusOr<std::vector<uint64_t>>
SecretSharedComparisonFinishReduction(
    const std::vector<uint64_t>& share_first_bit,
    const std::vector<uint64_t>& share_comparison_output,
    size_t batch_size) {
  // Check dimensions of inputs
  if (share_first_bit.empty() || share_comparison_output.empty() ||
      share_first_bit.size() != batch_size ||
      share_comparison_output.size() != batch_size) {
    return InvalidArgumentError("Parameters have invalid dimensions.");
  }
  ASSIGN_OR_RETURN(std::vector<uint64_t> comparison_output_share,
      VectorAdd(share_first_bit, share_comparison_output, 2));
  return comparison_output_share;
}

}  // namespace secure_comparison
}  // namespace private_join_and_compute
