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

#include "secret_sharing_mpc/gates/powers.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "secret_sharing_mpc/gates/hadamard_product.h"

namespace private_join_and_compute {

namespace internal {

// This preprocessing function is only for testing purposes.
// Hence, it is in internal namespace.
StatusOr<std::pair<std::vector<std::vector<uint64_t>>,
                   std::vector<std::vector<uint64_t>>>>
SamplePowersOfRandomVector(
        size_t k, size_t n, std::unique_ptr<FixedPointElementFactory>& fp_factory_,
        uint64_t modulus) {
  if (k < 1 || n < 1) {
    return InvalidArgumentError(
            "SamplePowersOfRandomVector: k and n must be positive integers.");
  }
  // Sample a PRNG to generate the shares.
  auto seed = BasicRng::GenerateSeed();
  if (!seed.ok()) {
    return InternalError("Random seed fails to be initialized.");
  }
  auto uptr_prng = BasicRng::Create(seed.value());
  if (!uptr_prng.ok()) {
    return InternalError("Prng fails to be initialized.");
  }
  BasicRng* prng = uptr_prng.value().get();

  std::vector<std::vector<uint64_t>> random_powers(k,
                                                   std::vector<uint64_t>(n, 0));
  ASSIGN_OR_RETURN(FixedPointElement fpe_zero,
                   fp_factory_->CreateFixedPointElementFromInt(0));
  std::vector<std::vector<FixedPointElement>> random_powers_fpe(
          k, std::vector<FixedPointElement>(n, fpe_zero));

  std::vector<std::vector<uint64_t>> random_powers_share_0(
          k, std::vector<uint64_t>(n, 0));
  std::vector<std::vector<uint64_t>> random_powers_share_1(
          k, std::vector<uint64_t>(n, 0));

  // First, generate the random k shares of size n for party 0
  // corresponding to b_0, b^2_0, ..., b^k_0
  for (size_t idx = 0; idx < k; idx++) {
    ASSIGN_OR_RETURN(random_powers_share_0[idx],
                     SampleVectorFromPrng(n, modulus, prng));
  }

  // First generate b in a suitable range [0,2^num_fractional_bits)
  // Keep both fixed-point element and ring form to do truncated multiplication
  for (size_t idx = 0; idx < n; idx++) {
    ASSIGN_OR_RETURN(uint64_t random_b, prng->Rand64());
    random_powers[0][idx] =
            random_b % fp_factory_->GetParams().fractional_multiplier;
    ASSIGN_OR_RETURN(
            random_powers_fpe[0][idx],
            fp_factory_->ImportFixedPointElementFromUint64(random_powers[0][idx]));
  }

  // Now generate b^2, ..., b^k i.e. b, b*b, b*b*b etc. such that *
  // represents truncated element-wise multiplication mod modulus.
  for (size_t k_idx = 1; k_idx < k; k_idx++) {
    for (size_t n_idx = 0; n_idx < n; n_idx++) {
      ASSIGN_OR_RETURN(random_powers_fpe[k_idx][n_idx],
                       random_powers_fpe[0][n_idx].TruncMulFP(
                               random_powers_fpe[k_idx - 1][n_idx]));
      random_powers[k_idx][n_idx] =
              random_powers_fpe[k_idx][n_idx].ExportToUint64();
    }
  }

  // Compute shares for party 1:
  // [b]_1, [b^2]_1, ..., [b^k]_1 <-- (b - b_0, b^2 - b^2_0, b^3 - b^3_0).
  for (size_t idx = 0; idx < k; idx++) {
    ASSIGN_OR_RETURN(
            random_powers_share_1[idx],
            BatchedModSub(random_powers[idx], random_powers_share_0[idx], modulus));
  }

  return std::make_pair(random_powers_share_0, random_powers_share_1);
}

// HELPER functions below, not client functions
// They are secure but should not be invoked by a client

// Helper function to compute powers (not invoked by the client but by PowersGenerateOTInputsPartyZero)
StatusOr<std::vector<std::vector<uint64_t>>>
HoneyBadgerPowersPartyZero(size_t k, size_t n,
                           const std::vector<std::vector<uint64_t>>& random_powers_share,
                           std::vector<FixedPointElement>& m_minus_b_option,
                           std::unique_ptr<FixedPointElementFactory>& fp_factory_,
                           uint64_t modulus) {
  std::vector<std::vector<uint64_t>> powers_share_option(
          k, std::vector<uint64_t>(n, 0));
  // mb_table:
  // One instance contains powers of m^i * b^j for i,j in [0, k + 1)
  // There are n instances
  // Thus, mb_table[i][j][t] holds (a share of): m[t]^i * b[t]^j
  // The order of this nested vector was selected intentionally this way as it
  // enables vectorization and using move semantics
  std::vector<std::vector<std::vector<uint64_t>>> mb_table(
          k + 1, std::vector<std::vector<uint64_t>>(k + 1,
                                                    std::vector<uint64_t>(n, 0)));

  // Initialize first rows of mb_table efficiently with move semantics
  // This is the step that differs across the two parties.
  // mb_table[0][0] needs to be a sharing of 1,
  // and hence is set by 1 party only
  // (It is set to 1 in ring representation i.e. 2^num_fractional_bits)
  // This is the part that differs across the two parties
  // This first cell of the table is equal to 1 as it corresponds to m^0 b^0.
  mb_table[0][0] =
          std::vector<uint64_t>(
                  n, fp_factory_->GetParams().fractional_multiplier);
  for (size_t col = 1; col < k + 1; col++) {
    mb_table[0][col] = std::move(random_powers_share[col - 1]);
  }

  // Compute for each powers instance [0,n]:
  // mb_table[i][j][0,n] s.t. d = i + j
  // This can be seen as sequentially computing the diagonals from top left
  // to bottom right for each of n powers invocations, with each diagonal
  // computation relying on the computation of the previous diagonal.
  // mb_table[i][j] needs to be assigned a share of m^i * b^j:
  // Use this rule m^i * b^j = b^d + (m-b) * sum_0^{i-1}[m^{i-1-d} * b^{j+d}]
  for (size_t d = 1; d <= k; d++) {
    // sum represents the addition of diagonal entries in mb_table
    std::vector<uint64_t> sum(n, 0);
    for (size_t i = 1; i <= d; i++) {
      size_t j = d - i;  // i and j must add to d
      ASSIGN_OR_RETURN(sum,
                       BatchedModAdd(sum, mb_table[i - 1][j], modulus));
      // Compute the product of m-b with the sum
      std::vector<uint64_t> sum_mb_prod(n, 0);
      for (size_t idx = 0; idx < n; idx++) {
        ASSIGN_OR_RETURN(
                FixedPointElement prod,
                fp_factory_->ImportFixedPointElementFromUint64(sum[idx])
                        ->TruncMulFP(m_minus_b_option[idx]));
        sum_mb_prod[idx] = prod.ExportToUint64();
      }
      ASSIGN_OR_RETURN(mb_table[i][j],
                       BatchedModAdd(mb_table[0][d], sum_mb_prod, modulus));
    }
  }

  // Retrieve the powers from the first column of mb_table
  for (size_t i = 0; i < k; i++) {
    powers_share_option[i] = std::move(mb_table[i + 1][0]);
  }


  return powers_share_option;
}

// Helper function to compute powers (not invoked by the client but by PowersGenerateOTInputsPartyOne)
StatusOr<std::vector<std::vector<uint64_t>>>
HoneyBadgerPowersPartyOne(size_t k, size_t n,
                          const std::vector<std::vector<uint64_t>>& random_powers_share,
                          std::vector<FixedPointElement>& m_minus_b_option,
                          std::unique_ptr<FixedPointElementFactory>& fp_factory_,
                          uint64_t modulus) {
  std::vector<std::vector<uint64_t>> powers_share_option(
          k, std::vector<uint64_t>(n, 0));
  // mb_table:
  // One instance contains powers of m^i * b^j for i,j in [0, k + 1)
  // There are n instances
  // Thus, mb_table[i][j][t] holds (a share of): m[t]^i * b[t]^j
  // The order of this nested vector was selected intentionally this way as it
  // enables vectorization and using move semantics
  // mb_table:
  // k + 1 rows
  // k + 1 columns
  // n entries in each [row, column]
  std::vector<std::vector<std::vector<uint64_t>>> mb_table(
          k + 1,
          std::vector<std::vector<uint64_t>>(k + 1, std::vector<uint64_t>(n, 0)));

  // Initialize first rows of mb_table efficiently with move semantics
  // The first row is set to the powers from preprocessing (b, b^2, ..., b^k)
  for (size_t col = 1; col < k + 1; col++) {
    mb_table[0][col] = std::move(random_powers_share[col - 1]);
  }

  // Compute for each powers instance [0,n]: mb_table[i][j][0,n] s.t. d = i +
  // j This can be seen as sequentially computing the diagonals from top left
  // to bottom right for each of n powers invocations, with each diagonal
  // computation relying on the computation of the previous diagonal.
  // mb_table[i][j] needs to be assigned a share of m^i * b^j:
  // Use this rule m^i * b^j = b^d + (m-b) * sum_0^{i-1}[m^{i-1-d} * b^{j+d}]
  for (size_t d = 1; d <= k; d++) {
    std::vector<uint64_t> sum(n, 0);
    for (size_t i = 1; i <= d; i++) {
      size_t j = d - i;  // i and j must add to d
      ASSIGN_OR_RETURN(sum, BatchedModAdd(sum, mb_table[i - 1][j], modulus));
      std::vector<uint64_t> sum_mb_prod(n, 0);
      for (size_t idx = 0; idx < n; idx++) {
        ASSIGN_OR_RETURN(
                FixedPointElement prod,
                fp_factory_->ImportFixedPointElementFromUint64(sum[idx])
                        ->TruncMulFP(m_minus_b_option[idx]));
        sum_mb_prod[idx] = prod.ExportToUint64();
      }
      ASSIGN_OR_RETURN(mb_table[i][j],
                       BatchedModAdd(mb_table[0][d], sum_mb_prod, modulus));
    }
  }
  // Retrieve the powers from the first column of mb_table
  for (size_t i = 0; i < k; i++) {
    powers_share_option[i] = std::move(mb_table[i + 1][0]);
  }
  return powers_share_option;
}
/*
// This preprocessing function is only for testing purposes.
// Hence, it is in internal namespace.
StatusOr<std::pair<PowersRandomOTPrecomputation, PowersRandomOTPrecomputation>>
PowersPreprocessRandomOTs (
    size_t k, size_t n,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_,
    uint64_t modulus) {
  // Generate n random bools
  std::vector<uint64_t> receiver_choice_vector_p0 (n, 0);
  std::vector<uint64_t> receiver_choice_vector_p1 (n, 0);
  // Generate send messages
  std::vector<std::vector<uint64_t>> sender_msg0_vector_p0 (k, std::vector<uint64_t> (n,0));
  std::vector<std::vector<uint64_t>> sender_msg1_vector_p0 (k, std::vector<uint64_t> (n,0));
  std::vector<std::vector<uint64_t>> sender_msg0_vector_p1 (k, std::vector<uint64_t> (n,0));
  std::vector<std::vector<uint64_t>> sender_msg1_vector_p1 (k, std::vector<uint64_t> (n,0));

  auto random_seed = BasicRng::GenerateSeed();
  if (!random_seed.ok()) {
    return InternalError("Random seed generation fails.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<BasicRng> prng
  BasicRng::Create(random_seed.value()));

  ASSIGN_OR_RETURN(receiver_choice_vector_p0,
      SampleVectorFromPrng(n, 2, prng.get()));
  ASSIGN_OR_RETURN(receiver_choice_vector_p1,
      SampleVectorFromPrng(n, 2, prng.get()));
  for (size_t idx = 0; idx < k; idx++) {
    ASSIGN_OR_RETURN(sender_msg0_vector_p0[idx],
        SampleVectorFromPrng(n, modulus, prng.get())
    );
    ASSIGN_OR_RETURN(sender_msg1_vector_p0[idx],
        SampleVectorFromPrng(n, modulus, prng.get())
    );
    ASSIGN_OR_RETURN(sender_msg0_vector_p1[idx],
        SampleVectorFromPrng(n, modulus, prng.get())
    );
    ASSIGN_OR_RETURN(sender_msg1_vector_p1[idx],
        SampleVectorFromPrng(n, modulus, prng.get())
    );
  }

  PowersRandomOTCorrelationSenderMessages sender_msgs_p0 = {
      .sender_msgs0 = std::move(sender_msg0_vector_p0),
      .sender_msgs1 = std::move(sender_msg1_vector_p0)
  };
  PowersRandomOTCorrelationSenderMessages sender_msgs_p1 = {
      .sender_msg0 = std::move(sender_msg0_vector_p1[n_idx]),
      .sender_msg1 = std::move(sender_msg1_vector_p1[n_idx])
  };
  PowersRandomOTCorrelationReceiverMessages receiver_msgs_p0;
  PowersRandomOTCorrelationReceiverMessages receiver_msgs_p1;

  for (size_t n_idx = 0; n_idx < n; n_idx++) {
    receiver_msgs_p0.receiver_choice.push_back(receiver_choice_vector_p0[n_idx]);
    receiver_msgs_p1.receiver_choice.push_back(receiver_choice_vector_p1[n_idx]);
  }
  for (size_t n_idx = 0; n_idx < n; n_idx++) {
    receiver_msgs_p0.receiver_msg.push_back(receiver_choice_vector_p0[n_idx]
          ? sender_msg1_vector_p1[n_idx]
          : sender_msg0_vector_p1[n_idx]);

    receiver_msgs_p1.receiver_msg.push_back(receiver_choice_vector_p1[n_idx]
          ? sender_msg1_vector_p0[n_idx]
          : sender_msg0_vector_p0[n_idx]);
  }
  RandomOTPrecomputation p0 = {
      .sender_msgs = sender_msgs_p0,
      .receiver_msgs = receiver_msgs_p0
  };
  RandomOTPrecomputation p1 = {
      .sender_msgs = sender_msgs_p1,
      .receiver_msgs = receiver_msgs_p1
  };
  return std::make_pair(p0, p1);
}*/

}  // namespace internal

// Helper function to generate the messages for the powers operation.
StatusOr<std::pair<PowersStateMminusB, PowersMessageMminusB>>
GenerateBatchedPowersMessage(
        const std::vector<uint64_t>& share_m,
        const std::vector<std::vector<uint64_t>>& random_powers_share,
        uint64_t num_fractional_bits,
        uint64_t modulus) {
  if (share_m.empty() || random_powers_share.empty()) {
    return InvalidArgumentError("GenerateBatchedPowersMessage: "
                                "input must not be empty.");
  }

  // n is number of powers invocations
  size_t n = share_m.size();
  size_t k = random_powers_share.size();

  for (size_t idx = 0; idx < k; idx++) {
    if (n != random_powers_share[idx].size()) {
      return InvalidArgumentError("GenerateBatchedPowersMessage: "
                                  "shares must have the same length.");
    }
  }

  ASSIGN_OR_RETURN(auto m_minus_b_full,
                   BatchedModSub(share_m, random_powers_share[0], modulus));
  // Save only the num_fractional_bits from m_minus_b_full
  std::vector<uint64_t> m_minus_b_fractional (n, 0);
  for (size_t idx = 0; idx < n; idx++) {
    // m_minus_b & (2^num_fractional_bits - 1)
    m_minus_b_fractional[idx] = m_minus_b_full[idx] &
            ((1ULL << num_fractional_bits) - 1);
  }

  PowersMessageMminusB pow_message;
  for (size_t idx = 0; idx < n; idx++) {
    pow_message.add_vector_m_minus_vector_b_shares(m_minus_b_fractional[idx]);
  }
  PowersStateMminusB state = {
          .share_m_minus_b_full = std::move(m_minus_b_full),
          .share_m_minus_b_fractional = std::move(m_minus_b_fractional)};
  return std::make_pair(std::move(state), std::move(pow_message));
}

StatusOr<std::pair<PowersStateOTInputs, PowersShareOfPowersShare>>
PowersGenerateOTInputsPartyZero(
        PowersStateMminusB state,
        const std::vector<std::vector<uint64_t>>& random_powers_share,
        PowersMessageMminusB other_party_message,
        std::unique_ptr<FixedPointElementFactory>& fp_factory_, uint64_t modulus) {
  if (state.share_m_minus_b_full.empty() ||
          state.share_m_minus_b_fractional.empty()) {
    return InvalidArgumentError("Powers: state must not be empty.");
  }

  // n is the number of instances of powers
  size_t n = state.share_m_minus_b_full.size();

  if (state.share_m_minus_b_fractional.size() != n) {
    return InvalidArgumentError("Powers: state has invalid dimensions.");
  }
  for (size_t idx = 0; idx < random_powers_share.size(); idx++) {
    if (n != random_powers_share[idx].size()) {
      return InvalidArgumentError(
              "Powers: incorrect number of preprocessed powers.");
    }
  }
  if (static_cast<size_t>(
          other_party_message.vector_m_minus_vector_b_shares_size()) != n) {
    return InvalidArgumentError("Powers: m - b message size must equal n.");
  }

  // Reconstruct (m - b), one for each of the n batched invocations
  // Recall this is (m - b)^fractional as it contains
  // only the last num_fractional_bits bits.
  std::vector<uint64_t> m_minus_b_fractional(n, 0);
  for (size_t idx = 0; idx < n; idx++) {
    m_minus_b_fractional[idx] =
            ModAdd(other_party_message.vector_m_minus_vector_b_shares(idx),
                   state.share_m_minus_b_fractional[idx], modulus);
  }
  // Elements of m_minus_b_fractional can have up to num_fractional_bits+1 bits
  // as there can be a carry

  // Save the carry
  std::vector<bool> carry_bits (n, false);
  for (size_t idx = 0; idx < n; idx++) {
    // AND m_minus_b_fractional with a mask of all zeros but a single one
    // in the num_fractional_bits position, then right shift to get the carry
    carry_bits[idx] = (m_minus_b_fractional[idx] &
            fp_factory_->GetParams().fractional_multiplier) >>
                                                            fp_factory_->GetParams().num_fractional_bits;
  }
  // Save p (bit at num_fractional_bits position of
  // share of P_0's m_minus_b_full)
  // p = [m-b]^num_fractional_bits_P0
  std::vector<bool> p_bits (n, false);
  for (size_t idx = 0; idx < n; idx++) {
    // AND m_minus_b_full with a mask of all zeros but a single one
    // in the num_fractional_bits position, then right shift to get the carry
    p_bits[idx] = (state.share_m_minus_b_full[idx] &
            fp_factory_->GetParams().fractional_multiplier) >>
                                                            fp_factory_->GetParams().num_fractional_bits;
  }

  // Form the two options for m_minus_b
  ASSIGN_OR_RETURN(FixedPointElement zero,
                   fp_factory_->CreateFixedPointElementFromInt(0));
  // 0......0 || m_minus_b_fractional
  std::vector<FixedPointElement> m_minus_b_option0 (n, zero);
  // 1......1 || m_minus_b_fractional
  std::vector<FixedPointElement> m_minus_b_option1 (n, zero);
  for (size_t idx = 0; idx < n; idx++) {
    // 2^num_fractional_bits - 1  -> all ones in last fractional bits
    // AND with 0 ... 0 || 1^num_fractional_bits
    ASSIGN_OR_RETURN(m_minus_b_option0[idx],
                     fp_factory_->ImportFixedPointElementFromUint64(
                             m_minus_b_fractional[idx] &
                                     (fp_factory_->GetParams().fractional_multiplier - 1)));
    // OR with 1 ... 1 || 0^num_fractional_bits
    // 2^63-1 is all 1s, xor with 0 ... 0 || 1^num_fractional_bits to get the
    // mask
    ASSIGN_OR_RETURN(
            m_minus_b_option1[idx],
            fp_factory_->ImportFixedPointElementFromUint64(
                    m_minus_b_option0[idx].ExportToUint64() |
                            ((fp_factory_->GetParams().primary_ring_modulus - 1) ^
                                    (fp_factory_->GetParams().fractional_multiplier - 1))));
  }

  // Evaluate honey badger powers for both options

  // k is the max power to which [m] is raised
  size_t k = random_powers_share.size();

  ASSIGN_OR_RETURN(auto powers_share_option0, internal::HoneyBadgerPowersPartyZero(
          k, n, random_powers_share, m_minus_b_option0, fp_factory_, modulus));
  ASSIGN_OR_RETURN(auto powers_share_option1, internal::HoneyBadgerPowersPartyZero(
          k, n, random_powers_share, m_minus_b_option1, fp_factory_, modulus));

  // Sample random values for the share of shares of powers protocol
  std::vector<std::vector<uint64_t>> share_of_share(
          k, std::vector<uint64_t>(n, 0));
  auto random_seed = BasicRng::GenerateSeed();
  if (!random_seed.ok()) {
    return InternalError("Random seed generation fails.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<BasicRng> prng,
                   BasicRng::Create(random_seed.value()));
  for (size_t idx = 0; idx < k; idx++) {
    ASSIGN_OR_RETURN(auto random_share,
                     SampleVectorFromPrng(n, modulus, prng.get()));
    share_of_share[idx] = std::move(random_share);
  }
  // Order the powers to the OT (where party acts as a sender)
  // First input is at carry_bit xor p_bit
  // Second input is at carry_bit xor not(p_bit)
  std::vector<std::vector<uint64_t>> powers_input_0(
          k, std::vector<uint64_t>(n, 0));
  std::vector<std::vector<uint64_t>> powers_input_1(
          k, std::vector<uint64_t>(n, 0));
  for (size_t idx_n = 0; idx_n < n; idx_n++) {
    if (carry_bits[idx_n] ^ p_bits[idx_n]) {
      for (size_t idx_k = 0; idx_k < k; idx_k++) {
        powers_input_0[idx_k][idx_n] =
                ModSub(powers_share_option1[idx_k][idx_n],
                       share_of_share[idx_k][idx_n], modulus);
        powers_input_1[idx_k][idx_n] =
                ModSub(powers_share_option0[idx_k][idx_n],
                       share_of_share[idx_k][idx_n], modulus);
      }
    } else {
      for (size_t idx_k = 0; idx_k < k; idx_k++) {
        powers_input_0[idx_k][idx_n] =
                ModSub(powers_share_option0[idx_k][idx_n],
                       share_of_share[idx_k][idx_n], modulus);
        powers_input_1[idx_k][idx_n] =
                ModSub(powers_share_option1[idx_k][idx_n],
                       share_of_share[idx_k][idx_n], modulus);
      }
    }
  }

  PowersShareOfPowersShare share_of_share_struct = {
          .powers_share = std::move(share_of_share)};

  PowersStateOTInputs ot_inps = {.receiver_bit = std::move(p_bits),
          .powers_input_0 = std::move(powers_input_0),
          .powers_input_1 = std::move(powers_input_1)};

  return std::make_pair(ot_inps, share_of_share_struct);
}

StatusOr<std::pair<PowersStateOTInputs, PowersShareOfPowersShare>>
PowersGenerateOTInputsPartyOne(
        PowersStateMminusB state,
        const std::vector<std::vector<uint64_t>>& random_powers_share,
        PowersMessageMminusB other_party_message,
        std::unique_ptr<FixedPointElementFactory>& fp_factory_, uint64_t modulus) {
  if (state.share_m_minus_b_full.empty() ||
          state.share_m_minus_b_fractional.empty()) {
    return InvalidArgumentError("Powers: state must not be empty.");
  }

  // n is the number of instances of powers
  size_t n = state.share_m_minus_b_full.size();
  if (state.share_m_minus_b_fractional.size() != n) {
    return InvalidArgumentError("Powers: state has invalid dimensions.");
  }
  for (size_t idx = 0; idx < random_powers_share.size(); idx++) {
    if (n != random_powers_share[idx].size()) {
      return InvalidArgumentError(
              "Powers: incorrect number of preprocessed powers.");
    }
  }
  if (static_cast<size_t>(
          other_party_message.vector_m_minus_vector_b_shares_size()) != n) {
    return InvalidArgumentError("Powers: m - b message size must equal n.");
  }

  // Reconstruct (m - b), one for each of the n batched invocations
  // Recall this is (m - b)^fractional_1 as it is party 1 share and contains
  // only the last num_fractional_bits bits.
  std::vector<uint64_t> m_minus_b_fractional(n, 0);
  for (size_t idx = 0; idx < n; idx++) {
    m_minus_b_fractional[idx] =
            ModAdd(other_party_message.vector_m_minus_vector_b_shares(idx),
                   state.share_m_minus_b_fractional[idx], modulus);
  }

  // Elements of m_minus_b_fractional can have up to num_fractional_bits+1 bits
  // as there can be a carry

  // Save the carry
  std::vector<bool> carry_bits(n, false);
  for (size_t idx = 0; idx < n; idx++) {
    // AND m_minus_b_fractional with a mask of all zeros but a single one
    // in the num_fractional_bits position, then right shift to get the carry
    carry_bits[idx] = (m_minus_b_fractional[idx] &
            fp_factory_->GetParams().fractional_multiplier) >>
                                                            fp_factory_->GetParams().num_fractional_bits;
  }

  // Save q (bit at num_fractional_bits position of share of P_1's
  // m_minus_b_full) q = [m-b]^num_fractional_bits_P1
  std::vector<bool> q_bits(n, false);
  for (size_t idx = 0; idx < n; idx++) {
    // AND m_minus_b_full with a mask of all zeros but a single one
    // in the num_fractional_bits position, then right shift to get the carry
    q_bits[idx] = (state.share_m_minus_b_full[idx] &
            fp_factory_->GetParams().fractional_multiplier) >>
                                                            fp_factory_->GetParams().num_fractional_bits;
  }

  // Form the two options for m_minus_b
  ASSIGN_OR_RETURN(FixedPointElement zero,
                   fp_factory_->CreateFixedPointElementFromInt(0));

  // 0......0 || m_minus_b_fractional
  std::vector<FixedPointElement> m_minus_b_option0(n, zero);
  // 1......1 || m_minus_b_fractional
  std::vector<FixedPointElement> m_minus_b_option1(n, zero);
  for (size_t idx = 0; idx < n; idx++) {
    // 2^num_fractional_bits - 1  -> all ones in last fractional bits
    // AND with 0 ... 0 || 1^num_fractional_bits
    ASSIGN_OR_RETURN(m_minus_b_option0[idx],
                     fp_factory_->ImportFixedPointElementFromUint64(
                             m_minus_b_fractional[idx] &
                                     (fp_factory_->GetParams().fractional_multiplier - 1)));
    // OR with 1 ... 1 || 0^num_fractional_bits
    // -1 is all 1s, xor with 0 ... 0 || 1^num_fractional_bits to get the mask
    ASSIGN_OR_RETURN(
            m_minus_b_option1[idx],
            fp_factory_->ImportFixedPointElementFromUint64(
                    m_minus_b_option0[idx].ExportToUint64() |
                            ((fp_factory_->GetParams().primary_ring_modulus - 1) ^
                                    (fp_factory_->GetParams().fractional_multiplier - 1))));
  }

  // Evaluate honey badger powers for both options

  // k is the max power to which [m] is raised
  size_t k = random_powers_share.size();

  ASSIGN_OR_RETURN(auto powers_share_option0, internal::HoneyBadgerPowersPartyOne(
          k, n, random_powers_share, m_minus_b_option0, fp_factory_, modulus));
  ASSIGN_OR_RETURN(auto powers_share_option1, internal::HoneyBadgerPowersPartyOne(
          k, n, random_powers_share, m_minus_b_option1, fp_factory_, modulus));

  // Sample random values for the share of shares of powers protocol
  std::vector<std::vector<uint64_t>> share_of_share(
          k, std::vector<uint64_t>(n, 0));
  auto random_seed = BasicRng::GenerateSeed();
  if (!random_seed.ok()) {
    return InternalError("Random seed generation fails.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<BasicRng> prng,
                   BasicRng::Create(random_seed.value()));
  for (size_t idx = 0; idx < k; idx++) {
    ASSIGN_OR_RETURN(auto random_share,
                     SampleVectorFromPrng(n, modulus, prng.get()));
    share_of_share[idx] = std::move(random_share);
  }

  // Order the powers to the OT (where party acts as a sender)
  // First input is at carry_bit xor q_bit
  // Second input is at carry_bit xor not(q_bit)
  std::vector<std::vector<uint64_t>> powers_input_0(
          k, std::vector<uint64_t>(n, 0));
  std::vector<std::vector<uint64_t>> powers_input_1(
          k, std::vector<uint64_t>(n, 0));
  for (size_t idx_n = 0; idx_n < n; idx_n++) {
    if (carry_bits[idx_n] ^ q_bits[idx_n]) {
      for (size_t idx_k = 0; idx_k < k; idx_k++) {
        powers_input_0[idx_k][idx_n] =
                ModSub(powers_share_option1[idx_k][idx_n],
                       share_of_share[idx_k][idx_n], modulus);
        powers_input_1[idx_k][idx_n] =
                ModSub(powers_share_option0[idx_k][idx_n],
                       share_of_share[idx_k][idx_n], modulus);
      }
    } else {
      for (size_t idx_k = 0; idx_k < k; idx_k++) {
        powers_input_0[idx_k][idx_n] =
                ModSub(powers_share_option0[idx_k][idx_n],
                       share_of_share[idx_k][idx_n], modulus);
        powers_input_1[idx_k][idx_n] =
                ModSub(powers_share_option1[idx_k][idx_n],
                       share_of_share[idx_k][idx_n], modulus);
      }
    }
  }

  PowersShareOfPowersShare share_of_share_struct = {
          .powers_share = std::move(share_of_share)};

  PowersStateOTInputs ot_inps = {.receiver_bit = std::move(q_bits),
          .powers_input_0 = std::move(powers_input_0),
          .powers_input_1 = std::move(powers_input_1)};

  return std::make_pair(ot_inps, share_of_share_struct);
}

StatusOr<std::pair<PowersShareOfPowersShare, PowersShareOfPowersShare>>
PowersGenerateOTOutputForTesting(
        PowersStateOTInputs p_0_inputs,
        PowersStateOTInputs p_1_inputs,
        std::unique_ptr<FixedPointElementFactory>& fp_factory_, uint64_t modulus) {
  size_t k = p_0_inputs.powers_input_0.size();
  if (k != p_1_inputs.powers_input_0.size() ||
          k != p_1_inputs.powers_input_1.size() ||
          k != p_0_inputs.powers_input_1.size()) {
    return InvalidArgumentError(
            "PowersGenerateOTOutputForTesting: p_0_inputs and p_1_inputs "
            "dimensions inconsistent.");
  }
  size_t n = p_0_inputs.receiver_bit.size();
  for (size_t idx = 0; idx < k; idx++) {
    if (p_0_inputs.powers_input_0[idx].size() != n ||
            p_0_inputs.powers_input_1[idx].size() != n ||
            p_1_inputs.powers_input_0[idx].size() != n ||
            p_1_inputs.powers_input_1[idx].size() != n) {
      return InvalidArgumentError(
              "PowersGenerateOTOutputForTesting: p_0_inputs and p_1_inputs "
              "dimensions inconsistent.");
    }
  }
  // 1 share of shares of powers is received via OT
  std::vector<std::vector<uint64_t>> share_of_share_p0 (
          k, std::vector<uint64_t>(n, 0));
  std::vector<std::vector<uint64_t>> share_of_share_p1 (
          k, std::vector<uint64_t>(n, 0));
  for (size_t idx_n = 0; idx_n < n; idx_n++) {
    // Select the right OT output for P_0
    if (p_0_inputs.receiver_bit[idx_n]) {
      for (size_t idx_k = 0; idx_k < k; idx_k++) {
        share_of_share_p0[idx_k][idx_n] =
                p_1_inputs.powers_input_1[idx_k][idx_n];
      }
    } else {
      for (size_t idx_k = 0; idx_k < k; idx_k++) {
        share_of_share_p0[idx_k][idx_n] =
                p_1_inputs.powers_input_0[idx_k][idx_n];
      }
    }
    // Select the right OT output for P_1
    if (p_1_inputs.receiver_bit[idx_n]) {
      for (size_t idx_k = 0; idx_k < k; idx_k++) {
        share_of_share_p1[idx_k][idx_n] =
                p_0_inputs.powers_input_1[idx_k][idx_n];
      }
    } else {
      for (size_t idx_k = 0; idx_k < k; idx_k++) {
        share_of_share_p1[idx_k][idx_n] =
                p_0_inputs.powers_input_0[idx_k][idx_n];
      }
    }
  }
  PowersShareOfPowersShare output_p0 = {
          .powers_share = std::move(share_of_share_p0)
  };
  PowersShareOfPowersShare output_p1 = {
          .powers_share = std::move(share_of_share_p1)
  };
  return std::make_pair(output_p0, output_p1);
}

StatusOr<std::vector<std::vector<uint64_t>>>
PowersOutput(
        PowersShareOfPowersShare share0,
        PowersShareOfPowersShare share1,
        uint64_t modulus) {
  size_t k = share0.powers_share.size();
  if (k != share1.powers_share.size()) {
    return InvalidArgumentError(
            "PowersOutput: share0 and share1 dimensions mismatch.");
  }
  size_t n = share0.powers_share[0].size();
  for (size_t idx = 0; idx < k; idx++) {
    if (share0.powers_share[idx].size() != n ||
            share1.powers_share[idx].size() != n) {
      return InvalidArgumentError(
              "PowersOutput: share0 and share1 dimensions inconsistent.");
    }
  }
  // Reconstruct the shares to get [m, m^2, ..., m^k]
  std::vector<std::vector<uint64_t>> share (k, std::vector<uint64_t>(n, 0));
  for (size_t idx = 0; idx < k; idx++) {
    ASSIGN_OR_RETURN(share[idx],
                     BatchedModAdd(share0.powers_share[idx],
                                   share1.powers_share[idx],
                                   modulus));
  }
  return share;
}

}  // namespace private_join_and_compute
