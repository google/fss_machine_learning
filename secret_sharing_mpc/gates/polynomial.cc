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

#include "secret_sharing_mpc/gates/polynomial.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "secret_sharing_mpc/gates/hadamard_product.h"
#include "secret_sharing_mpc/gates/scalar_vector_product.h"

namespace private_join_and_compute {

namespace internal {

// This preprocessing function is only for testing purposes.
// Hence, it is in internal namespace.
StatusOr<std::pair<std::vector<std::vector<uint64_t>>,
                   std::vector<std::vector<uint64_t>>>>
PolynomialSamplePowersOfRandomVector(
    size_t k, size_t n, std::unique_ptr<FixedPointElementFactory>& fp_factory_,
    uint64_t modulus) {
  ASSIGN_OR_RETURN(auto random_powers,
      internal::SamplePowersOfRandomVector(k, n, fp_factory_, modulus));
  return random_powers;
}

// This preprocessing function is only for testing purposes.
// Hence, it is in internal namespace.
StatusOr<std::pair<PolynomialRandomOTPrecomputation, PolynomialRandomOTPrecomputation>>
PolynomialPreprocessRandomOTs (
    size_t n,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_) {
  if (n < 1) {
    return InvalidArgumentError(
        "PolynomialPreprocessRandomOTs: n must be a positive integer.");
  }

  // Generate n random bools
  std::vector<uint64_t> receiver_choice_vector_p0 (n, 0);
  std::vector<uint64_t> receiver_choice_vector_p1 (n, 0);
  // Generate send messages
  std::vector<uint64_t > sender_msg0_vector_p0 (n, 0);
  std::vector<uint64_t > sender_msg1_vector_p0 (n, 0);
  std::vector<uint64_t > sender_msg0_vector_p1 (n, 0);
  std::vector<uint64_t > sender_msg1_vector_p1 (n, 0);

  auto random_seed = BasicRng::GenerateSeed();
  if (!random_seed.ok()) {
    return InternalError("Random seed generation fails.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<BasicRng> prng,
                   BasicRng::Create(random_seed.value()));

  ASSIGN_OR_RETURN(receiver_choice_vector_p0,
                   SampleVectorFromPrng(n, 2, prng.get()));
  ASSIGN_OR_RETURN(receiver_choice_vector_p1,
                   SampleVectorFromPrng(n, 2, prng.get()));
  ASSIGN_OR_RETURN(sender_msg0_vector_p0,
                   SampleVectorFromPrng(
                       n, fp_factory_->GetParams().integer_ring_modulus, prng.get()));
  ASSIGN_OR_RETURN(sender_msg1_vector_p0,
                   SampleVectorFromPrng(n,
                                        fp_factory_->GetParams().integer_ring_modulus, prng.get()));
  ASSIGN_OR_RETURN(sender_msg0_vector_p1,
                   SampleVectorFromPrng(n,
                                        fp_factory_->GetParams().integer_ring_modulus, prng.get()));
  ASSIGN_OR_RETURN(sender_msg1_vector_p1,
                   SampleVectorFromPrng(n,
                                        fp_factory_->GetParams().integer_ring_modulus, prng.get()));

  std::vector<PolynomialRandomOTCorrelationSenderMessage> sender_msgs_p0 (n);
  std::vector<PolynomialRandomOTCorrelationReceiverMessage> receiver_msgs_p0 (n);
  std::vector<PolynomialRandomOTCorrelationSenderMessage> sender_msgs_p1 (n);
  std::vector<PolynomialRandomOTCorrelationReceiverMessage> receiver_msgs_p1 (n);

  for (size_t idx = 0; idx < n; idx++) {
    sender_msgs_p0[idx] = {
        .sender_msg0 = sender_msg0_vector_p0[idx],
        .sender_msg1 = sender_msg1_vector_p0[idx]
    };
    sender_msgs_p1[idx] = {
        .sender_msg0 = sender_msg0_vector_p1[idx],
        .sender_msg1 = sender_msg1_vector_p1[idx]
    };
    receiver_msgs_p0[idx] = {
        .receiver_choice = receiver_choice_vector_p0[idx] == 1,
        .receiver_msg = (receiver_choice_vector_p0[idx] == 1)
                        ? sender_msg1_vector_p1[idx]
                        : sender_msg0_vector_p1[idx]
    };
    receiver_msgs_p1[idx] = {
        .receiver_choice = receiver_choice_vector_p1[idx] == 1,
        .receiver_msg = (receiver_choice_vector_p1[idx] == 1)
                        ? sender_msg1_vector_p0[idx]
                        : sender_msg0_vector_p0[idx]
    };
  }
  PolynomialRandomOTPrecomputation p0 = {
      .sender_msgs = std::move(sender_msgs_p0),
      .receiver_msgs = std::move(receiver_msgs_p0)
  };
  PolynomialRandomOTPrecomputation p1 = {
      .sender_msgs = std::move(sender_msgs_p1),
      .receiver_msgs = std::move(receiver_msgs_p1)
  };

  return std::make_pair(p0, p1);
}

}  // namespace internal

StatusOr<std::pair<PowersStateRoundOne, PowersMessageRoundOne>>
PolynomialGenerateRoundOneMessage(
    const std::vector<uint64_t>& share_m,
    const std::vector<std::vector<uint64_t>>& random_powers_share,
    PolynomialRandomOTPrecomputation precomputed_ots,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_,
    uint64_t modulus) {
  if (share_m.empty() || random_powers_share.empty()
      || precomputed_ots.sender_msgs.empty() || precomputed_ots.receiver_msgs.empty()) {
    return InvalidArgumentError("PolynomialGenerateRoundOneMessage: "
                                "input must not be empty.");
  }

  // n is number of powers invocations
  size_t n = share_m.size();
  size_t k = random_powers_share.size();

  for (size_t idx = 0; idx < k; idx++) {
    if (n != random_powers_share[idx].size()) {
      return InvalidArgumentError("PolynomialGenerateRoundOneMessage: "
                                  "shares must have the same length.");
    }
  }

  if (precomputed_ots.sender_msgs.size() != n || precomputed_ots.receiver_msgs.size() != n) {
    return InvalidArgumentError("PolynomialGenerateRoundOneMessage: "
                                "precomputed random ots invalid dimension.");
  }

  ASSIGN_OR_RETURN(auto m_minus_b_full,
                   BatchedModSub(share_m, random_powers_share[0], modulus));
  // Save only the num_fractional_bits from m_minus_b_full
  std::vector<uint64_t> m_minus_b_fractional (n, 0);
  for (size_t idx = 0; idx < n; idx++) {
    // m_minus_b & (2^num_fractional_bits - 1)
    m_minus_b_fractional[idx] = m_minus_b_full[idx] &
        ((1ULL << fp_factory_->GetParams().num_fractional_bits) - 1);
  }

  // Save p or q (bit at num_fractional_bits position of
  // share of P_0/1's m_minus_b_full)
  // p (or q for P_1) = [m-b]^num_fractional_bits_P0/1
  std::vector<bool> ot_receiver_bit (n, false);
  for (size_t idx = 0; idx < n; idx++) {
    // AND m_minus_b_full with a mask of all zeros but a single one
    // in the num_fractional_bits position, then right shift to get the carry
    ot_receiver_bit[idx] = (m_minus_b_full[idx] &
        fp_factory_->GetParams().fractional_multiplier) >>
                                                        fp_factory_->GetParams().num_fractional_bits;
  }

  // Xor 1-2 OT receiver_bits with  random ot choice bits
  // This is part of the beaver 1-2 ot from 1-2 random ot protocol
  std::vector<bool> receiver_bit_xor_rot_choice_bit (n, false);
  for (size_t idx = 0; idx < n; idx++) {
    receiver_bit_xor_rot_choice_bit[idx] =
        ot_receiver_bit[idx] ^ precomputed_ots.receiver_msgs[idx].receiver_choice;
  }

  PowersMessageRoundOne round_one_msg;
  for (size_t idx = 0; idx < n; idx++) {
    round_one_msg.add_vector_m_minus_vector_b_shares(m_minus_b_fractional[idx]);
    round_one_msg.add_receiver_bit_xor_rot_choice_bit(receiver_bit_xor_rot_choice_bit[idx]);
  }
  PowersStateRoundOne round_one_state = {
      .ot_receiver_bit = std::move(ot_receiver_bit),
      .share_m_minus_b_fractional = std::move(m_minus_b_fractional)
  };
  return std::make_pair(std::move(round_one_state), std::move(round_one_msg));
}

StatusOr<std::pair<PolynomialShareOfPolynomialShare, PolynomialMessageRoundTwo>>
PolynomialGenerateRoundTwoMessagePartyZero(
    PolynomialCoefficients polynomial_coefficients,
    PowersStateRoundOne state,
    const std::vector<std::vector<uint64_t>>& random_powers_share,
    PolynomialRandomOTPrecomputation precomputed_ots,
    PowersMessageRoundOne other_party_message,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_, uint64_t modulus) {
  if (state.ot_receiver_bit.empty() ||
      state.share_m_minus_b_fractional.empty()) {
    return InvalidArgumentError("Polynomial: state must not be empty.");
  }

  // n is the number of instances of powers
  size_t n = state.share_m_minus_b_fractional.size();

  // k is the max power to which [m] is raised
  size_t k = random_powers_share.size();

  if (state.ot_receiver_bit.size() != n ||
      state.share_m_minus_b_fractional.size() != n) {
    return InvalidArgumentError("Polynomial: state has invalid dimensions.");
  }
  for (size_t idx = 0; idx < k; idx++) {
    if (n != random_powers_share[idx].size()) {
      return InvalidArgumentError(
          "Polynomial: incorrect number of preprocessed powers.");
    }
  }

  if (static_cast<size_t>(
      other_party_message.vector_m_minus_vector_b_shares_size()) != n) {
    return InvalidArgumentError("Polynomial: m - b message size must equal n.");
  }

  if (static_cast<size_t>(
      other_party_message.receiver_bit_xor_rot_choice_bit_size()) != n) {
    return InvalidArgumentError("Polynomial: ot receiver bit xor random ot choice bit size must equal n.");
  }

  if (k != (polynomial_coefficients.coefficients.size() - 1)) {
    return InvalidArgumentError("Polynomial: coefficients vector has invalid dimensions.");
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

  ASSIGN_OR_RETURN(auto powers_share_option0, internal::HoneyBadgerPowersPartyZero(
      k, n, random_powers_share, m_minus_b_option0, fp_factory_, modulus));
  ASSIGN_OR_RETURN(auto powers_share_option1, internal::HoneyBadgerPowersPartyZero(
      k, n, random_powers_share, m_minus_b_option1, fp_factory_, modulus));
  // Evaluate the polynomial
  // For Party 0, initialize to a_0 (parties need a share of a_0)
  // P_0: a_0 (in uint64_t)
  // P_1: 0
  ASSIGN_OR_RETURN(FixedPointElement first_coef,
                   fp_factory_->CreateFixedPointElementFromDouble(
                       polynomial_coefficients.coefficients[0]));
  std::vector<uint64_t> polynomial_share_option0 (n, first_coef.ExportToUint64());
  std::vector<uint64_t> polynomial_share_option1 (n, first_coef.ExportToUint64());

  // Polynomial sum for option 0
  for (size_t idx = 0; idx < k; idx++) {
    // a_{idx+1} * m^{idx+1}
    // a has size k+1, while m has size k (reason for offset below)
    ASSIGN_OR_RETURN(auto a_times_m_idx_share,
                     ScalarVectorProductPartyZero(polynomial_coefficients.coefficients[idx + 1],
                                                  powers_share_option0[idx], fp_factory_, modulus));
    ASSIGN_OR_RETURN(polynomial_share_option0, BatchedModAdd(
        polynomial_share_option0, a_times_m_idx_share, modulus));
  }

  // Polynomial sum for option 1
  for (size_t idx = 0; idx < k; idx++) {
    // a_{idx+1} * m^{idx+1}
    // a has size k+1, while m has size k (reason for offset below)
    ASSIGN_OR_RETURN(auto a_times_m_idx_share,
                     ScalarVectorProductPartyZero(polynomial_coefficients.coefficients[idx + 1],
                                                  powers_share_option1[idx], fp_factory_, modulus));
    ASSIGN_OR_RETURN(polynomial_share_option1, BatchedModAdd(
        polynomial_share_option1, a_times_m_idx_share, modulus));
  }

  // Sample random values for the share of shares of polynomial protocol
  std::vector<uint64_t> share_of_share(n, 0);
  auto random_seed = BasicRng::GenerateSeed();
  if (!random_seed.ok()) {
    return InternalError("Random seed generation fails.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<BasicRng> prng,
                   BasicRng::Create(random_seed.value()));
  ASSIGN_OR_RETURN(share_of_share,
                   SampleVectorFromPrng(n, fp_factory_->GetParams().integer_ring_modulus, prng.get()));

  // Subtract the share from the OT inputs
  ASSIGN_OR_RETURN(polynomial_share_option0, BatchedModSub(
      polynomial_share_option0, share_of_share, modulus));
  ASSIGN_OR_RETURN(polynomial_share_option1, BatchedModSub(
      polynomial_share_option1, share_of_share, modulus));

  // Order the powers to the OT (where party acts as a sender)
  // First input is at carry_bit xor p
  // Second input is at carry_bit xor not(p)
  std::vector<uint64_t> polynomial_input_0(n, 0);
  std::vector<uint64_t> polynomial_input_1(n, 0);
  for (size_t idx_n = 0; idx_n < n; idx_n++) {
    if (carry_bits[idx_n] ^ state.ot_receiver_bit[idx_n]) {
      polynomial_input_0[idx_n] = polynomial_share_option1[idx_n];
      polynomial_input_1[idx_n] = polynomial_share_option0[idx_n];
    } else {
      polynomial_input_0[idx_n] = polynomial_share_option0[idx_n];
      polynomial_input_1[idx_n] = polynomial_share_option1[idx_n];
    }
  }

  // Mask the polynomial_input_{0,1}
  std::vector<uint64_t> masked_polynomial_input_0 (n, 0);
  std::vector<uint64_t> masked_polynomial_input_1 (n, 0);
  for (size_t idx = 0; idx < n; idx++) {
    uint64_t mask_0 = other_party_message.receiver_bit_xor_rot_choice_bit(idx)
                      ? precomputed_ots.sender_msgs[idx].sender_msg1
                      : precomputed_ots.sender_msgs[idx].sender_msg0;
    masked_polynomial_input_0[idx] = ModSub(polynomial_input_0[idx], mask_0, modulus);
    uint64_t mask_1 = other_party_message.receiver_bit_xor_rot_choice_bit(idx)
                      ? precomputed_ots.sender_msgs[idx].sender_msg0
                      : precomputed_ots.sender_msgs[idx].sender_msg1;
    masked_polynomial_input_1[idx] = ModSub(polynomial_input_1[idx], mask_1, modulus);
  }

  PolynomialShareOfPolynomialShare share_of_share_struct = {
      .polynomial_share = std::move(share_of_share)};

  PolynomialMessageRoundTwo round_two_msg;
  for (size_t idx = 0; idx < n; idx++) {
    round_two_msg.add_polynomial_input_0(masked_polynomial_input_0[idx]);
    round_two_msg.add_polynomial_input_1(masked_polynomial_input_1[idx]);
  }

  return std::make_pair(share_of_share_struct, round_two_msg);
}

StatusOr<std::pair<PolynomialShareOfPolynomialShare, PolynomialMessageRoundTwo>>
PolynomialGenerateRoundTwoMessagePartyOne(
    PolynomialCoefficients polynomial_coefficients,
    PowersStateRoundOne state,
    const std::vector<std::vector<uint64_t>>& random_powers_share,
    PolynomialRandomOTPrecomputation precomputed_ots,
    PowersMessageRoundOne other_party_message,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_, uint64_t modulus) {
  if (state.ot_receiver_bit.empty() ||
      state.share_m_minus_b_fractional.empty()) {
    return InvalidArgumentError("Polynomial: state must not be empty.");
  }

  // n is the number of instances of powers
  size_t n = state.share_m_minus_b_fractional.size();

  // k is the max power to which [m] is raised
  size_t k = random_powers_share.size();

  if (state.ot_receiver_bit.size() != n ||
      state.share_m_minus_b_fractional.size() != n) {
    return InvalidArgumentError("Polynomial: state has invalid dimensions.");
  }
  for (size_t idx = 0; idx < k; idx++) {
    if (n != random_powers_share[idx].size()) {
      return InvalidArgumentError(
          "Polynomial: incorrect number of preprocessed powers.");
    }
  }

  if (n !=
      static_cast<size_t>(other_party_message.vector_m_minus_vector_b_shares_size())) {
    return InvalidArgumentError("Polynomial: m - b message size must equal n.");
  }

  if (n !=
      static_cast<size_t>(other_party_message.receiver_bit_xor_rot_choice_bit_size())) {
    return InvalidArgumentError("Polynomial: ot receiver bit xor random ot choice bit size must equal n.");
  }

  if (k != (polynomial_coefficients.coefficients.size() - 1)) {
    return InvalidArgumentError("Polynomial: coefficients vector has invalid dimensions.");
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

  ASSIGN_OR_RETURN(auto powers_share_option0, internal::HoneyBadgerPowersPartyOne(
      k, n, random_powers_share, m_minus_b_option0, fp_factory_, modulus));
  ASSIGN_OR_RETURN(auto powers_share_option1, internal::HoneyBadgerPowersPartyOne(
      k, n, random_powers_share, m_minus_b_option1, fp_factory_, modulus));

  // Evaluate the polynomial
  // For Party 1, initialize to 0 (parties need a share of a_0)
  // P_0: a_0
  // P_1: 0
  std::vector<uint64_t> polynomial_share_option0 (n, 0);
  std::vector<uint64_t> polynomial_share_option1 (n, 0);

  // Polynomial sum for option 0
  for (size_t idx = 0; idx < k; idx++) {
    // a_{idx+1} * m^{idx+1}
    // a has size k+1, while m has size k (reason for offset below)
    ASSIGN_OR_RETURN(auto a_times_m_idx_share,
                     ScalarVectorProductPartyOne(polynomial_coefficients.coefficients[idx + 1],
                                                 powers_share_option0[idx], fp_factory_, modulus));
    ASSIGN_OR_RETURN(polynomial_share_option0, BatchedModAdd(
        polynomial_share_option0, a_times_m_idx_share, modulus));
  }

  // Polynomial sum for option 1
  for (size_t idx = 0; idx < k; idx++) {
    // a_{idx+1} * m^{idx+1}
    // a has size k+1, while m has size k (reason for offset below)
    ASSIGN_OR_RETURN(auto a_times_m_idx_share,
                     ScalarVectorProductPartyOne(polynomial_coefficients.coefficients[idx + 1],
                                                 powers_share_option1[idx], fp_factory_, modulus));
    ASSIGN_OR_RETURN(polynomial_share_option1, BatchedModAdd(
        polynomial_share_option1, a_times_m_idx_share, modulus));
  }

  // Sample random values for the share of shares of polynomial protocol
  std::vector<uint64_t> share_of_share(n, 0);
  auto random_seed = BasicRng::GenerateSeed();
  if (!random_seed.ok()) {
    return InternalError("Random seed generation fails.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<BasicRng> prng,
                   BasicRng::Create(random_seed.value()));
  ASSIGN_OR_RETURN(share_of_share,
                   SampleVectorFromPrng(n, fp_factory_->GetParams().integer_ring_modulus, prng.get()));

  // Subtract the share from the OT inputs
  ASSIGN_OR_RETURN(polynomial_share_option0, BatchedModSub(
      polynomial_share_option0, share_of_share, modulus));
  ASSIGN_OR_RETURN(polynomial_share_option1, BatchedModSub(
      polynomial_share_option1, share_of_share, modulus));

  // Order the powers to the OT (where party acts as a sender)
  // First input is at carry_bit xor q
  // Second input is at carry_bit xor not(q)
  std::vector<uint64_t> polynomial_input_0(n, 0);
  std::vector<uint64_t> polynomial_input_1(n, 0);
  for (size_t idx_n = 0; idx_n < n; idx_n++) {
    if (carry_bits[idx_n] ^ state.ot_receiver_bit[idx_n]) {
      polynomial_input_0[idx_n] = polynomial_share_option1[idx_n];
      polynomial_input_1[idx_n] = polynomial_share_option0[idx_n];
    } else {
      polynomial_input_0[idx_n] = polynomial_share_option0[idx_n];
      polynomial_input_1[idx_n] = polynomial_share_option1[idx_n];
    }
  }

  // Mask the polynomial_input_{0,1}
  std::vector<uint64_t> masked_polynomial_input_0 (n, 0);
  std::vector<uint64_t> masked_polynomial_input_1 (n, 0);
  for (size_t idx = 0; idx < n; idx++) {
    uint64_t mask_0 = other_party_message.receiver_bit_xor_rot_choice_bit(idx)
                      ? precomputed_ots.sender_msgs[idx].sender_msg1
                      : precomputed_ots.sender_msgs[idx].sender_msg0;
    masked_polynomial_input_0[idx] = ModSub(polynomial_input_0[idx], mask_0, modulus);
    uint64_t mask_1 = other_party_message.receiver_bit_xor_rot_choice_bit(idx)
                      ? precomputed_ots.sender_msgs[idx].sender_msg0
                      : precomputed_ots.sender_msgs[idx].sender_msg1;
    masked_polynomial_input_1[idx] = ModSub(polynomial_input_1[idx], mask_1, modulus);
  }

  PolynomialShareOfPolynomialShare share_of_share_struct = {
      .polynomial_share = std::move(share_of_share)};

  PolynomialMessageRoundTwo round_two_msg;
  for (size_t idx = 0; idx < n; idx++) {
    round_two_msg.add_polynomial_input_0(masked_polynomial_input_0[idx]);
    round_two_msg.add_polynomial_input_1(masked_polynomial_input_1[idx]);
  }

  return std::make_pair(share_of_share_struct, round_two_msg);
}

StatusOr<std::vector<uint64_t>>
PolynomialOutput(
    PolynomialMessageRoundTwo other_party_message,  // sender messages
    PolynomialRandomOTPrecomputation precomputed_ots, // random ot choice bit,receiver mask
    PolynomialShareOfPolynomialShare random_share,
    PowersStateRoundOne state, // p or q
    uint64_t modulus) {
  size_t n = state.ot_receiver_bit.size();
  if (n != random_share.polynomial_share.size() ||
      n != precomputed_ots.receiver_msgs.size()) {
    return InvalidArgumentError(
        "PolynomialOutput: dimensions mismatch.");
  }
  if (static_cast<size_t>(other_party_message.polynomial_input_0_size()) != n ||
      static_cast<size_t>(other_party_message.polynomial_input_1_size()) != n) {
    return InvalidArgumentError("Polynomial: polynomial_input message size must equal n.");
  }
  std::vector<uint64_t> polynomial_share (n, 0);
  for (size_t idx = 0; idx < n; idx++) {
    if (state.ot_receiver_bit[idx]) {
      polynomial_share[idx] = ModAdd(other_party_message.polynomial_input_1(idx),
                                     precomputed_ots.receiver_msgs[idx].receiver_msg, modulus);
    } else {
      polynomial_share[idx] = ModAdd(other_party_message.polynomial_input_0(idx),
                                     precomputed_ots.receiver_msgs[idx].receiver_msg, modulus);
    }
  }
  ASSIGN_OR_RETURN(polynomial_share,
                   BatchedModAdd(polynomial_share, random_share.polynomial_share, modulus));
  return polynomial_share;
}

}  // namespace private_join_and_compute
