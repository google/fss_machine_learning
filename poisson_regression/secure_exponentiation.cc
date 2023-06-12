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

#include "poisson_regression/secure_exponentiation.h"

#include <cstdint>

#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status.inc"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace private_join_and_compute {
namespace {

const double kLog2OfE = 1.4426950408889634073599;

// Returns true if the exponentiation params are valid.
// Specifically, exp_params is valid if
// fractional_multiplier * e^{2 * exponent_bound + 1} < prime_q
// Since, we first add a constant (exponent_bound  + 1) to the exponent x, to
// ensure that the exponent in the intger exponentiation is > 0, we need to make
// sure that the maximum possible computed value can fit in \Z_{prime_q}
//
// Does not check if prime_q is indeed prime.
bool CheckSecureExponentiationParameters(
    const ExponentiationParams& exp_params,
    uint64_t fractional_multiplier,
    uint64_t num_ring_bits) {

  // Check if \Z_{prime_q} is large enough.
  if (exp_params.exponent_bound >= 22) {
    // e^{22 * 2 + 1} is already not representable as a uint64_t.
    return false;
  }
  uint64_t max_exponentiation_result = static_cast<uint64_t>(
      exp(2 * exp_params.exponent_bound + 1));
  if (max_exponentiation_result >= exp_params.prime_q / fractional_multiplier) {
    return false;
  }

  // Check that exponent bound is small enough for division to work.
  uint64_t base2_exp_bound = ceil(kLog2OfE * exp_params.exponent_bound + 1);
  if (base2_exp_bound >= num_ring_bits) {
    return false;
  }
  return true;
}

bool CheckAlphaBetaValuesInRing(
    const ExponentiationParams& exp_params,
    uint64_t alpha, uint64_t beta) {
  if (0 > alpha || alpha >= exp_params.prime_q
      || 0 > beta || beta >= exp_params.prime_q) {
    return false;
  }
  return true;
}

// Splits the value of the given FixedPointElement fpe_share into its integer
// and fractional parts.
// Returns (integer_share, fractional_share) such that
// integer_share is an additive share in \Z_{integer_ring_modulus}
// of the integer part, and fractional part is a double.
// For a secret shared FixedPointElement with real value w.f where w is the
// whole number part and f is the fractional part, running the function locally
// on each party's share will result in one of the two cases:
// (1) The returned integer_share are shares of w and the returned
// fractional_share add up to (0.f).
// (2) The returned integer_share are shares of (w-1) and the returned
// fractional_share add up to (1.f).
std::pair<uint64_t, double> SplitIntoIntegerAndFractionalParts(
    const FixedPointElement& fpe_share) {
  uint64_t fpe_share_value = fpe_share.ExportToUint64();
  uint64_t fractional_multiplier = fpe_share.GetElementParams().
      fractional_multiplier;
  uint64_t integer_share = fpe_share_value / fractional_multiplier;
  double fractional_share =
      (1.0 * (fpe_share_value % fractional_multiplier)) / fractional_multiplier;
  return std::make_pair(integer_share, fractional_share);
}

// Takes as input a share of the integer part in Z_{prime_q - 1} and a share
// of the fractional part and returns a multiplicative share of the combined
// exponentiation (with an additional fractional_multiplier factor)
// in Z_{prime_q}.
uint64_t ExponentiateAndCombine(
    uint64_t integer_share, double fractional_share,
    uint64_t fractional_multiplier, uint64_t prime_q) {

  // Locally exponentiate to give mult share in Z_{prime_q} of the integer_share
  // (due to Fermat's Little Theorem).
  uint64_t int_exp_result = ModExp(2, integer_share, prime_q);

  // Exponentiate the fractional part in real numbers.
  double fractional_exp_result = pow(2, fractional_share);

  // Combine integer and fractional parts.
  uint64_t combined_mult_share = ModMul(
      int_exp_result, fractional_exp_result * fractional_multiplier, prime_q);

  return combined_mult_share;
}

}  // namespace


SecureExponentiationPartyZero::SecureExponentiationPartyZero(
    std::unique_ptr<FixedPointElementFactory> fpe_factory,
    const FixedPointElementFactory::Params* fpe_params,
    const ExponentiationParams* exp_params,
    std::unique_ptr<FixedPointElement> logbase2_e_fpe,
    std::unique_ptr<FixedPointElement> exp_bound_adder)
    : fpe_factory_(std::move(fpe_factory)),
      fpe_params_(fpe_params),
      exp_params_(exp_params),
      logbase2_e_fpe_(std::move(logbase2_e_fpe)),
      exp_bound_adder_(std::move(exp_bound_adder)),
      two_power_base2_bound_(1 << static_cast<uint64_t>(
        ceil(kLog2OfE * exp_params->exponent_bound + 1))) {}

StatusOr<std::unique_ptr<SecureExponentiationPartyZero>>
  SecureExponentiationPartyZero::Create(
      const FixedPointElementFactory::Params& fpe_params,
      const ExponentiationParams& exp_params) {
  if (!CheckSecureExponentiationParameters(
          exp_params,
          fpe_params.fractional_multiplier, fpe_params.num_ring_bits)) {
    return InvalidArgumentError(
        "SecureExponentiationPartyZero::Create: Invalid "
        "ExponentiationParams. prime_q needs to be prime and should be large "
        "enough to fit the maximum exponentiation result");
  }

  // Create a FixedPointElementFactory.
  ASSIGN_OR_RETURN(auto temp_factory,
                   FixedPointElementFactory::Create(
                       fpe_params.num_fractional_bits,
                       fpe_params.num_ring_bits));
  auto fpe_factory = absl::make_unique<FixedPointElementFactory>(
         std::move(temp_factory));

  // Create a FixedPointElement representing log_2(e).
  ASSIGN_OR_RETURN(auto temp_fpe,
                   fpe_factory->CreateFixedPointElementFromDouble(kLog2OfE));
  auto logbase2_e_fpe = absl::make_unique<FixedPointElement>(
         std::move(temp_fpe));

  // Create a FixedPointElement representing the base 2 exponent bound
  ASSIGN_OR_RETURN(auto temp_base2_bound,
                   fpe_factory->CreateFixedPointElementFromInt(
                       ceil(kLog2OfE * exp_params.exponent_bound + 1)));
  auto base2_exp_bound = absl::make_unique<FixedPointElement>(
         std::move(temp_base2_bound));

  return absl::WrapUnique(new SecureExponentiationPartyZero(
      std::move(fpe_factory), &fpe_params, &exp_params,
      std::move(logbase2_e_fpe), std::move(base2_exp_bound)));
}

StatusOr<std::pair<ExponentiationPartyZeroMultToAddMessage,
                  SecureExponentiationPartyZero::State>>
SecureExponentiationPartyZero::GenerateMultToAddMessage(
        const FixedPointElement& fpe_share_zero,
        uint64_t alpha_zero,
        uint64_t beta_zero) {
  if (!CheckAlphaBetaValuesInRing(*exp_params_, alpha_zero, beta_zero)) {
     return InvalidArgumentError(
        absl::StrCat("SecureExponentiationPartyZero::GenerateMultToAddMessage: "
                     "alpha_zero and beta_zero must be non-negative and less "
                     "than prime_q. Given: ", alpha_zero, " ", beta_zero));
  }
  // Convert to Base 2 exponent.
  ASSIGN_OR_RETURN(auto base2_fpe, fpe_share_zero.TruncMul(*logbase2_e_fpe_));

  // Make exponent >= 1.
  // This ensures that the integer exponentiation part works only with a
  // non-negative exponent.
  ASSIGN_OR_RETURN(auto positive_base2_fpe,
                   base2_fpe.ModAdd(*exp_bound_adder_));

  // Split into integer and fractional parts.
  auto split_parts = SplitIntoIntegerAndFractionalParts(positive_base2_fpe);
  auto integer_share = split_parts.first;
  auto fractional_share = split_parts.second;

  // Convert integer part to share in Z_{prime_q - 1}
  // Intuitively, for a randomly chosen secret sharing, with overwhelming
  // probability (1 - xmax/integer_ring_modulus)
  // where xmax is the maximum value of x, it holds that
  // x_0 + x_1 = integer_ring_modulus + x
  // where x_0 and x_1 are the two shares of x held by P_0 and P_1.
  // Now,
  // (x_0 + prime_q - 1 - integer_ring_modulus) + x_1 = (prime_q - 1) + x
  // This means that changing x_0 to (x_0 + prime_q - 1 -
  // integer_ring_modulus) mod (prime_q - 1) results in the two parties now
  // holding shares of x in Z_{prime_q - 1}.
  uint64_t int_share_in_q_minus_1 =
      ModAdd(integer_share,
             exp_params_->prime_q - 1 - fpe_params_->integer_ring_modulus,
             exp_params_->prime_q - 1);

  // Exponentiate integer and fractional parts separately and combine in
  // Z_{prime_q}.
  uint64_t mult_share_zero = ExponentiateAndCombine(
      int_share_in_q_minus_1, fractional_share,
      fpe_params_->fractional_multiplier, exp_params_->prime_q);

  // Conversion to additive shares in Z_{prime_q} is done by multiplying by
  // beta and sending a message to P_1.
  uint64_t beta_times_mult_share = ModMul(beta_zero, mult_share_zero,
                                          exp_params_->prime_q);

  // Create message to send to P_1.
  ExponentiationPartyZeroMultToAddMessage party_zero_message;
  party_zero_message.set_beta_zero_times_mult_share_zero(beta_times_mult_share);

  // Create state struct.
  State state = {mult_share_zero, alpha_zero, beta_zero};
  return std::make_pair(party_zero_message, state);
}

StatusOr<std::pair<BatchedExponentiationPartyZeroMultToAddMessage,
                   std::vector<SecureExponentiationPartyZero::State>>>
SecureExponentiationPartyZero::GenerateBatchedMultToAddMessage(
    const std::vector<FixedPointElement>& fpe_shares_zero,
    const std::vector<uint64_t>& alpha_zero,
    const std::vector<uint64_t>& beta_zero) {
  size_t length = fpe_shares_zero.size();
  if (alpha_zero.size() != length || beta_zero.size() != length) {
    return InvalidArgumentError(
        "GenerateBatchedMultToAddMessage: alpha_zero and beta_zero and "
        "fpe_shares_zero must have the same length.");
  }

  // Create message to send to P_1.
  BatchedExponentiationPartyZeroMultToAddMessage party_zero_message;
  // Create states struct.
  std::vector<State> states;

  for (size_t idx = 0; idx < length; idx++) {
    // Generate a message and state for a given share of the exponent
    // fpe_shares_zero, and a preprocessed tuple alpha, beta
    ASSIGN_OR_RETURN(auto p0_return,
                     GenerateMultToAddMessage(fpe_shares_zero[idx],
                                             alpha_zero[idx],
                                             beta_zero[idx]));
    // Add the message and state for the batched implementation
    party_zero_message.add_beta_zero_times_mult_share_zero(
        p0_return.first.beta_zero_times_mult_share_zero());
    states.push_back(p0_return.second);
  }
  return std::make_pair(party_zero_message, states);
}

StatusOr<FixedPointElement> SecureExponentiationPartyZero::OutputResult(
    const ExponentiationPartyOneMultToAddMessage& party_one_msg,
    const State& self_state) {
  uint64_t result_from_mult = ModMul(
      ModMul(self_state.mult_share_zero,
             self_state.alpha_zero,
             exp_params_->prime_q),
      party_one_msg.alpha_one_times_mult_share_one(),
      exp_params_->prime_q);

  // GenerateMultToAddMessage has an additional fractional_multiplier factor.
  // Furthermore, it added the base two exponent bound to the exponent to
  // ensure that it was >= 1.
  // Divide by these to get the final share in Z_Q.
  uint64_t additive_share_q = ModSub(
      exp_params_->prime_q,
      (exp_params_->prime_q - result_from_mult)
        / (fpe_params_->fractional_multiplier * two_power_base2_bound_),
      exp_params_->prime_q);

  // Convert to share in Z_{primary_ring_modulus}.
  uint64_t final_additive_share = ModAdd(
      additive_share_q,
      fpe_params_->primary_ring_modulus - exp_params_->prime_q,
      fpe_params_->primary_ring_modulus);

  // Create the FixedPointElement.
  ASSIGN_OR_RETURN(auto fpe_share,
                   fpe_factory_->ImportFixedPointElementFromUint64(
                       final_additive_share));
  return fpe_share;
}

StatusOr<std::vector<FixedPointElement>>
SecureExponentiationPartyZero::BatchedOutputResult(
    const BatchedExponentiationPartyOneMultToAddMessage& party_one_msg,
    const std::vector<State>& self_state) {
  size_t length = self_state.size();
  ASSIGN_OR_RETURN(FixedPointElement zero,
                   fpe_factory_->CreateFixedPointElementFromInt(0));
  std::vector<FixedPointElement> fpe_shares(length, zero);
  for (size_t idx = 0; idx < length; idx++) {
    // Retrieve the exponentiation message for a single party
    ExponentiationPartyOneMultToAddMessage party_one_single_message;
    party_one_single_message.set_alpha_one_times_mult_share_one(
        party_one_msg.alpha_one_times_mult_share_one(idx));

    ASSIGN_OR_RETURN(fpe_shares[idx],
                     OutputResult(party_one_single_message, self_state[idx]));
  }
  return fpe_shares;
}

SecureExponentiationPartyOne::SecureExponentiationPartyOne(
    std::unique_ptr<FixedPointElementFactory> fpe_factory,
    const FixedPointElementFactory::Params* fpe_params,
    const ExponentiationParams* exp_params,
    std::unique_ptr<FixedPointElement> logbase2_e_fpe)
    : fpe_factory_(std::move(fpe_factory)),
      fpe_params_(fpe_params),
      exp_params_(exp_params),
      logbase2_e_fpe_(std::move(logbase2_e_fpe)),
      two_power_base2_bound_(1 << static_cast<uint64_t>(
        ceil(kLog2OfE * exp_params->exponent_bound + 1))) {}

StatusOr<std::unique_ptr<SecureExponentiationPartyOne>>
  SecureExponentiationPartyOne::Create(
      const FixedPointElementFactory::Params& fpe_params,
      const ExponentiationParams& exp_params) {
  if (!CheckSecureExponentiationParameters(
          exp_params,
          fpe_params.fractional_multiplier, fpe_params.num_ring_bits)) {
    return InvalidArgumentError(
        "SecureExponentiationPartyOne::Create: Invalid "
        "ExponentiationParams. prime_q needs to be prime and should be large "
        "enough to fit the maximum exponentiation result");
  }

  // Create a FixedPointElementFactory.
  ASSIGN_OR_RETURN(auto temp_factory,
                   FixedPointElementFactory::Create(
                       fpe_params.num_fractional_bits,
                       fpe_params.num_ring_bits));
  auto fpe_factory = absl::make_unique<FixedPointElementFactory>(
         std::move(temp_factory));

  // Create a FixedPointElement representing log_2(e).
  ASSIGN_OR_RETURN(auto temp_fpe,
                   fpe_factory->CreateFixedPointElementFromDouble(kLog2OfE));
  auto logbase2_e_fpe = absl::make_unique<FixedPointElement>(
         std::move(temp_fpe));

  return absl::WrapUnique(new SecureExponentiationPartyOne(
      std::move(fpe_factory), &fpe_params, &exp_params,
      std::move(logbase2_e_fpe)));
}

StatusOr<std::pair<ExponentiationPartyOneMultToAddMessage,
                  SecureExponentiationPartyOne::State>>
  SecureExponentiationPartyOne::GenerateMultToAddMessage(
        const FixedPointElement& fpe_share_one,
        uint64_t alpha_one,
        uint64_t beta_one) {
  if (!CheckAlphaBetaValuesInRing(*exp_params_, alpha_one, beta_one)) {
     return InvalidArgumentError(
        absl::StrCat("SecureExponentiationPartyOne::GenerateMultToAddMessage: "
                     "alpha_one and beta_one must be non-negative and less "
                     "than prime_q. Given: ", alpha_one, " ", beta_one));
  }

  // Convert to Base 2 exponent.
  // This is computed as (c x f) - (c x m)
  // where c is fpe for logbase_2_e, f is the input fpe share,
  // m is the primary ring modulus and 'x' is the truncated multiplication
  // operation.
  ASSIGN_OR_RETURN(auto first_term_fpe,
                   fpe_share_one.TruncMul(*logbase2_e_fpe_));

  uint64_t second_term_uint = absl::Uint128Low64(
      (absl::uint128(logbase2_e_fpe_->ExportToUint64())
       * absl::uint128(fpe_params_->primary_ring_modulus))
      / fpe_params_->fractional_multiplier)
      % fpe_params_->primary_ring_modulus;
  ASSIGN_OR_RETURN(auto second_term_fpe,
                   fpe_factory_->ImportFixedPointElementFromUint64(
                       second_term_uint));

  ASSIGN_OR_RETURN(auto base2_fpe, first_term_fpe - second_term_fpe);

  // Split into integer and fractional parts.
  auto split_pair = SplitIntoIntegerAndFractionalParts(base2_fpe);

  // Exponentiate integer and fractional parts separately and combine in
  // Z_{prime_q}.
  // For P_1, split_pair.first is already a share in Z_{prime_q - 1}.
  uint64_t mult_share_one = ExponentiateAndCombine(
      split_pair.first, split_pair.second,
      fpe_params_->fractional_multiplier, exp_params_->prime_q);

  // Conversion to additive shares in Z_{prime_q} is done by multiplying by
  // alpha and sending a message to P_0.
  uint64_t alpha_one_times_mult_share = ModMul(alpha_one, mult_share_one,
                                               exp_params_->prime_q);

  // Create message to send to P_0.
  ExponentiationPartyOneMultToAddMessage party_one_message;
  party_one_message.set_alpha_one_times_mult_share_one(
      alpha_one_times_mult_share);

  // Create state struct.
  State state = {mult_share_one, alpha_one, beta_one};

  return std::make_pair(party_one_message, state);
}

StatusOr<std::pair<BatchedExponentiationPartyOneMultToAddMessage,
                   std::vector<SecureExponentiationPartyOne::State>>>
SecureExponentiationPartyOne::GenerateBatchedMultToAddMessage(
    const std::vector<FixedPointElement>& fpe_shares_one,
    const std::vector<uint64_t>& alpha_one,
    const std::vector<uint64_t>& beta_one) {
  size_t length = fpe_shares_one.size();
  if (alpha_one.size() != length || beta_one.size() != length) {
    return InvalidArgumentError(
        "GenerateBatchedMultToAddMessage: alpha_one and beta_one and "
        "fpe_shares_one must have the same length.");
  }

  // Create message to send to P_0.
  BatchedExponentiationPartyOneMultToAddMessage party_one_message;
  // Create states struct.
  std::vector<State> states;

  for (size_t idx = 0; idx < length; idx++) {
    // Generate a message and state for a given share of the exponent
    // fpe_shares_one, and a preprocessed tuple alpha, beta
    ASSIGN_OR_RETURN(auto p1_return,
                     GenerateMultToAddMessage(fpe_shares_one[idx],
                                             alpha_one[idx],
                                             beta_one[idx]));
    // Add the message and state for the batched implementation
    party_one_message.add_alpha_one_times_mult_share_one(
        p1_return.first.alpha_one_times_mult_share_one());
    states.push_back(p1_return.second);
  }
  return std::make_pair(party_one_message, states);
}

StatusOr<FixedPointElement> SecureExponentiationPartyOne::OutputResult(
    const ExponentiationPartyZeroMultToAddMessage& party_zero_msg,
    const State& self_state) {
  uint64_t result_from_mult = ModMul(
      ModMul(self_state.mult_share_one,
             self_state.beta_one,
             exp_params_->prime_q),
      party_zero_msg.beta_zero_times_mult_share_zero(),
      exp_params_->prime_q);

  // GenerateMultToAddMessage has an additional fractional_multiplier factor.
  // Furthermore, it added the base two exponent bound to the exponent to
  // ensure that it was >= 1.
  // Divide by these to get the final share in Z_Q.
  // For P_1, this is also the additive share in Z_{primary_ring_modulus}.
  uint64_t final_additive_share = result_from_mult
      / (fpe_params_->fractional_multiplier * two_power_base2_bound_);

  // Create the FixedPointElement.
  ASSIGN_OR_RETURN(auto fpe_share,
                   fpe_factory_->ImportFixedPointElementFromUint64(
                       final_additive_share));
  return fpe_share;
}

StatusOr<std::vector<FixedPointElement>>
SecureExponentiationPartyOne::BatchedOutputResult(
    const BatchedExponentiationPartyZeroMultToAddMessage& party_zero_msg,
    const std::vector<State>& self_state) {
  size_t length = self_state.size();
  ASSIGN_OR_RETURN(FixedPointElement zero,
                   fpe_factory_->CreateFixedPointElementFromInt(0));
  std::vector<FixedPointElement> fpe_shares(length, zero);
  for (size_t idx = 0; idx < length; idx++) {
    // Retrieve the exponentiation message for a single party
    ExponentiationPartyZeroMultToAddMessage party_zero_single_message;
    party_zero_single_message.set_beta_zero_times_mult_share_zero(
        party_zero_msg.beta_zero_times_mult_share_zero(idx));

    ASSIGN_OR_RETURN(fpe_shares[idx],
                     OutputResult(party_zero_single_message, self_state[idx]));
  }
  return fpe_shares;
}

}  // namespace private_join_and_compute
