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

#include "secret_sharing_mpc/gates/vector_exponentiation.h"

namespace private_join_and_compute {

StatusOr<std::pair<MultToAddShare, MultToAddShare>> SampleMultToAddSharesVector(
    size_t length, uint64_t modulus) {
  return internal::SampleMultToAddSharesWithPrng(length, modulus);
}

StatusOr<std::pair<BatchedExponentiationPartyZeroMultToAddMessage,
                   std::vector<private_join_and_compute::SecureExponentiationPartyZero::State>>>
GenerateVectorMultToAddMessagePartyZero(
    std::unique_ptr<SecureExponentiationPartyZero>& party_zero_,
    const std::vector<FixedPointElement>& fpe_shares_zero,
    const std::vector<uint64_t>& alpha_zero,
    const std::vector<uint64_t>& beta_zero) {
  // Run first part of computation to get messages and state.
  ASSIGN_OR_RETURN(auto p0_return, party_zero_->GenerateBatchedMultToAddMessage(
                                       fpe_shares_zero, alpha_zero, beta_zero));
  return p0_return;
}

StatusOr<std::pair<BatchedExponentiationPartyOneMultToAddMessage,
                   std::vector<private_join_and_compute::SecureExponentiationPartyOne::State>>>
GenerateVectorMultToAddMessagePartyOne(
    std::unique_ptr<SecureExponentiationPartyOne>& party_one_,
    const std::vector<FixedPointElement>& fpe_shares_one,
    const std::vector<uint64_t>& alpha_one,
    const std::vector<uint64_t>& beta_one) {
  // Run first part of computation to get messages and state.
  ASSIGN_OR_RETURN(auto p1_return, party_one_->GenerateBatchedMultToAddMessage(
                                       fpe_shares_one, alpha_one, beta_one));
  return p1_return;
}

StatusOr<std::vector<FixedPointElement>> VectorExponentiationPartyZero(
    std::unique_ptr<SecureExponentiationPartyZero>& party_zero_,
    const BatchedExponentiationPartyOneMultToAddMessage& party_one_msg,
    const std::vector<private_join_and_compute::SecureExponentiationPartyZero::State>& state) {
  // Run second part of computation to get exponentiation output.
  ASSIGN_OR_RETURN(auto p0_return,
                   party_zero_->BatchedOutputResult(party_one_msg, state));
  return p0_return;
}

StatusOr<std::vector<FixedPointElement>> VectorExponentiationPartyOne(
    std::unique_ptr<SecureExponentiationPartyOne>& party_one_,
    const BatchedExponentiationPartyZeroMultToAddMessage& party_zero_msg,
    const std::vector<private_join_and_compute::SecureExponentiationPartyOne::State>& state) {
  // Run second part of computation to get exponentiation output.
  ASSIGN_OR_RETURN(auto p1_return,
                   party_one_->BatchedOutputResult(party_zero_msg, state));
  return p1_return;
}

}  // namespace private_join_and_compute
