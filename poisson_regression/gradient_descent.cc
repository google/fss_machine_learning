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

#include "poisson_regression/gradient_descent.h"

#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/gradient_descent_messages.pb.h"
#include "poisson_regression/gradient_descent_utils.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "poisson_regression/secure_exponentiation.h"
#include "private_join_and_compute/util/status.inc"
#include "absl/memory/memory.h"

namespace private_join_and_compute {
namespace poisson_regression {

// Init the gradient descent with inputs and parameters.
StatusOr<GradientDescentPartyZero> GradientDescentPartyZero::Init(
    std::vector<uint64_t> share_x,
    std::vector<uint64_t> share_y,
    std::vector<uint64_t> share_delta,
    std::vector<uint64_t> share_theta,
    std::unique_ptr<ShareProvider> share_provider,
    const FixedPointElementFactory::Params& fpe_params,
    const ExponentiationParams& exp_params,
    const GradientDescentParams& param) {
  // Verify that shapes of the input vectors and matrices agree with the
  // provided parameters.
  if (share_x.size() != param.num_features * param.feature_length ||
      share_y.size() != param.num_features ||
      share_delta.size() != param.num_features ||
      share_theta.size() != param.feature_length) {
    return InvalidArgumentError("Gradient descent init: invalid input size.");
  }
  // Initialize the factory for fixed point elements.
  ASSIGN_OR_RETURN(auto temp,
                   FixedPointElementFactory::Create(param.num_fractional_bits,
                                                    param.num_ring_bits));
  auto fp_factory =
      absl::make_unique<FixedPointElementFactory>(std::move(temp));

  ASSIGN_OR_RETURN(auto secure_exp,
                   SecureExponentiationPartyZero::Create(fpe_params,
                                                         exp_params));
  // Initialize GradientDescentPartyZero.
  return GradientDescentPartyZero(share_x, share_y, share_delta, share_theta,
                                  std::move(share_provider),
                                  std::move(fp_factory), std::move(secure_exp),
                                  param);
}

StatusOr<std::pair<StateRound1, GradientDescentRoundOneMessage>>
GradientDescentPartyZero::GenerateGradientDescentRoundOneMessage() {
  // Get Beaver triple matrix ([A], [B], [C]) for round one.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix,
                   share_provider_->GetRoundOneBeaverMatrix());

  // Generate the state and message ([X-A], [Y-B]) to send to the other party.
  ASSIGN_OR_RETURN(
      auto matrix_mult_state_and_message,
      GenerateMatrixMultiplicationGateMessage(
          share_x_, share_theta_, beaver_triple_matrix, param_.modulus));
  auto state = matrix_mult_state_and_message.first;
  auto matrix_mult_message = matrix_mult_state_and_message.second;
  // Copy message.
  GradientDescentRoundOneMessage round_one_message;
  *round_one_message.mutable_matrix_mult_message() =
      std::move(matrix_mult_message);
  StateRound1 state_one = {
      .share_x_minus_a = std::move(state.share_x_minus_a),
      .share_theta_minus_b = std::move(state.share_y_minus_b)};
  return std::make_pair(state_one, round_one_message);
}

StatusOr<std::pair<StateRound2, GradientDescentPartyZeroRoundTwoMessage>>
GradientDescentPartyZero::GenerateGradientDescentRoundTwoMessage(
    StateRound1 state_one, GradientDescentRoundOneMessage other_party_message) {
  // Get Beaver triple matrix ([A], [B], [C]) used in round one.
  ASSIGN_OR_RETURN(BeaverTripleMatrix<uint64_t> beaver_triple_matrix,
                   share_provider_->GetRoundOneBeaverMatrix());
  // Convert StateRound1 to the State struct.
  MatrixMultState state = {
    .share_x_minus_a = std::move(state_one.share_x_minus_a),
    .share_y_minus_b = std::move(state_one.share_theta_minus_b)
  };
  // Compute [U] = [X*Theta]: * represents matrix multiplication mod modulus.
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_u,
                   GenerateMatrixMultiplicationOutputPartyZero(
                       state, beaver_triple_matrix,
                       other_party_message.matrix_mult_message(),
                       param_.modulus));
  // Get MultToAdd shares that are used to compute the message in round two.
  ASSIGN_OR_RETURN(auto mult_to_add_shares,
                   share_provider_->GetRoundTwoMultToAdd());
  // Generate messages that will be used to compute exp([u_i]).
  // The messages will be processed in batches.
  std::vector<uint64_t> mult_shares(share_u.size());
  GradientDescentPartyZeroRoundTwoMessage round_two_message;
  for (size_t idx = 0; idx < share_u.size(); idx++) {
    auto alpha_zero = mult_to_add_shares.first[idx];
    auto beta_zero = mult_to_add_shares.second[idx];
    // Store the share as fixed point element.
    ASSIGN_OR_RETURN(
        FixedPointElement fpe_share_zero,
        fp_factory_->ImportFixedPointElementFromUint64(share_u[idx]));
    // Generate the state and message in round two. The message will be sent
    // to the other party.
    ASSIGN_OR_RETURN(auto exp_state_and_message,
                     secure_exp_->GenerateMultToAddMessage(
                         fpe_share_zero, alpha_zero, beta_zero));
    mult_shares[idx] = exp_state_and_message.second.mult_share_zero;
    *round_two_message.add_exp_message() =
        std::move(exp_state_and_message.first);
  }
  StateRound2 state_two = {.share_mult = std::move(mult_shares)};
  return std::make_pair(std::move(state_two), std::move(round_two_message));
}

StatusOr<std::pair<StateRound3, GradientDescentRoundThreeMessage>>
GradientDescentPartyZero::GenerateGradientDescentRoundThreeMessage(
    StateRound2 state_two,
    GradientDescentPartyOneRoundTwoMessage other_party_message) {
  size_t length = state_two.share_mult.size();
  // Get MultToAdd shares that are used in round two.
  ASSIGN_OR_RETURN(MultToAddShare mult_to_add_shares,
                   share_provider_->GetRoundTwoMultToAdd());
  // Compute [v_i] = exp([u_i]).
  std::vector<uint64_t> share_v(length);
  for (size_t idx = 0; idx < length; idx++) {
    // Convert StateRound2 to the State struct used in secure exponentiation.
    SecureExponentiationPartyZero::State state = {
        .mult_share_zero = state_two.share_mult[idx],
        .alpha_zero = mult_to_add_shares.first[idx],
        .beta_zero = mult_to_add_shares.second[idx]};
    ExponentiationPartyOneMultToAddMessage exp_message =
        other_party_message.exp_message().Get(idx);
    // Output exp([u_i) as fixed point. The output is exported to unit64.
    ASSIGN_OR_RETURN(auto fpe_exp,
                     secure_exp_->OutputResult(exp_message, state));
    share_v[idx] = fpe_exp.ExportToUint64();
  }
  // Get Beaver triple vector ([A], [B], [C]) for round three.
  ASSIGN_OR_RETURN(BeaverTripleVector<uint64_t> beaver_triple_vector,
                   share_provider_->GetRoundThreeBeaverVector());
  // Generate the next state and message from exp([u_i]) and [delta_t_i].
  ASSIGN_OR_RETURN(
      auto batched_mult_state_and_message,
      GenerateBatchedMultiplicationGateMessage(
          share_delta_t_, share_v, beaver_triple_vector, param_.modulus));
  auto state = batched_mult_state_and_message.first;
  auto batched_mult_message = batched_mult_state_and_message.second;
  GradientDescentRoundThreeMessage round_three_message;
  *round_three_message.mutable_batched_mult_message() =
      std::move(batched_mult_message);
  StateRound3 state_three = {
      .share_delta_t_minus_a = std::move(state.share_x_minus_a),
      .share_v_minus_b = std::move(state.share_y_minus_b)};
  return std::make_pair(std::move(state_three), std::move(round_three_message));
}

StatusOr<std::pair<StateRound4, GradientDescentRoundFourMessage>>
GradientDescentPartyZero::GenerateGradientDescentRoundFourMessage(
    StateRound3 state_three,
    GradientDescentRoundThreeMessage other_party_message) {
  // Convert StateRound3 to State struct used in BatchedMultiplication.
  BatchedMultState state = {
      .share_x_minus_a = std::move(state_three.share_delta_t_minus_a),
      .share_y_minus_b = std::move(state_three.share_v_minus_b)
  };
  // Get Beaver triple vector ([A], [B], [C]) used in round three.
  ASSIGN_OR_RETURN(BeaverTripleVector<uint64_t> beaver_triple_vector,
                   share_provider_->GetRoundThreeBeaverVector());
  // Compute [w_i] = [delta_t_i*v_i] where * represents regular multiplication
  // mod modulus.
  ASSIGN_OR_RETURN(
      auto share_delta_v,
      GenerateBatchedMultiplicationOutputPartyZero(
          state, beaver_triple_vector,
          other_party_message.batched_mult_message(), param_.modulus));
  // Compute [S] = [Y - W].
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_s,
                   BatchedModSub(share_y_, share_delta_v, param_.modulus));
  // Generate state and message to compute [Z] = [S*X]: * represents matrix
  // multiplication mod modulus.
  ASSIGN_OR_RETURN(BeaverTripleMatrix<uint64_t> beaver_triple_matrix,
                   share_provider_->GetRoundFourBeaverMatrix());
  ASSIGN_OR_RETURN(
      auto matrix_mult_state_and_message,
      GenerateMatrixMultiplicationGateMessage(
          share_s, share_x_, beaver_triple_matrix, param_.modulus));
  StateRound4 state_four = {
      .share_s_minus_a =
          std::move(matrix_mult_state_and_message.first.share_x_minus_a),
      .share_x_minus_b =
          std::move(matrix_mult_state_and_message.first.share_y_minus_b)};
  auto matrix_mult_message = std::move(matrix_mult_state_and_message.second);
  GradientDescentRoundFourMessage round_four_message;
  *round_four_message.mutable_matrix_mult_message() =
      std::move(matrix_mult_message);
  return std::make_pair(std::move(state_four), round_four_message);
}

Status GradientDescentPartyZero::ComputeGradientUpdate(
    StateRound4 state_four,
    GradientDescentRoundFourMessage other_party_message) {
  // Compute [Z] = [S*X].
  ASSIGN_OR_RETURN(BeaverTripleMatrix<uint64_t> beaver_triple_matrix,
                   share_provider_->GetRoundFourBeaverMatrix());
  MatrixMultState state = {
    .share_x_minus_a = std::move(state_four.share_s_minus_a),
    .share_y_minus_b = std::move(state_four.share_x_minus_b)
  };
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_z,
                   GenerateMatrixMultiplicationOutputPartyZero(
                       state, beaver_triple_matrix,
                       other_party_message.matrix_mult_message(),
                       param_.modulus));
  // Compute update for Theta.
  for (size_t idx = 0; idx < share_theta_.size(); idx++) {
    share_theta_[idx] = ModAdd(
        ModMul(param_.one_minus_beta, share_theta_[idx], param_.modulus),
        ModMul(param_.alpha, share_z[idx], param_.modulus), param_.modulus);
  }
  // The multiplication in the previous step returns 2^{2l_f}*Theta.
  // The truncation is needed to bring it back to the form 2^{l_f}*Theta.
  ASSIGN_OR_RETURN(
      share_theta_,
      TruncateSharePartyZero(share_theta_, (1ULL << param_.num_fractional_bits),
                             param_.modulus));
  // Throw away the consumed preprocessed shares.
  return share_provider_->Shrink();
}

// Init the gradient descent with inputs and parameters.
StatusOr<GradientDescentPartyOne> GradientDescentPartyOne::Init(
    std::vector<uint64_t> share_x,
    std::vector<uint64_t> share_y,
    std::vector<uint64_t> share_delta, std::vector<uint64_t> share_theta,
    std::unique_ptr<ShareProvider> share_provider,
    const FixedPointElementFactory::Params& fpe_params,
    const ExponentiationParams& exp_params,
    const GradientDescentParams& param) {
  // Verify that shapes of the input vectors and matrices agree with the
  // provided parameters.
  if (share_x.size() != param.num_features * param.feature_length ||
      share_y.size() != param.num_features ||
      share_delta.size() != param.num_features ||
      share_theta.size() != param.feature_length) {
    return InvalidArgumentError("Gradient descent: invalid input size.");
  }
  // Initialize the factory for fixed point elements.
  ASSIGN_OR_RETURN(auto temp,
                   FixedPointElementFactory::Create(param.num_fractional_bits,
                                                    param.num_ring_bits));
  auto fp_factory =
      absl::make_unique<FixedPointElementFactory>(std::move(temp));

  ASSIGN_OR_RETURN(auto secure_exp,
                   SecureExponentiationPartyOne::Create(fpe_params,
                                                        exp_params));
  // Initialize GradientDescentPartyZero.
  return GradientDescentPartyOne(share_x, share_y, share_delta, share_theta,
                                 std::move(share_provider),
                                 std::move(fp_factory), std::move(secure_exp),
                                 param);
}

StatusOr<std::pair<StateRound1, GradientDescentRoundOneMessage>>
GradientDescentPartyOne::GenerateGradientDescentRoundOneMessage() {
  // Get Beaver triple matrix ([A], [B], [C]) for round one.
  ASSIGN_OR_RETURN(BeaverTripleMatrix<uint64_t> beaver_triple_matrix,
                   share_provider_->GetRoundOneBeaverMatrix());

  // Generate the state and message ([X-A], [Y-B]) to send to the other party.
  ASSIGN_OR_RETURN(
      auto matrix_mult_state_and_message,
      GenerateMatrixMultiplicationGateMessage(
          share_x_, share_theta_, beaver_triple_matrix, param_.modulus));
  MatrixMultState state = matrix_mult_state_and_message.first;
  auto matrix_mult_message = matrix_mult_state_and_message.second;
  GradientDescentRoundOneMessage round_one_message;
  *round_one_message.mutable_matrix_mult_message() =
      std::move(matrix_mult_message);
  StateRound1 state_one = {
      .share_x_minus_a = std::move(state.share_x_minus_a),
      .share_theta_minus_b = std::move(state.share_y_minus_b)};
  return std::make_pair(state_one, round_one_message);
}

StatusOr<std::pair<StateRound2, GradientDescentPartyOneRoundTwoMessage>>
GradientDescentPartyOne::GenerateGradientDescentRoundTwoMessage(
    StateRound1 state_one, GradientDescentRoundOneMessage other_party_message) {
  // Get Beaver triple matrix ([A], [B], [C]) used in round one.
  ASSIGN_OR_RETURN(BeaverTripleMatrix<uint64_t> beaver_triple_matrix,
                   share_provider_->GetRoundOneBeaverMatrix());
  // Convert StateRound1 to the State struct.
  MatrixMultState state = {
    .share_x_minus_a = std::move(state_one.share_x_minus_a),
    .share_y_minus_b = std::move(state_one.share_theta_minus_b)
  };
  // Compute [U] = [X*Theta]: * represents matrix multiplication mod modulus.
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_u,
                   GenerateMatrixMultiplicationOutputPartyOne(
                       state, beaver_triple_matrix,
                       other_party_message.matrix_mult_message(),
                       param_.modulus));
  // Get MultToAdd shares that are used to compute the message in round two.
  ASSIGN_OR_RETURN(MultToAddShare mult_to_add_shares,
                   share_provider_->GetRoundTwoMultToAdd());
  // Generate messages that will be used to compute exp([u_i]).
  // The messages will be processed in batches.
  std::vector<uint64_t> mult_shares(share_u.size());
  GradientDescentPartyOneRoundTwoMessage round_two_message;
  for (size_t idx = 0; idx < share_u.size(); idx++) {
    auto alpha_one = mult_to_add_shares.first[idx];
    auto beta_one = mult_to_add_shares.second[idx];
    // Store the share as fixed point element.
    ASSIGN_OR_RETURN(
        FixedPointElement fpe_share_one,
        fp_factory_->ImportFixedPointElementFromUint64(share_u[idx]));
    // Generate the state and message in round two. The message will be sent
    // to the other party.
    ASSIGN_OR_RETURN(auto exp_state_and_message,
                     secure_exp_->GenerateMultToAddMessage(
                         fpe_share_one, alpha_one, beta_one));
    mult_shares[idx] = exp_state_and_message.second.mult_share_one;
    *round_two_message.add_exp_message() =
        std::move(exp_state_and_message.first);
  }
  StateRound2 state_two = {.share_mult = std::move(mult_shares)};
  return std::make_pair(std::move(state_two), std::move(round_two_message));
}

StatusOr<std::pair<StateRound3, GradientDescentRoundThreeMessage>>
GradientDescentPartyOne::GenerateGradientDescentRoundThreeMessage(
    StateRound2 state_two,
    GradientDescentPartyZeroRoundTwoMessage other_party_message) {
  size_t length = state_two.share_mult.size();
  // Get MultToAdd shares that are used in round two.
  ASSIGN_OR_RETURN(MultToAddShare mult_to_add_shares,
                   share_provider_->GetRoundTwoMultToAdd());
  // Compute [v_i] = exp([u_i]).
  std::vector<uint64_t> share_v(length);
  for (size_t idx = 0; idx < length; idx++) {
    // Convert StateRound2 to the State struct used in secure exponentiation.
    SecureExponentiationPartyOne::State state = {
        .mult_share_one = state_two.share_mult[idx],
        .alpha_one = mult_to_add_shares.first[idx],
        .beta_one = mult_to_add_shares.second[idx]};
    ExponentiationPartyZeroMultToAddMessage exp_message =
        other_party_message.exp_message().Get(idx);
    // Output exp([u_i) as fixed point. The output is exported to unit64.
    ASSIGN_OR_RETURN(auto fpe_exp,
                     secure_exp_->OutputResult(exp_message, state));
    share_v[idx] = fpe_exp.ExportToUint64();
  }
  // Get Beaver triple vector ([A], [B], [C]) for round three.
  ASSIGN_OR_RETURN(BeaverTripleVector<uint64_t> beaver_triple_vector,
                   share_provider_->GetRoundThreeBeaverVector());
  // Generate the next state and message from exp([u_i]) and [delta_t_i].
  ASSIGN_OR_RETURN(
      auto batched_mult_state_and_message,
      GenerateBatchedMultiplicationGateMessage(
          share_delta_t_, share_v, beaver_triple_vector, param_.modulus));
  BatchedMultState state = batched_mult_state_and_message.first;
  MultiplicationGateMessage batched_mult_message =
      batched_mult_state_and_message.second;
  GradientDescentRoundThreeMessage round_three_message;
  *round_three_message.mutable_batched_mult_message() =
      std::move(batched_mult_message);
  StateRound3 state_three = {
      .share_delta_t_minus_a = std::move(state.share_x_minus_a),
      .share_v_minus_b = std::move(state.share_y_minus_b)};
  return std::make_pair(std::move(state_three), std::move(round_three_message));
}

StatusOr<std::pair<StateRound4, GradientDescentRoundFourMessage>>
GradientDescentPartyOne::GenerateGradientDescentRoundFourMessage(
    StateRound3 state_three,
    GradientDescentRoundThreeMessage other_party_message) {
  // Convert StateRound3 to State struct used in BatchedMultiplication.
  BatchedMultState state = {
      .share_x_minus_a = std::move(state_three.share_delta_t_minus_a),
      .share_y_minus_b = std::move(state_three.share_v_minus_b)};
  // Get Beaver triple vector ([A], [B], [C]) used in round three.
  ASSIGN_OR_RETURN(BeaverTripleVector<uint64_t> beaver_triple_vector,
                   share_provider_->GetRoundThreeBeaverVector());
  // Compute [w_i] = [delta_t_i*v_i] where * represents regular multiplication
  // mod modulus.
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_delta_v,
      GenerateBatchedMultiplicationOutputPartyOne(
          state, beaver_triple_vector,
          other_party_message.batched_mult_message(), param_.modulus));
  // Compute [S] = [Y - W].
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_s,
                   BatchedModSub(share_y_, share_delta_v, param_.modulus));
  // Generate state and message to compute [Z] = [S*X]: * represents matrix
  // multiplication mod modulus.
  ASSIGN_OR_RETURN(BeaverTripleMatrix<uint64_t> beaver_triple_matrix,
                   share_provider_->GetRoundFourBeaverMatrix());
  ASSIGN_OR_RETURN(
      auto matrix_mult_state_and_message,
      GenerateMatrixMultiplicationGateMessage(
          share_s, share_x_, beaver_triple_matrix, param_.modulus));
  StateRound4 state_four = {
      .share_s_minus_a =
          std::move(matrix_mult_state_and_message.first.share_x_minus_a),
      .share_x_minus_b =
          std::move(matrix_mult_state_and_message.first.share_y_minus_b)};
  auto matrix_mult_message = std::move(matrix_mult_state_and_message.second);
  GradientDescentRoundFourMessage round_four_message;
  *round_four_message.mutable_matrix_mult_message() =
      std::move(matrix_mult_message);
  return std::make_pair(std::move(state_four), round_four_message);
}

Status GradientDescentPartyOne::ComputeGradientUpdate(
    StateRound4 state_four,
    GradientDescentRoundFourMessage other_party_message) {
  // Compute [Z] = [S*X].
  ASSIGN_OR_RETURN(BeaverTripleMatrix<uint64_t> beaver_triple_matrix,
                   share_provider_->GetRoundFourBeaverMatrix());
  MatrixMultState state = {
    .share_x_minus_a = std::move(state_four.share_s_minus_a),
    .share_y_minus_b = std::move(state_four.share_x_minus_b)
  };
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_z,
                   GenerateMatrixMultiplicationOutputPartyOne(
                       state, beaver_triple_matrix,
                       other_party_message.matrix_mult_message(),
                       param_.modulus));
  // Compute update for Theta.
  for (size_t idx = 0; idx < share_theta_.size(); idx++) {
    share_theta_[idx] = ModAdd(
        ModMul(param_.one_minus_beta, share_theta_[idx], param_.modulus),
        ModMul(param_.alpha, share_z[idx], param_.modulus), param_.modulus);
  }
  // The multiplication in the previous step returns 2^{2l_f}*Theta.
  // The truncation is needed to bring it back to the form 2^{l_f}*Theta.
  ASSIGN_OR_RETURN(
      share_theta_,
      TruncateSharePartyOne(share_theta_,
                            (1ULL << param_.num_fractional_bits),
                            param_.modulus));
  // Throw away the consumed preprocessed shares.
  return share_provider_->Shrink();
}

}  // namespace poisson_regression
}  // namespace private_join_and_compute
