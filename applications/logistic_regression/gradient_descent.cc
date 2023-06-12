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

#include "applications/logistic_regression/gradient_descent.h"

#include <cmath>

#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "poisson_regression/secure_exponentiation.h"
#include "private_join_and_compute/util/status.inc"
#include "secret_sharing_mpc/arithmetic/matrix_arithmetic.h"
#include "secret_sharing_mpc/gates/correlated_matrix_product.h"
#include "secret_sharing_mpc/gates/scalar_vector_product.h"
#include "secret_sharing_mpc/gates/vector_addition.h"
#include "secret_sharing_mpc/gates/vector_subtraction.h"

#include "absl/memory/memory.h"

namespace private_join_and_compute {
namespace logistic_regression {

// Init the gradient descent with inputs and parameters.
StatusOr<GradientDescentPartyZero> GradientDescentPartyZero::Init(
    std::vector<uint64_t> share_x,
    std::vector<uint64_t> share_y,
    std::vector<uint64_t> share_theta,
    std::unique_ptr<LogRegShareProvider> share_provider,
    const FixedPointElementFactory::Params& fpe_params,
    const GradientDescentParams& param) {
  // Verify that shapes of the input vectors and matrices agree with the
  // provided parameters.
  if (share_x.size() != param.num_examples * param.num_features ||
      share_y.size() != param.num_examples ||
      share_theta.size() != param.num_features) {
    return InvalidArgumentError("Gradient descent init: invalid input size.");
  }
  // Initialize the factory for fixed point elements.
  ASSIGN_OR_RETURN(auto temp,
                   FixedPointElementFactory::Create(param.num_fractional_bits,
                                                    param.num_ring_bits));
  auto fp_factory =
      absl::make_unique<FixedPointElementFactory>(std::move(temp));

  // Transpose share_x (Computing on a transpose will also be needed)
  // (num_example, num_features) -> (num_features, num_examples)
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_x_transpose,
                   Transpose(share_x, param.num_examples, param.num_features));

  // Initialize GradientDescentPartyZero.
  return GradientDescentPartyZero(share_x, share_x_transpose, share_y,
                                  share_theta, std::move(share_provider),
                                  std::move(fp_factory),
                                  param);
}

StatusOr<std::pair<StateMaskedX, MaskedXMessage>>
GradientDescentPartyZero::GenerateCorrelatedProductMessageForX() {
  // Get Beaver triple matrix ([A] only) for round one.
  ASSIGN_OR_RETURN(std::vector<uint64_t> beaver_triple_matrix_a,
                   share_provider_->GetRoundOneBeaverMatrixA());

  // Generate the state and message [X-A] to send to other party.
  ASSIGN_OR_RETURN(auto matrix_mult_state_and_message,
                   GenerateMatrixXminusAProductMessage(
                       share_x_, beaver_triple_matrix_a, param_.modulus));
  std::vector<uint64_t> share_x_minus_a = matrix_mult_state_and_message.first;
  MatrixXminusAProductMessage x_minus_a_message =
      matrix_mult_state_and_message.second;
  // Copy message.
  MaskedXMessage masked_x_message;
  *masked_x_message.mutable_matrix_x_minus_matrix_a_message() =
      std::move(x_minus_a_message);
  StateMaskedX state_x_minus_a = {.share_x_minus_a =
                                      std::move(share_x_minus_a)};
  return std::make_pair(state_x_minus_a, masked_x_message);
}

StatusOr<std::pair<StateMaskedXTranspose, MaskedXTransposeMessage>>
GradientDescentPartyZero::GenerateCorrelatedProductMessageForXTranspose() {
  // Set up to compute X.transpose() * d

  // Generate state and message to compute [g] = [X.transpose * d]:
  // * represents matrix product mod modulus.

  // Get Beaver triple matrix ([A], [B], [C]) for round three.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a,
                   share_provider_->GetRoundThreeBeaverMatrixA());

  // Generate the state and message [X^T-A] to send to other party.
  ASSIGN_OR_RETURN(
      auto matrix_mult_state_and_message,
      GenerateMatrixXminusAProductMessage(
          share_x_transpose_, beaver_triple_matrix_a, param_.modulus));
  auto share_x_transpose_minus_a = matrix_mult_state_and_message.first;
  auto x_transpose_minus_a_message = matrix_mult_state_and_message.second;
  // Copy message.
  MaskedXTransposeMessage masked_x_transpose_message;
  *masked_x_transpose_message
       .mutable_matrix_x_transpose_minus_matrix_a_message() =
      std::move(x_transpose_minus_a_message);
  StateMaskedXTranspose state_x_transpose_minus_a = {
      .share_x_transpose_minus_a = std::move(share_x_transpose_minus_a)};
  return std::make_pair(state_x_transpose_minus_a, masked_x_transpose_message);
}

StatusOr<std::pair<StateRound1, RoundOneGradientDescentMessage>>
GradientDescentPartyZero::GenerateGradientDescentRoundOneMessage(
    StateMaskedX share_x_minus_a) {
  // Get Beaver triple matrix ([A], [B], [C]) for round one.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
                   share_provider_->GetRoundOneBeaverMatrixBandC());

  // Generate the state and message ([X-A], [Theta-B]) to send to other party.
  ASSIGN_OR_RETURN(auto matrix_mult_state_and_message,
                   GenerateMatrixYminusBProductMessage(
                       share_theta_, share_x_minus_a.share_x_minus_a,
                       beaver_triple_matrix_b_c.first, param_.modulus));
  auto state = matrix_mult_state_and_message.first;
  auto matrix_theta_minus_b_message = matrix_mult_state_and_message.second;
  // Copy message.
  RoundOneGradientDescentMessage round_one_message;
  *round_one_message.mutable_matrix_theta_minus_matrix_b_message() =
      std::move(matrix_theta_minus_b_message);
  StateRound1 state_one = {
      .share_x_minus_a = std::move(state.share_x_minus_a),
      .share_theta_minus_b = std::move(state.share_y_minus_b)};
  return std::make_pair(state_one, round_one_message);
}

StatusOr<SigmoidInput> GradientDescentPartyZero::GenerateSigmoidInput(
    StateRound1 state_one, MaskedXMessage other_party_x_minus_a_message,
    RoundOneGradientDescentMessage other_party_theta_minus_b_message) {
  // Get Beaver triple matrix ([A], [B], [C]) used in round one.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a,
                   share_provider_->GetRoundOneBeaverMatrixA());
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
                   share_provider_->GetRoundOneBeaverMatrixBandC());
  // Convert StateRound1 to the MatrixMultState struct.
  MatrixMultState state = {
    .share_x_minus_a = std::move(state_one.share_x_minus_a),
    .share_y_minus_b = std::move(state_one.share_theta_minus_b)
  };
  // Compute [U] = [X*Theta]: * represents matrix product mod modulus.
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_u,
      CorrelatedMatrixProductPartyZero(
          state, beaver_triple_matrix_a, beaver_triple_matrix_b_c.first,
          beaver_triple_matrix_b_c.second,
          other_party_x_minus_a_message.matrix_x_minus_matrix_a_message(),
          other_party_theta_minus_b_message
              .matrix_theta_minus_matrix_b_message(),
          param_.num_examples, param_.num_features, 1,
          param_.num_fractional_bits, param_.modulus));
  SigmoidInput sigmoid_input = {.sigmoid_input = std::move(share_u)};
  return sigmoid_input;
}

StatusOr<SigmoidInput> GradientDescentPartyZero::GenerateSigmoidInputMinibatch(
    StateRound1 state_one, MaskedXMessage other_party_x_minus_a_message,
    RoundOneGradientDescentMessage other_party_theta_minus_b_message,
    size_t batch_size, size_t idx_batch, size_t size_per_minibatch) {
  // Get Beaver triple matrix ([A], [B], [C]) used in round one.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a_full,
                   share_provider_->GetRoundOneBeaverMatrixA());
  // Retrieve the correct batch from this beaver matrix a
  size_t start_idx = idx_batch * size_per_minibatch;
  std::vector<uint64_t> beaver_triple_matrix_a (size_per_minibatch);
  for (size_t idx = 0; idx < size_per_minibatch; idx++) {
    beaver_triple_matrix_a[idx] = beaver_triple_matrix_a_full[start_idx + idx];
  }

  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
                   share_provider_->GetRoundOneBeaverMatrixBandC());
  // Convert StateRound1 to the MatrixMultState struct.
  MatrixMultState state = {
      .share_x_minus_a = std::move(state_one.share_x_minus_a),
      .share_y_minus_b = std::move(state_one.share_theta_minus_b)
  };
  // Compute [U] = [X*Theta]: * represents matrix product mod modulus.
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_u,
      CorrelatedMatrixProductPartyZero(
          state, beaver_triple_matrix_a, beaver_triple_matrix_b_c.first,
          beaver_triple_matrix_b_c.second,
          other_party_x_minus_a_message.matrix_x_minus_matrix_a_message(),
          other_party_theta_minus_b_message
              .matrix_theta_minus_matrix_b_message(),
          batch_size, param_.num_features, 1,
          param_.num_fractional_bits, param_.modulus));
  SigmoidInput sigmoid_input = {.sigmoid_input = std::move(share_u)};
  return sigmoid_input;
}

StatusOr<std::pair<SigmoidOutput, SigmoidOutput>>
GradientDescentPartyZero::GenerateSigmoidOutputForTesting(
    SigmoidInput sigmoid_input_share_p0, SigmoidInput sigmoid_input_share_p1) {
  size_t batch_length = param_.num_examples;

  // Reconstruct the sigmoid inputs
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> reconstructed_input,
      BatchedModAdd(sigmoid_input_share_p0.sigmoid_input,
                    sigmoid_input_share_p1.sigmoid_input, param_.modulus));
  // Compute sigmoid in plaintext
  std::vector<uint64_t> sigmoid_output(batch_length, 0);
  for (size_t idx = 0; idx < batch_length; idx++) {
    // Compute sigmoid on double for simplicity
    double reconstructed_double_input =
        fp_factory_->ImportFixedPointElementFromUint64(reconstructed_input[idx])
            ->ExportToDouble();
    double double_output = 1. / (1. + exp(-reconstructed_double_input));
    ASSIGN_OR_RETURN(
        FixedPointElement fpe_output,
        fp_factory_->CreateFixedPointElementFromDouble(double_output));
    sigmoid_output[idx] = fpe_output.ExportToUint64();
  }
  // Secret share the computed sigmoid
  ASSIGN_OR_RETURN(auto share_zero, internal::SampleShareOfZero(
                                            batch_length, param_.modulus));
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> sigmoid_output_p0_share,
      BatchedModAdd(share_zero.first, sigmoid_output, param_.modulus));
  std::vector<uint64_t> sigmoid_output_p1_share = std::move(share_zero.second);

  // Convert to SigmoidOutput
  SigmoidOutput sigmoid_output_p0_struct = {
    .sigmoid_output = std::move(sigmoid_output_p0_share)
  };
  SigmoidOutput sigmoid_output_p1_struct = {
    .sigmoid_output = std::move(sigmoid_output_p1_share)
  };

  return std::make_pair(
      std::move(sigmoid_output_p0_struct), std::move(sigmoid_output_p1_struct));
}

StatusOr<std::pair<StateRound3, RoundThreeGradientDescentMessage>>
GradientDescentPartyZero::GenerateGradientDescentRoundThreeMessageMinibatch(
    SigmoidOutput sigmoid_output_share,
    StateMaskedXTranspose share_x_transpose_minus_a,
    size_t batch_size, size_t idx_batch) {
  // Retrieve the correct batch from this share_y
  size_t start_idx = idx_batch * batch_size;
  std::vector<uint64_t> share_y_minibatch (batch_size);
  for (size_t idx = 0; idx < batch_size; idx++) {
    share_y_minibatch[idx] = share_y_[start_idx + idx];
  }

  // Compute d = s - y
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_d,
                   VectorSubtract(sigmoid_output_share.sigmoid_output,
                                  share_y_minibatch, param_.modulus));
  // Set up to compute X.transpose() * d

  // Generate state and message to compute [g] = [X.transpose * d]:
  // * represents matrix product mod modulus.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
                   share_provider_->GetRoundThreeBeaverMatrixBandC());
  ASSIGN_OR_RETURN(
      auto matrix_mult_state_and_message,
      GenerateMatrixYminusBProductMessage(
          share_d, share_x_transpose_minus_a.share_x_transpose_minus_a,
          beaver_triple_matrix_b_c.first, param_.modulus));
  auto state = matrix_mult_state_and_message.first;
  auto matrix_d_minus_b_message = matrix_mult_state_and_message.second;
  // Copy message.
  RoundThreeGradientDescentMessage round_three_message;
  *round_three_message.mutable_matrix_d_minus_matrix_b_message() =
      std::move(matrix_d_minus_b_message);
  StateRound3 state_three = {
      .share_x_transpose_minus_a = std::move(state.share_x_minus_a),
      .share_d_minus_b = std::move(state.share_y_minus_b)};
  return std::make_pair(std::move(state_three), round_three_message);
}

StatusOr<std::pair<StateRound3, RoundThreeGradientDescentMessage>>
GradientDescentPartyZero::GenerateGradientDescentRoundThreeMessage(
    SigmoidOutput sigmoid_output_share,
    StateMaskedXTranspose share_x_transpose_minus_a) {
  // Compute d = s - y
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_d,
                   VectorSubtract(sigmoid_output_share.sigmoid_output,
                                 share_y_, param_.modulus));
  // Set up to compute X.transpose() * d

  // Generate state and message to compute [g] = [X.transpose * d]:
  // * represents matrix product mod modulus.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
                   share_provider_->GetRoundThreeBeaverMatrixBandC());
  ASSIGN_OR_RETURN(
      auto matrix_mult_state_and_message,
      GenerateMatrixYminusBProductMessage(
          share_d, share_x_transpose_minus_a.share_x_transpose_minus_a,
          beaver_triple_matrix_b_c.first, param_.modulus));
  auto state = matrix_mult_state_and_message.first;
  auto matrix_d_minus_b_message = matrix_mult_state_and_message.second;
  // Copy message.
  RoundThreeGradientDescentMessage round_three_message;
  *round_three_message.mutable_matrix_d_minus_matrix_b_message() =
      std::move(matrix_d_minus_b_message);
  StateRound3 state_three = {
      .share_x_transpose_minus_a = std::move(state.share_x_minus_a),
      .share_d_minus_b = std::move(state.share_y_minus_b)};
  return std::make_pair(std::move(state_three), round_three_message);
}

Status GradientDescentPartyZero::ComputeGradientUpdateMinibatch(
    StateRound3 state_three,
    MaskedXTransposeMessage other_party_x_transpose_minus_a_message,
    RoundThreeGradientDescentMessage other_party_d_minus_b_message,
    size_t batch_size, size_t idx_batch, size_t size_per_minibatch) {
  // Compute [g] = [X.transpose * d].
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a_full,
                   share_provider_->GetRoundThreeBeaverMatrixA());
  // Use just the part necessary in this batch
  std::vector<uint64_t> beaver_triple_matrix_a (size_per_minibatch);
  for (size_t idx_feature = 0; idx_feature < param_.num_features; idx_feature++) {
    for (size_t idx_batchsize = 0; idx_batchsize < batch_size; idx_batchsize++) {
      beaver_triple_matrix_a[idx_feature * batch_size + idx_batchsize] =
          beaver_triple_matrix_a_full[idx_feature * param_.num_examples + idx_batch * batch_size + idx_batchsize];
    }
  }

  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
                   share_provider_->GetRoundThreeBeaverMatrixBandC());
  MatrixMultState state = {
      .share_x_minus_a = std::move(state_three.share_x_transpose_minus_a),
      .share_y_minus_b = std::move(state_three.share_d_minus_b)
  };
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_g,
      CorrelatedMatrixProductPartyZero(
          state, beaver_triple_matrix_a, beaver_triple_matrix_b_c.first,
          beaver_triple_matrix_b_c.second,
          other_party_x_transpose_minus_a_message
              .matrix_x_transpose_minus_matrix_a_message(),
          other_party_d_minus_b_message.matrix_d_minus_matrix_b_message(),
          param_.num_features, batch_size, 1,
          param_.num_fractional_bits, param_.modulus));

  // Compute scaled_g <-- (alpha/ X.size) * g
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_scaled_g,
                   ScalarVectorProductPartyZero(
                       param_.alpha / param_.num_examples, share_g,
                       fp_factory_,
                       param_.modulus));

  // scaled_regularization <-- (alpha * lambda / X.size()) * theta(t)
  double scaled_lambda = (param_.alpha * param_.lambda) / param_.num_examples;
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_scaled_regularization,
                   ScalarVectorProductPartyZero(
                       scaled_lambda, share_theta_,
                       fp_factory_,
                       param_.modulus));

  // scaled_g_regularized <-- scaled_g + scaled_regularization
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_scaled_g_regularized,
      VectorAdd(share_scaled_g,
                share_scaled_regularization,
                param_.modulus));

  // Compute update for Theta.
  ASSIGN_OR_RETURN(share_theta_, VectorSubtract(share_theta_,
                                                share_scaled_g_regularized,
                                                param_.modulus));

  // Throw away the consumed preprocessed shares.
  return share_provider_->Shrink();
}

Status GradientDescentPartyZero::ComputeGradientUpdate(
    StateRound3 state_three,
    MaskedXTransposeMessage other_party_x_transpose_minus_a_message,
    RoundThreeGradientDescentMessage other_party_d_minus_b_message) {
  // Compute [g] = [X.transpose * d].
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a,
                   share_provider_->GetRoundThreeBeaverMatrixA());
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
                   share_provider_->GetRoundThreeBeaverMatrixBandC());
  MatrixMultState state = {
    .share_x_minus_a = std::move(state_three.share_x_transpose_minus_a),
    .share_y_minus_b = std::move(state_three.share_d_minus_b)
  };
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_g,
      CorrelatedMatrixProductPartyZero(
          state, beaver_triple_matrix_a, beaver_triple_matrix_b_c.first,
          beaver_triple_matrix_b_c.second,
          other_party_x_transpose_minus_a_message
              .matrix_x_transpose_minus_matrix_a_message(),
          other_party_d_minus_b_message.matrix_d_minus_matrix_b_message(),
          param_.num_features, param_.num_examples, 1,
          param_.num_fractional_bits, param_.modulus));

  // Compute scaled_g <-- (alpha/ X.size) * g
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_scaled_g,
                   ScalarVectorProductPartyZero(
    param_.alpha / param_.num_examples, share_g,
    fp_factory_,
    param_.modulus));

  // scaled_regularization <-- (alpha * lambda / X.size()) * theta(t)
  double scaled_lambda = (param_.alpha * param_.lambda) / param_.num_examples;
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_scaled_regularization,
                   ScalarVectorProductPartyZero(
                       scaled_lambda, share_theta_,
                       fp_factory_,
                       param_.modulus));

  // scaled_g_regularized <-- scaled_g + scaled_regularization
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_scaled_g_regularized,
      VectorAdd(share_scaled_g,
                    share_scaled_regularization,
                    param_.modulus));

  // Compute update for Theta.
  ASSIGN_OR_RETURN(share_theta_, VectorSubtract(share_theta_,
                                               share_scaled_g_regularized,
                                               param_.modulus));

  // Throw away the consumed preprocessed shares.
  return share_provider_->Shrink();
}

// Init the gradient descent with inputs and parameters.
StatusOr<GradientDescentPartyOne> GradientDescentPartyOne::Init(
    std::vector<uint64_t> share_x,
    std::vector<uint64_t> share_y,
    std::vector<uint64_t> share_theta,
    std::unique_ptr<LogRegShareProvider> share_provider,
    const FixedPointElementFactory::Params& fpe_params,
    const GradientDescentParams& param) {
  // Verify that shapes of the input vectors and matrices agree with the
  // provided parameters.
  if (share_x.size() != param.num_examples * param.num_features ||
      share_y.size() != param.num_examples ||
      share_theta.size() != param.num_features) {
    return InvalidArgumentError("Gradient descent init: invalid input size.");
  }
  // Initialize the factory for fixed point elements.
  ASSIGN_OR_RETURN(auto temp,
                   FixedPointElementFactory::Create(param.num_fractional_bits,
                                                    param.num_ring_bits));
  auto fp_factory =
      absl::make_unique<FixedPointElementFactory>(std::move(temp));

  // Transpose share_x (Computing on a transpose will also be needed)
  // (num_example, num_features) -> (num_features, num_examples)
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_x_transpose,
                   Transpose(share_x, param.num_examples, param.num_features));

  // Initialize GradientDescentPartyOne.
  return GradientDescentPartyOne(share_x, share_x_transpose, share_y,
                                 share_theta, std::move(share_provider),
                                 std::move(fp_factory),
                                 param);
}

StatusOr<std::pair<StateMaskedX, MaskedXMessage>>
GradientDescentPartyOne::GenerateCorrelatedProductMessageForX() {
  // Get Beaver triple matrix ([A] only) for round one.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a,
                   share_provider_->GetRoundOneBeaverMatrixA());

  // Generate the state and message [X-A] to send to other party.
  ASSIGN_OR_RETURN(auto matrix_mult_state_and_message,
                   GenerateMatrixXminusAProductMessage(
                       share_x_, beaver_triple_matrix_a, param_.modulus));
  std::vector<uint64_t> share_x_minus_a = matrix_mult_state_and_message.first;
  MatrixXminusAProductMessage x_minus_a_message =
      matrix_mult_state_and_message.second;
  // Copy message.
  MaskedXMessage masked_x_message;
  *masked_x_message.mutable_matrix_x_minus_matrix_a_message() =
      std::move(x_minus_a_message);
  StateMaskedX state_x_minus_a = {.share_x_minus_a =
                                      std::move(share_x_minus_a)};
  return std::make_pair(state_x_minus_a, masked_x_message);
}

StatusOr<std::pair<StateMaskedXTranspose, MaskedXTransposeMessage>>
GradientDescentPartyOne::GenerateCorrelatedProductMessageForXTranspose() {
  // Set up to compute X.transpose() * d

  // Generate state and message to compute [g] = [X.transpose * d]:
  // * represents matrix product mod modulus.

  // Get Beaver triple matrix ([A], [B], [C]) for round three.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a,
                   share_provider_->GetRoundThreeBeaverMatrixA());

  // Generate the state and message [X^T-A] to send to other party.
  ASSIGN_OR_RETURN(
      auto matrix_mult_state_and_message,
      GenerateMatrixXminusAProductMessage(
          share_x_transpose_, beaver_triple_matrix_a, param_.modulus));
  auto share_x_transpose_minus_a = matrix_mult_state_and_message.first;
  auto x_transpose_minus_a_message = matrix_mult_state_and_message.second;
  // Copy message.
  MaskedXTransposeMessage masked_x_transpose_message;
  *masked_x_transpose_message
       .mutable_matrix_x_transpose_minus_matrix_a_message() =
      std::move(x_transpose_minus_a_message);
  StateMaskedXTranspose state_x_transpose_minus_a = {
      .share_x_transpose_minus_a = std::move(share_x_transpose_minus_a)};
  return std::make_pair(state_x_transpose_minus_a, masked_x_transpose_message);
}

StatusOr<std::pair<StateRound1, RoundOneGradientDescentMessage>>
GradientDescentPartyOne::GenerateGradientDescentRoundOneMessage(
    StateMaskedX share_x_minus_a) {
  // Get Beaver triple matrix ([A], [B], [C]) for round one.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
                   share_provider_->GetRoundOneBeaverMatrixBandC());

  // Generate the state and message ([X-A], [Theta-B]) to send to other party.
  ASSIGN_OR_RETURN(auto matrix_mult_state_and_message,
                   GenerateMatrixYminusBProductMessage(
                       share_theta_, share_x_minus_a.share_x_minus_a,
                       beaver_triple_matrix_b_c.first, param_.modulus));
  MatrixMultState state = matrix_mult_state_and_message.first;
  MatrixYminusBProductMessage matrix_theta_minus_b_message =
      matrix_mult_state_and_message.second;
  // Copy message.
  RoundOneGradientDescentMessage round_one_message;
  *round_one_message.mutable_matrix_theta_minus_matrix_b_message() =
      std::move(matrix_theta_minus_b_message);
  StateRound1 state_one = {
      .share_x_minus_a = std::move(state.share_x_minus_a),
      .share_theta_minus_b = std::move(state.share_y_minus_b)};
  return std::make_pair(state_one, round_one_message);
}

StatusOr<SigmoidInput> GradientDescentPartyOne::GenerateSigmoidInput(
    StateRound1 state_one, MaskedXMessage other_party_x_minus_a_message,
    RoundOneGradientDescentMessage other_party_theta_minus_b_message) {
  // Get Beaver triple matrix ([A], [B], [C]) used in round one.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a,
                   share_provider_->GetRoundOneBeaverMatrixA());
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
                   share_provider_->GetRoundOneBeaverMatrixBandC());
  // Convert StateRound1 to the MatrixMultState struct.
  MatrixMultState state = {
      .share_x_minus_a = std::move(state_one.share_x_minus_a),
      .share_y_minus_b = std::move(state_one.share_theta_minus_b)};
  // Compute [U] = [X*Theta]: * represents matrix product mod modulus.
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_u,
      CorrelatedMatrixProductPartyOne(
          state, beaver_triple_matrix_a, beaver_triple_matrix_b_c.first,
          beaver_triple_matrix_b_c.second,
          other_party_x_minus_a_message.matrix_x_minus_matrix_a_message(),
          other_party_theta_minus_b_message
              .matrix_theta_minus_matrix_b_message(),
          param_.num_examples, param_.num_features, 1,
          param_.num_fractional_bits, param_.modulus));
  SigmoidInput sigmoid_input = {.sigmoid_input = std::move(share_u)};
  return sigmoid_input;
}

StatusOr<SigmoidInput> GradientDescentPartyOne::GenerateSigmoidInputMinibatch(
    StateRound1 state_one, MaskedXMessage other_party_x_minus_a_message,
    RoundOneGradientDescentMessage other_party_theta_minus_b_message,
    size_t batch_size, size_t idx_batch, size_t size_per_minibatch) {
  // Get Beaver triple matrix ([A], [B], [C]) used in round one.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a_full,
                   share_provider_->GetRoundOneBeaverMatrixA());
  // Retrieve the correct batch from this beaver matrix a
  size_t start_idx = idx_batch * size_per_minibatch;
  std::vector<uint64_t> beaver_triple_matrix_a (size_per_minibatch);
  for (size_t idx = 0; idx < size_per_minibatch; idx++) {
    beaver_triple_matrix_a[idx] = beaver_triple_matrix_a_full[start_idx + idx];
  }
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
                   share_provider_->GetRoundOneBeaverMatrixBandC());
  // Convert StateRound1 to the MatrixMultState struct.
  MatrixMultState state = {
      .share_x_minus_a = std::move(state_one.share_x_minus_a),
      .share_y_minus_b = std::move(state_one.share_theta_minus_b)};
  // Compute [U] = [X*Theta]: * represents matrix product mod modulus.
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_u,
      CorrelatedMatrixProductPartyOne(
          state, beaver_triple_matrix_a, beaver_triple_matrix_b_c.first,
          beaver_triple_matrix_b_c.second,
          other_party_x_minus_a_message.matrix_x_minus_matrix_a_message(),
          other_party_theta_minus_b_message
              .matrix_theta_minus_matrix_b_message(),
          batch_size, param_.num_features, 1,
          param_.num_fractional_bits, param_.modulus));
  SigmoidInput sigmoid_input = {.sigmoid_input = std::move(share_u)};
  return sigmoid_input;
}

StatusOr<std::pair<SigmoidOutput, SigmoidOutput>>
GradientDescentPartyOne::GenerateSigmoidOutputForTesting(
    SigmoidInput sigmoid_input_share_p0, SigmoidInput sigmoid_input_share_p1) {
  size_t batch_length = param_.num_examples;

  // Reconstruct the sigmoid inputs
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> reconstructed_input,
      BatchedModAdd(sigmoid_input_share_p0.sigmoid_input,
                    sigmoid_input_share_p1.sigmoid_input, param_.modulus));
  // Compute sigmoid in plaintext
  std::vector<uint64_t> sigmoid_output(batch_length, 0);
  for (size_t idx = 0; idx < batch_length; idx++) {
    // Compute sigmoid on double for simplicity
    double reconstructed_double_input =
        fp_factory_->ImportFixedPointElementFromUint64(reconstructed_input[idx])
            ->ExportToDouble();
    double double_output = 1. / (1. + exp(-reconstructed_double_input));
    ASSIGN_OR_RETURN(
        FixedPointElement fpe_output,
        fp_factory_->CreateFixedPointElementFromDouble(double_output));
    sigmoid_output[idx] = fpe_output.ExportToUint64();
  }
  // Secret share the computed sigmoid
  ASSIGN_OR_RETURN(auto share_zero,
                   internal::SampleShareOfZero(batch_length, param_.modulus));
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> sigmoid_output_p0_share,
      BatchedModAdd(share_zero.first, sigmoid_output, param_.modulus));
  std::vector<uint64_t> sigmoid_output_p1_share = std::move(share_zero.second);

  // Convert to SigmoidOutput
  SigmoidOutput sigmoid_output_p0_struct = {
      .sigmoid_output = std::move(sigmoid_output_p0_share)};
  SigmoidOutput sigmoid_output_p1_struct = {
      .sigmoid_output = std::move(sigmoid_output_p1_share)};

  return std::make_pair(std::move(sigmoid_output_p0_struct),
                        std::move(sigmoid_output_p1_struct));
}

StatusOr<std::pair<StateRound3, RoundThreeGradientDescentMessage>>
GradientDescentPartyOne::GenerateGradientDescentRoundThreeMessageMinibatch(
    SigmoidOutput sigmoid_output_share,
    StateMaskedXTranspose share_x_transpose_minus_a,
    size_t batch_size, size_t idx_batch) {
  // Retrieve the correct batch from this share_y
  size_t start_idx = idx_batch * batch_size;
  std::vector<uint64_t> share_y_minibatch (batch_size);
  for (size_t idx = 0; idx < batch_size; idx++) {
    share_y_minibatch[idx] = share_y_[start_idx + idx];
  }
  // Compute d = s - y
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_d,
                   VectorSubtract(sigmoid_output_share.sigmoid_output, share_y_minibatch,
                                  param_.modulus));

  // Set up to compute X.transpose() * d

  // Generate state and message to compute [g] = [X.transpose * d]:
  // * represents matrix product mod modulus.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix,
                   share_provider_->GetRoundThreeBeaverMatrixBandC());
  ASSIGN_OR_RETURN(
      auto matrix_mult_state_and_message,
      GenerateMatrixYminusBProductMessage(
          share_d, share_x_transpose_minus_a.share_x_transpose_minus_a,
          beaver_triple_matrix.first, param_.modulus));
  auto state = matrix_mult_state_and_message.first;
  auto matrix_d_minus_b_message = matrix_mult_state_and_message.second;
  // Copy message.
  RoundThreeGradientDescentMessage round_three_message;
  *round_three_message.mutable_matrix_d_minus_matrix_b_message() =
      std::move(matrix_d_minus_b_message);
  StateRound3 state_three = {
      .share_x_transpose_minus_a = std::move(state.share_x_minus_a),
      .share_d_minus_b = std::move(state.share_y_minus_b)};
  return std::make_pair(std::move(state_three), round_three_message);
}

StatusOr<std::pair<StateRound3, RoundThreeGradientDescentMessage>>
GradientDescentPartyOne::GenerateGradientDescentRoundThreeMessage(
    SigmoidOutput sigmoid_output_share,
    StateMaskedXTranspose share_x_transpose_minus_a) {
  // Compute d = s - y
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_d,
                   VectorSubtract(sigmoid_output_share.sigmoid_output, share_y_,
                                  param_.modulus));

  // Set up to compute X.transpose() * d

  // Generate state and message to compute [g] = [X.transpose * d]:
  // * represents matrix product mod modulus.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix,
                   share_provider_->GetRoundThreeBeaverMatrixBandC());
  ASSIGN_OR_RETURN(
      auto matrix_mult_state_and_message,
      GenerateMatrixYminusBProductMessage(
          share_d, share_x_transpose_minus_a.share_x_transpose_minus_a,
          beaver_triple_matrix.first, param_.modulus));
  auto state = matrix_mult_state_and_message.first;
  auto matrix_d_minus_b_message = matrix_mult_state_and_message.second;
  // Copy message.
  RoundThreeGradientDescentMessage round_three_message;
  *round_three_message.mutable_matrix_d_minus_matrix_b_message() =
      std::move(matrix_d_minus_b_message);
  StateRound3 state_three = {
      .share_x_transpose_minus_a = std::move(state.share_x_minus_a),
      .share_d_minus_b = std::move(state.share_y_minus_b)};
  return std::make_pair(std::move(state_three), round_three_message);
}

Status GradientDescentPartyOne::ComputeGradientUpdateMinibatch(
    StateRound3 state_three,
    MaskedXTransposeMessage other_party_x_transpose_minus_a_message,
    RoundThreeGradientDescentMessage other_party_d_minus_b_message,
    size_t batch_size, size_t idx_batch, size_t size_per_minibatch) {
  // Compute [g] = [X.transpose * d].
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a_full,
                   share_provider_->GetRoundThreeBeaverMatrixA());
  // Use just the part necessary in this batch
  std::vector<uint64_t> beaver_triple_matrix_a (size_per_minibatch);
  for (size_t idx_feature = 0; idx_feature < param_.num_features; idx_feature++) {
    for (size_t idx_batchsize = 0; idx_batchsize < batch_size; idx_batchsize++) {
      beaver_triple_matrix_a[idx_feature * batch_size + idx_batchsize] =
          beaver_triple_matrix_a_full[idx_feature * param_.num_examples + idx_batch * batch_size + idx_batchsize];
    }
  }

  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
                   share_provider_->GetRoundThreeBeaverMatrixBandC());
  MatrixMultState state = {
      .share_x_minus_a = std::move(state_three.share_x_transpose_minus_a),
      .share_y_minus_b = std::move(state_three.share_d_minus_b)
  };
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_g,
      CorrelatedMatrixProductPartyOne(
          state, beaver_triple_matrix_a, beaver_triple_matrix_b_c.first,
          beaver_triple_matrix_b_c.second,
          other_party_x_transpose_minus_a_message
              .matrix_x_transpose_minus_matrix_a_message(),
          other_party_d_minus_b_message.matrix_d_minus_matrix_b_message(),
          param_.num_features, batch_size, 1,
          param_.num_fractional_bits, param_.modulus));

  // Compute scaled_g <-- (alpha/ X.size) * g
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_scaled_g,
      ScalarVectorProductPartyOne(param_.alpha / param_.num_examples, share_g,
                                  fp_factory_, param_.modulus));

  // scaled_regularization <-- (alpha * lambda / X.size()) * theta(t)
  double scaled_lambda = (param_.alpha * param_.lambda) / param_.num_examples;
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_scaled_regularization,
                   ScalarVectorProductPartyOne(
                       scaled_lambda, share_theta_,
                       fp_factory_,
                       param_.modulus));

  // scaled_g_regularized <-- scaled_g + scaled_regularization
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_scaled_g_regularized,
      VectorAdd(share_scaled_g,
                share_scaled_regularization,
                param_.modulus));

  // Compute update for Theta.
  ASSIGN_OR_RETURN(share_theta_, VectorSubtract(share_theta_,
                                                share_scaled_g_regularized,
                                                param_.modulus));

  // Throw away the consumed preprocessed shares.
  return share_provider_->Shrink();
}

Status GradientDescentPartyOne::ComputeGradientUpdate(
    StateRound3 state_three,
    MaskedXTransposeMessage other_party_x_transpose_minus_a_message,
    RoundThreeGradientDescentMessage other_party_d_minus_b_message) {
  // Compute [g] = [X.transpose * d].
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a,
                   share_provider_->GetRoundThreeBeaverMatrixA());
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
                   share_provider_->GetRoundThreeBeaverMatrixBandC());
  MatrixMultState state = {
    .share_x_minus_a = std::move(state_three.share_x_transpose_minus_a),
    .share_y_minus_b = std::move(state_three.share_d_minus_b)
  };
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_g,
      CorrelatedMatrixProductPartyOne(
          state, beaver_triple_matrix_a, beaver_triple_matrix_b_c.first,
          beaver_triple_matrix_b_c.second,
          other_party_x_transpose_minus_a_message
              .matrix_x_transpose_minus_matrix_a_message(),
          other_party_d_minus_b_message.matrix_d_minus_matrix_b_message(),
          param_.num_features, param_.num_examples, 1,
          param_.num_fractional_bits, param_.modulus));

  // Compute scaled_g <-- (alpha/ X.size) * g
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_scaled_g,
      ScalarVectorProductPartyOne(param_.alpha / param_.num_examples, share_g,
                                  fp_factory_, param_.modulus));

  // scaled_regularization <-- (alpha * lambda / X.size()) * theta(t)
  double scaled_lambda = (param_.alpha * param_.lambda) / param_.num_examples;
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_scaled_regularization,
                   ScalarVectorProductPartyOne(
                       scaled_lambda, share_theta_,
                       fp_factory_,
                       param_.modulus));

  // scaled_g_regularized <-- scaled_g + scaled_regularization
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_scaled_g_regularized,
      VectorAdd(share_scaled_g,
                    share_scaled_regularization,
                    param_.modulus));

  // Compute update for Theta.
  ASSIGN_OR_RETURN(share_theta_, VectorSubtract(share_theta_,
                                                share_scaled_g_regularized,
                                                param_.modulus));

  // Throw away the consumed preprocessed shares.
  return share_provider_->Shrink();
}

}  // namespace logistic_regression
}  // namespace private_join_and_compute
