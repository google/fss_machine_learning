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

#include "gradient_descent_dp.h"

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
#include "secret_sharing_mpc/gates/vector_subtraction.h"
#include "absl/memory/memory.h"

namespace private_join_and_compute {
namespace logistic_regression_dp {

// Init the gradient descent with inputs and parameters.
StatusOr<GradientDescentPartyZero> GradientDescentPartyZero::Init(
    std::vector<uint64_t> share_x,
    std::vector<uint64_t> share_y,
    std::vector<double> theta,
    std::unique_ptr<LogRegDPShareProvider> share_provider,
    const FixedPointElementFactory::Params& fpe_params,
    const GradientDescentParams& param) {
  // Verify that shapes of the input vectors and matrices agree with the
  // provided parameters.
  if (share_x.size() != param.num_examples * param.num_features ||
      share_y.size() != param.num_examples ||
      theta.size() != param.num_features) {
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
                                  theta, std::move(share_provider),
                                  std::move(fp_factory),
                                  param);
}

StatusOr<std::pair<StateMaskedXTranspose, MaskedXTransposeMessage>>
GradientDescentPartyZero::GenerateCorrelatedProductMessageForXTranspose() {
  // Set up to compute X.transpose() * d

  // Generate state and message to compute [g] = [X.transpose * d]:
  // * represents matrix product mod modulus.

  // Get Beaver triple matrix ([A], [B], [C]) for round three.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a,
      share_provider_->GetXTransposeDMatrixA());

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

StatusOr<SigmoidInput> GradientDescentPartyZero::GenerateSigmoidInput() {
  // Compute [U] = [X*Theta]: * represents matrix product mod modulus.
  // TODO Create more suitable gate for this operation
  std::vector<uint64_t> share_u (param_.num_examples, 0);
  for (unsigned int idx_example = 0; idx_example < param_.num_examples; idx_example++) {
    for (unsigned int idx_feature = 0; idx_feature < param_.num_features; idx_feature++) {
      ASSIGN_OR_RETURN(std::vector<uint64_t> prod, ScalarVectorProductPartyZero(theta_[idx_feature],
          {share_x_[idx_example * param_.num_features + idx_feature]}, fp_factory_, param_.modulus));
      share_u[idx_example] =
          ModAdd(share_u[idx_example], prod[0], param_.modulus);  // prod is vector of 1 element
    }
  }
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
    std::cerr << "sigmoid output " << double_output << std::endl;
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

StatusOr<std::pair<StateXTransposeD, XTransposeDMessage>>
GradientDescentPartyZero::GenerateXTransposeDMessage(
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
      share_provider_->GetXTransposeDMatrixBandC());
  ASSIGN_OR_RETURN(
      auto matrix_mult_state_and_message,
      GenerateMatrixYminusBProductMessage(
          share_d, share_x_transpose_minus_a.share_x_transpose_minus_a,
          beaver_triple_matrix_b_c.first, param_.modulus));
  auto state = matrix_mult_state_and_message.first;
  auto matrix_d_minus_b_message = matrix_mult_state_and_message.second;
  // Copy message.
  XTransposeDMessage out_message;
  *out_message.mutable_matrix_d_minus_matrix_b_message() =
      std::move(matrix_d_minus_b_message);
  StateXTransposeD out_state = {
      .share_x_transpose_minus_a = std::move(state.share_x_minus_a),
      .share_d_minus_b = std::move(state.share_y_minus_b)
  };
  return std::make_pair(out_state, out_message);
}

StatusOr<std::pair<StateReconstructGradient, ReconstructGradientMessage>>
GradientDescentPartyZero::GenerateReconstructGradientMessage(
    StateXTransposeD state_x_transpose_d,
    XTransposeDMessage x_transpose_d_message,
    MaskedXTransposeMessage x_transpose_minus_a_message) {
  // Compute [g] = [X.transpose * d].
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a,
      share_provider_->GetXTransposeDMatrixA());
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
      share_provider_->GetXTransposeDMatrixBandC());
  MatrixMultState state = {
      .share_x_minus_a = std::move(state_x_transpose_d.share_x_transpose_minus_a),
      .share_y_minus_b = std::move(state_x_transpose_d.share_d_minus_b)
  };
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_g,
      CorrelatedMatrixProductPartyZero(
          state, beaver_triple_matrix_a, beaver_triple_matrix_b_c.first,
          beaver_triple_matrix_b_c.second,
          x_transpose_minus_a_message.matrix_x_transpose_minus_matrix_a_message(),
          x_transpose_d_message.matrix_d_minus_matrix_b_message(),
          param_.num_features, param_.num_examples, 1,
          param_.num_fractional_bits, param_.modulus));

  // Compute scaled_g <-- (1 / X.size) * g
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_scaled_g,
                   ScalarVectorProductPartyZero(
                       param_.alpha / param_.num_examples,
                       share_g,
                       fp_factory_,
                       param_.modulus));

  // Compute scaled_g_noise <-- scaled_g + alpha * noise
  ASSIGN_OR_RETURN(std::vector<uint64_t> noise,
                   share_provider_->GetNoise());
  ASSIGN_OR_RETURN(std::vector<uint64_t> scaled_noise,
                   ScalarVectorProductPartyZero(
                       param_.alpha,
                       noise,
                       fp_factory_,
                       param_.modulus));
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_scaled_g_noise,
      BatchedModAdd(share_scaled_g,
                    scaled_noise,
                    param_.modulus));

  // Compute scaled_regularization <-- (lambda / X.size()) * theta(t)
  std::vector<uint64_t> share_scaled_regularization (param_.num_features, 0);
  double scaled_lambda = (param_.alpha * param_.lambda) / param_.num_examples;
  for (unsigned int i = 0; i < param_.num_features; i++) {
    ASSIGN_OR_RETURN(FixedPointElement tmp_scaled_regularization,
        fp_factory_->CreateFixedPointElementFromDouble(scaled_lambda * theta_[i]));
    share_scaled_regularization[i] =
        tmp_scaled_regularization.ExportToUint64();
  }

  // Compute scaled_g_noise_regularized <-- scaled_g_noise + scaled_regularization
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_scaled_g_noise_regularized,
      BatchedModAdd(share_scaled_g_noise,
                    share_scaled_regularization,
                    param_.modulus));

  // Construct message from scaled_g_noise_regularized share.
  ReconstructGradientMessage out_message;
  for (size_t idx = 0; idx < param_.num_features; idx++) {
    out_message.add_vector_scaled_g_noise_regularized_gradient(share_scaled_g_noise_regularized[idx]);
  }
  StateReconstructGradient out_state = {
      .share_scaled_g_noise_regularized = std::move(share_scaled_g_noise)
  };

  return std::make_pair(out_state, out_message);
}

Status GradientDescentPartyZero::ComputeGradientUpdate(
    StateReconstructGradient state_reconstruct_gradient,
    ReconstructGradientMessage message_reconstruct_gradient) {
  // Reconstruct scaled_g_noise_regularized
  std::vector<double> scaled_g_noise_regularized (param_.num_features, 0);
  for (size_t idx = 0; idx < param_.num_features; idx++) {
    scaled_g_noise_regularized[idx] = fp_factory_->ImportFixedPointElementFromUint64(ModAdd(message_reconstruct_gradient.vector_scaled_g_noise_regularized_gradient(idx),
                                                                                state_reconstruct_gradient.share_scaled_g_noise_regularized[idx],
                                                                                param_.modulus))->ExportToDouble();
  }

  // Compute update for Theta.
  for (unsigned int idx = 0; idx < param_.num_features; idx++) {
    theta_[idx] -= scaled_g_noise_regularized[idx];
  }

  // Throw away the consumed preprocessed shares.
  return share_provider_->Shrink();
}

// Init the gradient descent with inputs and parameters.
StatusOr<GradientDescentPartyOne> GradientDescentPartyOne::Init(
    std::vector<uint64_t> share_x,
    std::vector<uint64_t> share_y,
    std::vector<double> theta,
    std::unique_ptr<LogRegDPShareProvider> share_provider,
    const FixedPointElementFactory::Params& fpe_params,
    const GradientDescentParams& param) {
  // Verify that shapes of the input vectors and matrices agree with the
  // provided parameters.
  if (share_x.size() != param.num_examples * param.num_features ||
      share_y.size() != param.num_examples ||
      theta.size() != param.num_features) {
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
                                 theta, std::move(share_provider),
                                 std::move(fp_factory),
                                 param);
}

StatusOr<std::pair<StateMaskedXTranspose, MaskedXTransposeMessage>>
GradientDescentPartyOne::GenerateCorrelatedProductMessageForXTranspose() {
  // Set up to compute X.transpose() * d

  // Generate state and message to compute [g] = [X.transpose * d]:
  // * represents matrix product mod modulus.

  // Get Beaver triple matrix ([A], [B], [C]) for round three.
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a,
      share_provider_->GetXTransposeDMatrixA());

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

StatusOr<SigmoidInput> GradientDescentPartyOne::GenerateSigmoidInput() {
  // Compute [U] = [X*Theta]: * represents matrix product mod modulus.
  // TODO Create more suitable gate for this operation
  std::vector<uint64_t> share_u (param_.num_examples, 0);
  for (unsigned int idx_example = 0; idx_example < param_.num_examples; idx_example++) {
    for (unsigned int idx_feature = 0; idx_feature < param_.num_features; idx_feature++) {
      ASSIGN_OR_RETURN(std::vector<uint64_t> prod, ScalarVectorProductPartyOne(theta_[idx_feature],
          {share_x_[idx_example * param_.num_features + idx_feature]}, fp_factory_, param_.modulus));
      share_u[idx_example] =
          ModAdd(share_u[idx_example], prod[0], param_.modulus);
    }
  }
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

StatusOr<std::pair<StateXTransposeD, XTransposeDMessage>>
GradientDescentPartyOne::GenerateXTransposeDMessage(
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
                   share_provider_->GetXTransposeDMatrixBandC());
  ASSIGN_OR_RETURN(
      auto matrix_mult_state_and_message,
      GenerateMatrixYminusBProductMessage(
          share_d, share_x_transpose_minus_a.share_x_transpose_minus_a,
          beaver_triple_matrix_b_c.first, param_.modulus));
  auto state = matrix_mult_state_and_message.first;
  auto matrix_d_minus_b_message = matrix_mult_state_and_message.second;
  // Copy message.
  XTransposeDMessage out_message;
  *out_message.mutable_matrix_d_minus_matrix_b_message() =
      std::move(matrix_d_minus_b_message);
  StateXTransposeD out_state = {
      .share_x_transpose_minus_a = std::move(state.share_x_minus_a),
      .share_d_minus_b = std::move(state.share_y_minus_b)
  };
  return std::make_pair(out_state, out_message);
}

StatusOr<std::pair<StateReconstructGradient, ReconstructGradientMessage>>
GradientDescentPartyOne::GenerateReconstructGradientMessage(
    StateXTransposeD state_x_transpose_d,
    XTransposeDMessage x_transpose_d_message,
    MaskedXTransposeMessage x_transpose_minus_a_message) {
  // Compute [g] = [X.transpose * d].
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_a,
      share_provider_->GetXTransposeDMatrixA());
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_b_c,
      share_provider_->GetXTransposeDMatrixBandC());
  MatrixMultState state = {
      .share_x_minus_a = std::move(state_x_transpose_d.share_x_transpose_minus_a),
      .share_y_minus_b = std::move(state_x_transpose_d.share_d_minus_b)
  };
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_g,
      CorrelatedMatrixProductPartyOne(
          state, beaver_triple_matrix_a, beaver_triple_matrix_b_c.first,
          beaver_triple_matrix_b_c.second,
          x_transpose_minus_a_message.matrix_x_transpose_minus_matrix_a_message(),
          x_transpose_d_message.matrix_d_minus_matrix_b_message(),
          param_.num_features, param_.num_examples, 1,
          param_.num_fractional_bits, param_.modulus));

  // Compute scaled_g <-- (alpha / X.size) * g
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_scaled_g,
                   ScalarVectorProductPartyOne(
                       param_.alpha / param_.num_examples,
                       share_g,
                       fp_factory_,
                       param_.modulus));

  // Compute scaled_g_noise <-- scaled_g + alpha * noise
  ASSIGN_OR_RETURN(std::vector<uint64_t> noise,
                   share_provider_->GetNoise());
  ASSIGN_OR_RETURN(std::vector<uint64_t> scaled_noise,
                   ScalarVectorProductPartyOne(
                       param_.alpha,
                       noise,
                       fp_factory_,
                       param_.modulus));
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_scaled_g_noise,
      BatchedModAdd(share_scaled_g,
                    scaled_noise,
                    param_.modulus));

  // Compute scaled_regularization <-- (lambda / X.size()) * theta(t)
  std::vector<uint64_t> share_scaled_regularization (param_.num_features, 0);
  double scaled_lambda = (param_.alpha * param_.lambda) / param_.num_examples;
  for (unsigned int i = 0; i < param_.num_features; i++) {
    ASSIGN_OR_RETURN(FixedPointElement tmp_scaled_regularization,
        fp_factory_->CreateFixedPointElementFromDouble(scaled_lambda * theta_[i]));
    share_scaled_regularization[i] =
        tmp_scaled_regularization.ExportToUint64();
  }

  // Compute scaled_g_noise_regularized <-- scaled_g_noise + scaled_regularization
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> share_scaled_g_noise_regularized,
      BatchedModAdd(share_scaled_g_noise,
                    share_scaled_regularization,
                    param_.modulus));

  // Construct message from scaled_g_noise_regularized share.
  ReconstructGradientMessage out_message;
  for (size_t idx = 0; idx < param_.num_features; idx++) {
    out_message.add_vector_scaled_g_noise_regularized_gradient(share_scaled_g_noise_regularized[idx]);
  }
  StateReconstructGradient out_state = {
      .share_scaled_g_noise_regularized = std::move(share_scaled_g_noise)
  };
  return std::make_pair(std::move(out_state), out_message);
}


Status GradientDescentPartyOne::ComputeGradientUpdate(
    StateReconstructGradient state_reconstruct_gradient,
    ReconstructGradientMessage message_reconstruct_gradient) {
  // Reconstruct alpha_scaled_g_noise_regularized
  std::vector<double> scaled_g_noise_regularized (param_.num_features, 0);
  for (size_t idx = 0; idx < param_.num_features; idx++) {
    scaled_g_noise_regularized[idx] = fp_factory_->ImportFixedPointElementFromUint64(ModAdd(message_reconstruct_gradient.vector_scaled_g_noise_regularized_gradient(idx),
                                                                                state_reconstruct_gradient.share_scaled_g_noise_regularized[idx],
                                                                                param_.modulus))->ExportToDouble();
  }

  // Compute update for Theta.
  for (unsigned int idx = 0; idx < param_.num_features; idx++) {
    theta_[idx] -= scaled_g_noise_regularized[idx];
  }

  // Throw away the consumed preprocessed shares.
  return share_provider_->Shrink();
}

}  // namespace logistic_regression_dp
}  // namespace private_join_and_compute

