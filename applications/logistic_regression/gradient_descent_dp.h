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

#ifndef GOOGLE_CODE_APPLICATIONS_LOGISTIC_REGRESSION_GRADIENT_DESCENT_DP_H_
#define GOOGLE_CODE_APPLICATIONS_LOGISTIC_REGRESSION_GRADIENT_DESCENT_DP_H_

#include "applications/logistic_regression/gradient_descent_dp_messages.pb.h"
#include "applications/logistic_regression/gradient_descent_dp_utils.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/secure_exponentiation.h"

namespace private_join_and_compute {
namespace logistic_regression_dp {

// Execute the gradient descent.
// The gradient descent is computed as:
// theta(t+1) = theta(t) -
//              alpha * [(1 / X.size) * X.transpose * (sigmoid(X * theta(t)) - y) + noise + (lambda / X.size) * theta(t)]
// theta(t+1) = theta(t) -
//              [(alpha / X.size) * X.transpose * (sigmoid(X * theta(t)) - y) + alpha * noise + alpha * lambda / X.size * theta(t)]
// Notation:
//
// alpha, lambda, and X.size are public scalars
// theta(t) is public vector of doubles (not uint64_t)
// X, y are private & secret shared
// X is n x k matrix (k already includes the intercept)
// y is n x 1 vector
// theta is k x 1 vector
//
// u <-- <X, theta(t)>. u is the matrix_product between private X and public theta(t)
// and results in [u], secret shares of X * theta(t)
// u is a n x 1 vector
// s <-- sigmoid(u). s is the element-wise sigmoid of each element in vector u.
// s is a n x 1 vector
// d <-- s - y. - is vector_subtraction.
// d is a n x 1 vector
// g <-- <X.transpose, d>, g is the matrix_vector product between X.transpose
// and d.
// g is k x 1 vector
// scaled_g <-- (alpha / X.size) * g where * represents scalar_vector_product.
// scaled_g is k x 1 vector
// scaled_g_noise <-- scaled_g + alpha * noise
// scaled_g_noise is k x 1 vector
// scaled_regularization <-- (alpha * lambda / X.size()) * theta(t)
// scaled_regularization is kx1 vector
// scaled_g_noise_regularized <-- scaled_g_noise + scaled_regularization
// scaled_g_noise.reveal()
// public operation: theta(t+1) = theta(t) - revealed_gradient

// Struct to store gradient descent parameters.
struct GradientDescentParams {
  size_t num_examples;          // Num of training examples in X.
  size_t num_features;          // Num of features of X (including intercept).
  size_t num_iterations;        // Number of iterations: num_iterations.
  uint8_t num_ring_bits;        // Ring size of the fixed point (l).
  uint8_t num_fractional_bits;  // Num of fractional bits in fixed point (lf).
  double alpha;                 // Learning rate: alpha.
  double lambda;                // regularization parameter.
  uint64_t modulus;             // Ring: modulus.
};

class GradientDescentPartyZero {
 public:
  // Init the gradient descent with inputs and parameters.
  static StatusOr<GradientDescentPartyZero> Init(
      std::vector<uint64_t> share_x,
      std::vector<uint64_t> share_y,
      std::vector<double> theta,
      std::unique_ptr<LogRegDPShareProvider> share_provider,
      const FixedPointElementFactory::Params& fpe_params,
      const GradientDescentParams& param);

  // Function to generate message for correlated beaver triple products
  // Generates the message and state associated with masking X^T
  // The message contains shares [X^T - A] and can be reused for all training
  // steps
  StatusOr<std::pair<StateMaskedXTranspose, MaskedXTransposeMessage>>
  GenerateCorrelatedProductMessageForXTranspose();

  // Functions to generate intermediate messages during the gradient descent
  // iterations.

  // Generates the sigmoid inputs by computing the shared output of the matrix
  // product [u] = [X] * theta.
  // u is a vector and sigmoid will be computed for each entry of u.
  StatusOr<SigmoidInput> GenerateSigmoidInput();

  // Functions to evaluate Sigmoid on the vector u (defined in applications/secure_sigmoid)
  // Invoke one by one at each party:
  // GenerateSigmoidRoundOneMessage
  // GenerateSigmoidRoundTwoMessage
  // GenerateSigmoidRoundThreeMessage
  // GenerateSigmoidRoundFourMessage
  // GenerateSigmoidRoundFiveMessage (depending on which sigmoid is used)
  // GenerateSigmoidResult

  // Since these functions are defined outside of this class, we need to be able to
  // access share_provider.
  // TODO create these functions also in this class and make share_provider private
  std::unique_ptr<LogRegDPShareProvider> share_provider_;


  // Generates the message and state needed to compute the matrix product
  // g = X.transpose() * d.
  // First, the function finishes computing the shared output of d = s - y.
  // After that, the state/message is a function of shares [X], [d] and the
  // preprocessed Beaver triple matrix.
  StatusOr<std::pair<StateXTransposeD, XTransposeDMessage>>
  GenerateXTransposeDMessage(
      SigmoidOutput sigmoid_output_share,
      StateMaskedXTranspose share_x_transpose_minus_a);

  // Computes:
  // [g] = [X.transpose] * [d] and then
  // scaled_g = (alpha / X.size) * g   (no new communication needed)
  // scaled_g_noise <-- scaled_g + alpha * noise
  // returns as messages share of scaled_g_noise
  StatusOr<std::pair<StateReconstructGradient, ReconstructGradientMessage>>
  GenerateReconstructGradientMessage(
      StateXTransposeD state_x_transpose_d,
      XTransposeDMessage x_transpose_d_message,
      MaskedXTransposeMessage x_transpose_minus_a_message);

  // Computes the gradient descent update.
  // Reconstructs revealed gradient
  // Then the weights theta are updated as:
  // public operation: theta(t+1) = theta(t) - alpha * revealed_gradient
  Status ComputeGradientUpdate(
      StateReconstructGradient state_reconstruct_gradient,
      ReconstructGradientMessage message_reconstruct_gradient);

  // Return the intermediate shares of logistic regression model.
  const std::vector<double>& GetTheta() const {
    return theta_;
  }

  // Test function to compute Sigmoid Outputs
  // This temporary function receives the input shares from each party and
  // outputs shares for BOTH parties.
  // The temporary function is insecure.
  StatusOr<std::pair<SigmoidOutput, SigmoidOutput>>
  GenerateSigmoidOutputForTesting(SigmoidInput sigmoid_input_share_p0,
                                  SigmoidInput sigmoid_input_share_p1);

 private:
  // Constructor: initialize the gradient descent with shares of training data,
  // training parameters, and the share initialized theta.
  GradientDescentPartyZero(
      std::vector<uint64_t> share_x, std::vector<uint64_t> share_x_transpose,
      std::vector<uint64_t> share_y, std::vector<double> theta,
      std::unique_ptr<LogRegDPShareProvider> share_provider,
      std::unique_ptr<FixedPointElementFactory> fp_factory,
      const GradientDescentParams& param)
      : share_x_(std::move(share_x)),
        share_x_transpose_(std::move(share_x_transpose)),
        share_y_(std::move(share_y)),
        theta_(std::move(theta)),
        share_provider_(std::move(share_provider)),
        fp_factory_(std::move(fp_factory)),
        param_(param) {}

  // Stores the training data ([X], [Y]) and [X^T] for the gradient descent.
  std::vector<uint64_t> share_x_;
  std::vector<uint64_t> share_x_transpose_;
  std::vector<uint64_t> share_y_;

  // Theta stores the result of the regression.
  std::vector<double> theta_;

  // Stores the preprocessed data needed for gradient descent.
  // TODO Uncomment after you move sigmoid functions to this class
  // std::unique_ptr<LogRegDPShareProvider> share_provider_;

  // Stores the fixed point factory.
  std::unique_ptr<FixedPointElementFactory> fp_factory_;

  // The parameters used for gradient descent.
  const GradientDescentParams param_;
};

class GradientDescentPartyOne {
 public:
  // Init the gradient descent with inputs and parameters.
  static StatusOr<GradientDescentPartyOne> Init(
      std::vector<uint64_t> share_x,
      std::vector<uint64_t> share_y,
      std::vector<double> theta,
      std::unique_ptr<LogRegDPShareProvider> share_provider,
      const FixedPointElementFactory::Params& fpe_params,
      const GradientDescentParams& param);

  // Function to generate message for correlated beaver triple products
  // Generates the message and state associated with masking X^T
  // The message contains shares [X^T - A] and can be reused for all training
  // steps.
  StatusOr<std::pair<StateMaskedXTranspose, MaskedXTransposeMessage>>
  GenerateCorrelatedProductMessageForXTranspose();

  // Functions to generate intermediate messages during the gradient descent
  // iterations.

  // Generates the sigmoid inputs by computing the shared output of the matrix
  // product [u] = [X] * theta.
  // u is a vector and sigmoid will be computed for each entry of u.
  StatusOr<SigmoidInput> GenerateSigmoidInput();

  // Functions to evaluate Sigmoid on the vector u (defined in applications/secure_sigmoid)
  // Invoke one by one at each party:
  // GenerateSigmoidRoundOneMessage
  // GenerateSigmoidRoundTwoMessage
  // GenerateSigmoidRoundThreeMessage
  // GenerateSigmoidRoundFourMessage
  // GenerateSigmoidResult

  // Since these functions are defined outside of this class, we need to be able to
  // access share_provider.
  // TODO create these functions also in this class and make share_provider private
  std::unique_ptr<LogRegDPShareProvider> share_provider_;

  // Generates the message and state needed to compute the matrix product
  // g = X.transpose() * d.
  // First, the function finishes computing the shared output of d = s - y.
  // After that, the state/message is a function of shares [X], [d] and the
  // preprocessed Beaver triple matrix.
  StatusOr<std::pair<StateXTransposeD, XTransposeDMessage>>
  GenerateXTransposeDMessage(
      SigmoidOutput sigmoid_output_share,
      StateMaskedXTranspose share_x_transpose_minus_a);

  // Computes:
  // [g] = [X.transpose] * [d] and then
  // scaled_g = (alpha / X.size) * g   (no new communication needed)
  // scaled_g_noise <-- scaled_g + alpha * noise
  // returns as messages share of scaled_g_noise
  StatusOr<std::pair<StateReconstructGradient, ReconstructGradientMessage>>
  GenerateReconstructGradientMessage(
      StateXTransposeD state_x_transpose_d,
      XTransposeDMessage x_transpose_d_message,
      MaskedXTransposeMessage x_transpose_minus_a_message);

  // Computes the gradient descent update.
  // Reconstructs revealed gradient
  // Then the weights theta are updated as:
  // public operation: theta(t+1) = theta(t) - alpha * revealed_gradient
  Status ComputeGradientUpdate(
      StateReconstructGradient state_reconstruct_gradient,
      ReconstructGradientMessage message_reconstruct_gradient);

  // Return the intermediate shares of logistic regression model.
  const std::vector<double>& GetTheta() const {
    return theta_;
  }

  // Test function to compute Sigmoid Outputs
  // This temporary function receives the input shares from each party and
  // outputs shares for BOTH parties.
  // The temporary function is insecure.
  StatusOr<std::pair<SigmoidOutput, SigmoidOutput>>
  GenerateSigmoidOutputForTesting(SigmoidInput sigmoid_input_share_p0,
                                  SigmoidInput sigmoid_input_share_p1);

 private:
  // Constructor: initialize the gradient descent with shares of training data
  // training parameters, and the share initialized theta.
  GradientDescentPartyOne(
      std::vector<uint64_t> share_x, std::vector<uint64_t> share_x_transpose,
      std::vector<uint64_t> share_y, std::vector<double> theta,
      std::unique_ptr<LogRegDPShareProvider> share_provider,
      std::unique_ptr<FixedPointElementFactory> fp_factory,
      const GradientDescentParams& param)
      : share_x_(std::move(share_x)),
        share_x_transpose_(std::move(share_x_transpose)),
        share_y_(std::move(share_y)),
        theta_(std::move(theta)),
        share_provider_(std::move(share_provider)),
        fp_factory_(std::move(fp_factory)),
        param_(param) {}

  // Stores the training data ([X], [Y]) and [X^T] for the gradient descent.
  std::vector<uint64_t> share_x_;
  std::vector<uint64_t> share_x_transpose_;
  std::vector<uint64_t> share_y_;

  // Theta stores the result of the regression.
  std::vector<double> theta_;

  // Stores the preprocessed data needed for gradient descent.
  // TODO Uncomment after you move sigmoid functions to this class
  // std::unique_ptr<LogRegDPShareProvider> share_provider_;

  // Stores the fixed point factory.
  std::unique_ptr<FixedPointElementFactory> fp_factory_;

  // The parameters used for gradient descent.
  const GradientDescentParams param_;
};

}  // namespace logistic_regression_dp
}  // namespace private_join_and_compute

#endif //GOOGLE_CODE_APPLICATIONS_LOGISTIC_REGRESSION_GRADIENT_DESCENT_DP_H_
