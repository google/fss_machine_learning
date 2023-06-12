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

#ifndef PRIVATE_JOIN_AND_COMPUTE_APPLICATIONS_LOGISTIC_REGRESSION_GRADIENT_DESCENT_H_
#define PRIVATE_JOIN_AND_COMPUTE_APPLICATIONS_LOGISTIC_REGRESSION_GRADIENT_DESCENT_H_

#include "applications/logistic_regression/gradient_descent_messages.pb.h"
#include "applications/logistic_regression/gradient_descent_utils.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/secure_exponentiation.h"

namespace private_join_and_compute {
namespace logistic_regression {
// Execute the gradient descent.
// The gradient descent is computed as:
// theta(t+1) = theta(t) -
//              [(alpha/ X.size) * X.transpose * (sigmoid(X * theta(t)) - y) + (alpha * lambda) / X.size * theta(t)]
// Notation:
//
// alpha and X.size are public scalars, the rest is private & secret shared
// X is n x k matrix (k already includes the intercept)
// y is n x 1 vector
// theta is k x 1 vector
//
// u <-- <X, theta(t)>. u is the matrix_product between X and theta(t).
// u is a n x 1 vector
// s <-- sigmoid(u). s is the element-wise sigmoid of each element in vector u.
// s is a n x 1 vector
// d <-- s - y. - is vector_subtraction.
// d is a n x 1 vector
// g <-- <X.transpose, d>, g is the matrix_vector product between X.transpose
// and d.
// g is k x 1 vector
// scaled_g <-- (alpha/ X.size) * g where * represents scalar_vector_product.
// scaled_regularization <-- (alpha * lambda / X.size()) * theta(t)
// scaled_regularization is kx1 vector
// scaled_g_regularized <-- scaled_g + scaled_regularization
// theta(t+1) <-- theta(t) - scaled_g_regularized. - is vector_subtraction.

// Struct to store gradient descent parameters.
struct GradientDescentParams {
  size_t num_examples;          // Num of training examples in X.
  size_t num_features;          // Num of features of X (including intercept).
  size_t num_iterations;        // Number of iterations: num_iterations.
  uint8_t num_ring_bits;        // Ring size of the fixed point (l).
  uint8_t num_fractional_bits;  // Num of fractional bits in fixed point (lf).
  double alpha;                 // Learning rate: alpha.
  double lambda;                // Regularization parameter.
  uint64_t modulus;             // Ring: modulus.
};

class GradientDescentPartyZero {
 public:
  // Init the gradient descent with inputs and parameters.
  static StatusOr<GradientDescentPartyZero> Init(
      std::vector<uint64_t> share_x,
      std::vector<uint64_t> share_y,
      std::vector<uint64_t> share_theta,
      std::unique_ptr<LogRegShareProvider> share_provider,
      const FixedPointElementFactory::Params& fpe_params,
      const GradientDescentParams& param);

  // Functions to generate messages for correlated beaver triple products

  // Generates the message and state associated with masking X
  // The message contains shares [X - A] and can be reused for all training
  // steps
  StatusOr<std::pair<StateMaskedX, MaskedXMessage>>
  GenerateCorrelatedProductMessageForX();

  // Generates the message and state associated with masking X^T
  // The message contains shares [X^T - A] and can be reused for all training
  // steps
  StatusOr<std::pair<StateMaskedXTranspose, MaskedXTransposeMessage>>
  GenerateCorrelatedProductMessageForXTranspose();

  // Functions to generate intermediate messages during the gradient descent
  // iterations.

  // Generates the Theta - B message needed to finish the matrix product
  // U = X * Theta. The message contains shares [Theta - B].
  StatusOr<std::pair<StateRound1, RoundOneGradientDescentMessage>>
  GenerateGradientDescentRoundOneMessage(StateMaskedX share_x_minus_a);

  // Generates the sigmoid inputs by computing the shared output of the matrix
  // product [u] = [X * theta].
  // u is a vector and sigmoid will be computed for each entry of u.
  StatusOr<SigmoidInput> GenerateSigmoidInput(
      StateRound1 state_one, MaskedXMessage other_party_x_minus_a_message,
      RoundOneGradientDescentMessage other_party_theta_minus_b_message);

  StatusOr<SigmoidInput> GenerateSigmoidInputMinibatch(
      StateRound1 state_one, MaskedXMessage other_party_x_minus_a_message,
      RoundOneGradientDescentMessage other_party_theta_minus_b_message,
      size_t batch_size, size_t idx_batch, size_t size_per_minibatch);

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
  // TODO Create these functions also in this class and make share_provider private
  std::unique_ptr<LogRegShareProvider> share_provider_;

  // Generates the message and state needed to compute the matrix product
  // g = X.transpose() * d.
  // First, the function finishes computing the shared output of d = s - y.
  // After that, the state/message is a function of shares [X], [d] and the
  // preprocessed Beaver triple matrix.
  StatusOr<std::pair<StateRound3, RoundThreeGradientDescentMessage>>
  GenerateGradientDescentRoundThreeMessage(
      SigmoidOutput sigmoid_output_share,
      StateMaskedXTranspose share_x_transpose_minus_a);

  StatusOr<std::pair<StateRound3, RoundThreeGradientDescentMessage>>
  GenerateGradientDescentRoundThreeMessageMinibatch(
      SigmoidOutput sigmoid_output_share,
      StateMaskedXTranspose share_x_transpose_minus_a,
      size_t batch_size, size_t idx_batch);

  // Computes the gradient descent update.
  // Compute:
  // [g] = [X.transpose] * [d] and then
  // scaled_g = (alpha/ X.size) * g   (no new communication needed)
  // Then the weights theta are updated as:
  // theta(t+1) = theta(t) - scaled_g.
  // alpha is the learning rate.
  Status ComputeGradientUpdate(
      StateRound3 state_three,
      MaskedXTransposeMessage other_party_x_transpose_minus_a_message,
      RoundThreeGradientDescentMessage other_party_d_minus_b_message);

  Status ComputeGradientUpdateMinibatch(
      StateRound3 state_three,
      MaskedXTransposeMessage other_party_x_transpose_minus_a_message,
      RoundThreeGradientDescentMessage other_party_d_minus_b_message,
      size_t batch_size, size_t idx_batch, size_t size_per_minibatch);

  // Return the intermediate shares of logistic regression model.
  const std::vector<uint64_t>& GetTheta() const {
    return share_theta_;
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
      std::vector<uint64_t> share_y, std::vector<uint64_t> share_theta,
      std::unique_ptr<LogRegShareProvider> share_provider,
      std::unique_ptr<FixedPointElementFactory> fp_factory,
      const GradientDescentParams& param)
      : share_x_(std::move(share_x)),
        share_x_transpose_(std::move(share_x_transpose)),
        share_y_(std::move(share_y)),
        share_theta_(std::move(share_theta)),
        share_provider_(std::move(share_provider)),
        fp_factory_(std::move(fp_factory)),
        param_(param) {}

  // Stores the training data ([X], [Y]) and [X^T] for the gradient descent.
  std::vector<uint64_t> share_x_;
  std::vector<uint64_t> share_x_transpose_;
  std::vector<uint64_t> share_y_;

  // Theta stores the result of the regression.
  std::vector<uint64_t> share_theta_;

  // Stores the preprocessed data needed for gradient descent.
  // TODO Uncomment after you move sigmoid functions to this class
  // std::unique_ptr<LogRegShareProvider> share_provider_;

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
      std::vector<uint64_t> share_theta,
      std::unique_ptr<LogRegShareProvider> share_provider,
      const FixedPointElementFactory::Params& fpe_params,
      const GradientDescentParams& param);

  // Functions to generate messages for correlated beaver triple products

  // Generates the message and state associated with masking X.
  // The message contains shares [X - A] and can be reused for all training
  // steps.
  StatusOr<std::pair<StateMaskedX, MaskedXMessage>>
  GenerateCorrelatedProductMessageForX();

  // Generates the message and state associated with masking X^T
  // The message contains shares [X^T - A] and can be reused for all training
  // steps.
  StatusOr<std::pair<StateMaskedXTranspose, MaskedXTransposeMessage>>
  GenerateCorrelatedProductMessageForXTranspose();

  // Functions to generate intermediate messages during the gradient descent
  // iterations.

  // Generates the Theta - B message needed to finish the matrix product
  // U = X * Theta. The message contains shares [Theta - B].
  StatusOr<std::pair<StateRound1, RoundOneGradientDescentMessage>>
  GenerateGradientDescentRoundOneMessage(StateMaskedX share_x_minus_a);

  // Generates the sigmoid inputs by computing the shared output of the matrix
  // product [u] = [X*theta].
  // u is a vector and sigmoid will be computed for each entry of u.
  StatusOr<SigmoidInput> GenerateSigmoidInput(
      StateRound1 state_one, MaskedXMessage other_party_x_minus_a_message,
      RoundOneGradientDescentMessage other_party_theta_minus_b_message);

  StatusOr<SigmoidInput> GenerateSigmoidInputMinibatch(
      StateRound1 state_one, MaskedXMessage other_party_x_minus_a_message,
      RoundOneGradientDescentMessage other_party_theta_minus_b_message,
      size_t batch_size, size_t idx_batch, size_t size_per_minibatch);

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
  std::unique_ptr<LogRegShareProvider> share_provider_;

  // Generates the message and state needed to compute the matrix product
  // g = X.transpose() * d.
  // First, the function finishes computing the shared output of d = s - y.
  // After that, the state/message is a function of shares [X], [d] and the
  // preprocessed Beaver triple matrix.
  StatusOr<std::pair<StateRound3, RoundThreeGradientDescentMessage>>
  GenerateGradientDescentRoundThreeMessage(
      SigmoidOutput sigmoid_output_share,
      StateMaskedXTranspose share_x_transpose_minus_a);

  StatusOr<std::pair<StateRound3, RoundThreeGradientDescentMessage>>
  GenerateGradientDescentRoundThreeMessageMinibatch(
      SigmoidOutput sigmoid_output_share,
      StateMaskedXTranspose share_x_transpose_minus_a,
      size_t batch_size, size_t idx_batch);

  // Computes the gradient descent update.
  // Compute:
  // [g] = [X.transpose] * [d] and then
  // scaled_g = (alpha/ X.size) * g   (no new communication needed)
  // Then the weights theta are updated as:
  // theta(t+1) = theta(t) - scaled_g.
  // alpha is the learning rate.
  Status ComputeGradientUpdate(
      StateRound3 state_three,
      MaskedXTransposeMessage other_party_x_transpose_minus_a_message,
      RoundThreeGradientDescentMessage other_party_d_minus_b_message);

  Status ComputeGradientUpdateMinibatch(
      StateRound3 state_three,
      MaskedXTransposeMessage other_party_x_transpose_minus_a_message,
      RoundThreeGradientDescentMessage other_party_d_minus_b_message,
      size_t batch_size, size_t idx_batch, size_t size_per_minibatch);

  // Return the intermediate shares of logistic regression model.
  const std::vector<uint64_t>& GetTheta() const {
    return share_theta_;
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
      std::vector<uint64_t> share_y, std::vector<uint64_t> share_theta,
      std::unique_ptr<LogRegShareProvider> share_provider,
      std::unique_ptr<FixedPointElementFactory> fp_factory,
      const GradientDescentParams& param)
      : share_x_(std::move(share_x)),
        share_x_transpose_(std::move(share_x_transpose)),
        share_y_(std::move(share_y)),
        share_theta_(std::move(share_theta)),
        share_provider_(std::move(share_provider)),
        fp_factory_(std::move(fp_factory)),
        param_(param) {}

  // Stores the training data ([X], [Y]) and [X^T] for the gradient descent.
  std::vector<uint64_t> share_x_;
  std::vector<uint64_t> share_x_transpose_;
  std::vector<uint64_t> share_y_;

  // Theta stores the result of the regression.
  std::vector<uint64_t> share_theta_;

  // Stores the preprocessed data needed for gradient descent.
  // TODO Uncomment after you move sigmoid functions to this class
  // std::unique_ptr<LogRegShareProvider> share_provider_;

  // Stores the fixed point factory.
  std::unique_ptr<FixedPointElementFactory> fp_factory_;

  // The parameters used for gradient descent.
  const GradientDescentParams param_;
};

}  // namespace logistic_regression
}  // namespace private_join_and_compute


#endif  // PRIVATE_JOIN_AND_COMPUTE_APPLICATIONS_LOGISTIC_REGRESSION_GRADIENT_DESCENT_H_
