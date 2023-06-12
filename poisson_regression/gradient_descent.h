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

#ifndef PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_GRADIENT_DESCENT_H_
#define PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_GRADIENT_DESCENT_H_

#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/gradient_descent_messages.pb.h"
#include "poisson_regression/gradient_descent_utils.h"
#include "poisson_regression/secure_exponentiation.h"

namespace private_join_and_compute {
namespace poisson_regression {
// Execute the gradient descent.
// The gradient descent is computed as:
// theta(t+1) = (1 - beta)*theta(t) +
//              alpha*Sum_i x_i*(y_i - delta_t_i*exp(<theta(t), x_i>).
// Notation:
// u_i <-- <theta(t), x_i>. u_i is the dot product between theta(t) and x_i.
// v_i <-- exp(u_i). v_i is the exponentiation of u_i.
// w_i <-- delta_t_i*v_i. * is the regular product.
// s_i <-- y_i - w_i.
// Z <-- Sum_i x_i*w_i = X*W. * represents matrix multiplication mod modulus.

// Struct to store gradient descent parameters.
struct GradientDescentParams {
  size_t num_features;          // Number of feature x_i.
  size_t feature_length;        // The number of fields each x_i has.
  size_t num_iterations;        // Number of iterations: num_iterations.
  uint8_t num_ring_bits;        // Ring size of the fixed point.
  uint8_t num_fractional_bits;  // Number of fractional bits in fixed point.
  uint64_t alpha;               // Learning rate: alpha.
  uint64_t one_minus_beta;      // Regularization term: beta.
  uint64_t modulus;             // Ring: modulus.
  uint64_t exponent_bound;      // Parameters for secure exponentiation.
  uint64_t prime_q;
};

class GradientDescentPartyZero {
 public:
  // Init the gradient descent with inputs and parameters.
  static StatusOr<GradientDescentPartyZero> Init(
      std::vector<uint64_t> share_x,
      std::vector<uint64_t> share_y,
      std::vector<uint64_t> share_delta_t,
      std::vector<uint64_t> share_theta,
      std::unique_ptr<ShareProvider> share_provider,
      const FixedPointElementFactory::Params& fpe_params,
      const ExponentiationParams& exp_params,
      const GradientDescentParams& param);

  // Functions to generate intermediate messages.

  // Generate the message needed to compute the matrix multiplication
  // U = X*Theta. The matrix multiplication is equivalent to the batched
  // dot products u_i = <theta, x_i>. The message contains shares
  // [X - A] and [Theta - B].
  StatusOr<std::pair<StateRound1, GradientDescentRoundOneMessage>>
  GenerateGradientDescentRoundOneMessage();

  // Generate the message needed to compute the exponentiation of the shares
  // [u_i]. First, the function computes the shared output of the matrix
  // multiplication [U] = [X*Theta] from the previous step. Then, the message is
  // a function of share [u_i] and the pre-processed mult_to_add_share.
  // The secure exponentiation is done in batches.
  StatusOr<std::pair<StateRound2, GradientDescentPartyZeroRoundTwoMessage>>
  GenerateGradientDescentRoundTwoMessage(
      StateRound1 state_one,
      GradientDescentRoundOneMessage other_party_message);

  // Generate the message needed to compute the pointwise multiplication
  // [w_i] = [delta_t_i*v_i]. First, the function computes the shared output of
  // the exponentiation gate [v_i] = exp([u_i]) from the previous step.
  // After that, the message is a function of share [v_i] and the pre-processed
  // Beaver triple vector.
  StatusOr<std::pair<StateRound3, GradientDescentRoundThreeMessage>>
  GenerateGradientDescentRoundThreeMessage(
      StateRound2 state_two,
      GradientDescentPartyOneRoundTwoMessage other_party_message);

  // Generate the message needed to compute the matrix multiplication
  // Z = (Y - W)*X = S*X.  First, the function computes the shared output of
  // the batched multiplication gate [w_i] = [delta_t_i*v_i] from the previous
  // step and followed by S = Y - W.
  // After that, the message is a function of share [S], [X], and the
  // pre-processed Beaver triple matrix.
  StatusOr<std::pair<StateRound4, GradientDescentRoundFourMessage>>
  GenerateGradientDescentRoundFourMessage(
      StateRound3 state_three,
      GradientDescentRoundThreeMessage other_party_message);

  // Function to compute the gradient descent update.
  // Upon receiving the message from the previous step, the shared output
  // [Z] = [S*X] can be computed. Then the weights Theta can be updated as:
  // theta(t+1) = (1 - beta)*theta(t) + alpha*Z.
  // alpha is the learning rate, beta is the regularization term.
  Status ComputeGradientUpdate(
      StateRound4 state_four,
      GradientDescentRoundFourMessage other_party_message);

  // Return the intermediate shares of poisson regression model.
  const std::vector<uint64_t>& GetTheta() const {
    return share_theta_;
  }

 private:
  // Constructor: initialize the gradient descent with shares of training data
  // training parameters, and the share initialized theta.
  GradientDescentPartyZero(
      std::vector<uint64_t> share_x, std::vector<uint64_t> share_y,
      std::vector<uint64_t> share_delta_t, std::vector<uint64_t> share_theta,
      std::unique_ptr<ShareProvider> share_provider,
      std::unique_ptr<FixedPointElementFactory> fp_factory,
      std::unique_ptr<SecureExponentiationPartyZero> secure_exp,
      const GradientDescentParams& param)
      : share_x_(std::move(share_x)),
        share_y_(std::move(share_y)),
        share_delta_t_(std::move(share_delta_t)),
        share_theta_(std::move(share_theta)),
        share_provider_(std::move(share_provider)),
        fp_factory_(std::move(fp_factory)),
        secure_exp_(std::move(secure_exp)),
        param_(param) {}

  // Store the training data ([X], [Y], [Delta_t]).
  std::vector<uint64_t> share_x_;
  std::vector<uint64_t> share_y_;
  std::vector<uint64_t> share_delta_t_;

  // Theta store the result of the regression.
  std::vector<uint64_t> share_theta_;

  // Store the preprocessed data needed for gradent descent.
  std::unique_ptr<ShareProvider> share_provider_;

  // Store the fixed point factory.
  std::unique_ptr<FixedPointElementFactory> fp_factory_;

  // Store the SecureExponentiationPartyZero.
  std::unique_ptr<SecureExponentiationPartyZero> secure_exp_;

  // The parameters used for gradient descent.
  const GradientDescentParams param_;
};

class GradientDescentPartyOne {
 public:
  // Init the gradient descent with inputs and parameters.
  static StatusOr<GradientDescentPartyOne> Init(
      std::vector<uint64_t> share_x,
      std::vector<uint64_t> share_y,
      std::vector<uint64_t> share_delta_t,
      std::vector<uint64_t> share_theta,
      std::unique_ptr<ShareProvider> share_provider,
      const FixedPointElementFactory::Params& fpe_params,
      const ExponentiationParams& exp_params,
      const GradientDescentParams& param);

  // Functions to generate intermediate messages.

  // Generate the message needed to compute the matrix multiplication
  // U = X*Theta. The matrix multiplication is equivalent to the batched
  // dot products u_i = <theta, x_i>. The message contains shares
  // [X - A] and [Theta - B].
  StatusOr<std::pair<StateRound1, GradientDescentRoundOneMessage>>
  GenerateGradientDescentRoundOneMessage();

  // Generate the message needed to compute the exponentiation of the shares
  // [u_i]. First, the function computes the shared output of the matrix
  // multiplication [U] = [X*Theta] from the previous step. Then, the message is
  // a function of share [u_i] and the pre-processed mult_to_add_share.
  // The secure exponentiation is done in batches.
  StatusOr<std::pair<StateRound2, GradientDescentPartyOneRoundTwoMessage>>
  GenerateGradientDescentRoundTwoMessage(
      StateRound1 state_one,
      GradientDescentRoundOneMessage other_party_message);

  // Generate the message needed to compute the pointwise multiplication
  // [w_i] = [delta_t_i*v_i]. First, the function computes the shared output of
  // the exponentiation gate [v_i] = exp([u_i]) from the previous step.
  // After that, the message is a function of share [v_i] and the pre-processed
  // Beaver triple vector.
  StatusOr<std::pair<StateRound3, GradientDescentRoundThreeMessage>>
  GenerateGradientDescentRoundThreeMessage(
      StateRound2 state_two,
      GradientDescentPartyZeroRoundTwoMessage other_party_message);

  // Generate the message needed to compute the matrix multiplication
  // Z = (Y - W)*X = S*X.  First, the function computes the shared output of
  // the batched multiplication gate [w_i] = [delta_t_i*v_i] from the previous
  // step and followed by S = Y - W.
  // After that, the message is a function of share [S], [X], and the
  // pre-processed Beaver triple matrix.
  StatusOr<std::pair<StateRound4, GradientDescentRoundFourMessage>>
  GenerateGradientDescentRoundFourMessage(
      StateRound3 state_three,
      GradientDescentRoundThreeMessage other_party_message);

  // Function to compute the gradient descent update.
  // Upon receiving the message from the previous step, the shared output
  // [Z] = [S*X] can be computed. Then the weights Theta can be updated as:
  // theta(t+1) = (1 - beta)*theta(t) + alpha*Z.
  // alpha is the learning rate, beta is the regularization term.
  Status ComputeGradientUpdate(
      StateRound4 state_four,
      GradientDescentRoundFourMessage other_party_message);

  // Return the intermediate shares of poisson regression model.
  const std::vector<uint64_t>& GetTheta() const {
    return share_theta_;
  }

 private:
  // Constructor: initialize the gradient descent with shares of training data
  // training parameters, and the share initialized theta.
  GradientDescentPartyOne(
      std::vector<uint64_t> share_x, std::vector<uint64_t> share_y,
      std::vector<uint64_t> share_delta_t, std::vector<uint64_t> share_theta,
      std::unique_ptr<ShareProvider> share_provider,
      std::unique_ptr<FixedPointElementFactory> fp_factory,
      std::unique_ptr<SecureExponentiationPartyOne> secure_exp,
      const GradientDescentParams& param)
      : share_x_(std::move(share_x)),
        share_y_(std::move(share_y)),
        share_delta_t_(std::move(share_delta_t)),
        share_theta_(std::move(share_theta)),
        share_provider_(std::move(share_provider)),
        fp_factory_(std::move(fp_factory)),
        secure_exp_(std::move(secure_exp)),
        param_(param) {}

  // Store the training data ([X], [Y], [Delta_t]).
  std::vector<uint64_t> share_x_;
  std::vector<uint64_t> share_y_;
  std::vector<uint64_t> share_delta_t_;

  // Theta store the intermediate model parameters of the poisson regression.
  std::vector<uint64_t> share_theta_;

  // Store the preprocessed data needed for gradent descent.
  std::unique_ptr<ShareProvider> share_provider_;

  // Store the fixed point factory.
  std::unique_ptr<FixedPointElementFactory> fp_factory_;

  // Store the SecureExponentiationPartyZero.
  std::unique_ptr<SecureExponentiationPartyOne> secure_exp_;

  // The parameters used for gradient descent.
  const GradientDescentParams param_;
};

}  // namespace poisson_regression
}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_GRADIENT_DESCENT_H_
