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

#ifndef GOOGLE_CODE_APPLICATIONS_LOGISTIC_REGRESSION_GRADIENT_DESCENT_DP_UTILS_H_
#define GOOGLE_CODE_APPLICATIONS_LOGISTIC_REGRESSION_GRADIENT_DESCENT_DP_UTILS_H_

#include "applications/secure_sigmoid/secure_sigmoid.h"
#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic.h"
#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "private_join_and_compute/util/status.inc"

namespace private_join_and_compute {
namespace logistic_regression_dp {

// The structs below store the internal states used for generating the
// messages for correlated beaver triple products

// StateMaskedXTranspose stores shares [X^T-A].
// These shares are used in the computation
// of secure correlated matrix product [X^T * d].
// [A] is part of the Beaver triple matrix ([A], [B], [C = A*B]).
struct StateMaskedXTranspose {
  std::vector<uint64_t> share_x_transpose_minus_a;
};

// The structs below store the internal states used for generating the messages
// and for doing intermediate computations during the rounds of the gradient
// descent.

// SigmoidInput stores sigmoid inputs i.e. [U] = [X * Theta].
struct SigmoidInput {
  std::vector<uint64_t> sigmoid_input;
};

// SigmoidOutput stores the output of the sigmoid invocations (s = sigmoid([U]).
struct SigmoidOutput {
  std::vector<uint64_t> sigmoid_output;
};

// StateXTransposeD stores [X.transpose - a] and [d] = [s - y].
// LogReg:
// Share [X.transpose], [d] together with the preprocessed data [A], [B],
// are used to generate the third message that will be used to compute
// [g] = [X.transpose * d] where * represents matrix multiplication mod modulus
// ([A], [B]) is part of the Beaver triple matrix ([A], [B], [C = A*B]).
struct StateXTransposeD {
  std::vector<uint64_t> share_x_transpose_minus_a;
  std::vector<uint64_t> share_d_minus_b;
};

// StateReconstructGradient stores alpha_scaled_g_noise_regularized (see description in gradient_descent_dp.h)
struct StateReconstructGradient {
  std::vector<uint64_t> share_scaled_g_noise_regularized;
};

// The class LogRegShareProvider acts as a manager of all the preprocessed
// shares: Beaver triple matrix shares (for matrix product) SigmoidShareProvider
// (for sigmoid computation)
class LogRegDPShareProvider {
 public:
  // Stores the preprocessed data. Verify that the input is valid.
  static StatusOr<LogRegDPShareProvider> Create(
  // sigmoid preprocessing (FSS, powers, exp, etc.)
  // we need sigmoid preprocessing per iteration
  std::vector<applications::SigmoidPrecomputedValue> sigmoid,
  // For each iteration, there is only one mask A for X^T
  std::vector<uint64_t> beaver_matrix_a,
  // For each Theta_i, there is a new B, C
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>> beaver_matrix_b_c,
  std::vector<std::vector<uint64_t>> noise,
  size_t num_examples, size_t num_features, size_t num_iterations);

  static StatusOr<LogRegDPShareProvider> CreateNewMic(
      // sigmoid preprocessing (FSS, powers, exp, etc.)
      // we need sigmoid preprocessing per iteration
      std::vector<applications::SigmoidPrecomputedValueNewMic> sigmoid,
      // For each iteration, there is only one mask A for X^T
      std::vector<uint64_t> beaver_matrix_a,
      // For each Theta_i, there is a new B, C
      std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>> beaver_matrix_b_c,
      std::vector<std::vector<uint64_t>> noise,
      size_t num_examples, size_t num_features, size_t num_iterations);

  // Function to access preprocessed data needed for one round.
  // Returns the next piece of preprocessed data needed for one round.
  // Returns INVALID_ARGUMENT if there are no more shares available.
  StatusOr<applications::SigmoidPrecomputedValue> GetSigmoidPrecomputedValue() const;
  applications::SigmoidPrecomputedValueNewMic& GetSigmoidPrecomputedValueNewMic();
  StatusOr<std::vector<uint64_t>> GetXTransposeDMatrixA() const;
  StatusOr<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>> GetXTransposeDMatrixBandC() const;
  StatusOr<std::vector<uint64_t>> GetNoise() const;

  // After one round, the consumed preprocessed data is deleted.
  Status Shrink();

 private:
  // Constructor.
  LogRegDPShareProvider(
      std::vector<applications::SigmoidPrecomputedValue> sigmoid,
      std::vector<uint64_t> beaver_matrix_a,
      std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>> beaver_matrix_b_c,
      std::vector<std::vector<uint64_t>> noise,
      size_t num_iterations) :
        sigmoid_(std::move(sigmoid)),
        beaver_matrix_a_(std::move(beaver_matrix_a)),
        beaver_matrix_b_c_(std::move(beaver_matrix_b_c)),
        noise_(std::move(noise)),
        counter_(num_iterations) {}

  LogRegDPShareProvider(
      std::vector<applications::SigmoidPrecomputedValueNewMic> sigmoid,
      std::vector<uint64_t> beaver_matrix_a,
      std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>> beaver_matrix_b_c,
      std::vector<std::vector<uint64_t>> noise,
      size_t num_iterations) :
      sigmoid_new_mic_(std::move(sigmoid)),
      beaver_matrix_a_(std::move(beaver_matrix_a)),
      beaver_matrix_b_c_(std::move(beaver_matrix_b_c)),
      noise_(std::move(noise)),
      counter_(num_iterations) {}

  // Store preprocessed data for each round of the gradient descent.
  std::vector<applications::SigmoidPrecomputedValue> sigmoid_;
  std::vector<applications::SigmoidPrecomputedValueNewMic> sigmoid_new_mic_;
  std::vector<uint64_t> beaver_matrix_a_;
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
      beaver_matrix_b_c_;
  std::vector<std::vector<uint64_t>> noise_;
  // This counter keeps track of the available resources.
  // It is set to the actual number of rounds the preprocessed data can support.
  int counter_;
};

}  // namespace logistic_regression_dp
}  // namespace private_join_and_compute


#endif //GOOGLE_CODE_APPLICATIONS_LOGISTIC_REGRESSION_GRADIENT_DESCENT_DP_UTILS_H_
