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

#ifndef PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_GRADIENT_DESCENT_UTILS_H_
#define PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_GRADIENT_DESCENT_UTILS_H_

#include "poisson_regression/beaver_triple.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "private_join_and_compute/util/status.inc"

namespace private_join_and_compute {
namespace poisson_regression {

// The class ShareProvider acts as a manager of all the preprocessed shares:
// Beaver triple shares (for secure batched pointwise multiplications and
// secure matrix multiplications) and MultToAdd shares (for secure
// exponentiations).
class ShareProvider {
 public:
  // Store the preprocessed data. Verify that the input is valid.
  static StatusOr<ShareProvider> Create(
      std::vector<BeaverTripleMatrix<uint64_t>> beaver_matrix_round_1,
      std::vector<MultToAddShare> mult_to_add_share_round_2,
      std::vector<BeaverTripleVector<uint64_t>> beaver_vector_round_3,
      std::vector<BeaverTripleMatrix<uint64_t>> beaver_matrix_round_4,
      size_t num_features, size_t feature_length, size_t num_iterations);

  // Function to access preprocessed data needed for one round.
  // Return the next piece of preprocessed data needed for one round.
  // Return INVALID_ARGUMENT if there is no more shares available.
  StatusOr<BeaverTripleMatrix<uint64_t>> GetRoundOneBeaverMatrix() const;
  StatusOr<MultToAddShare> GetRoundTwoMultToAdd() const;
  StatusOr<BeaverTripleVector<uint64_t>> GetRoundThreeBeaverVector() const;
  StatusOr<BeaverTripleMatrix<uint64_t>> GetRoundFourBeaverMatrix() const;

  // After one round, the consumed preprocessed data is deleted.
  Status Shrink();

 private:
  // Constructor.
  ShareProvider(std::vector<BeaverTripleMatrix<uint64_t>> beaver_matrix_round_1,
                std::vector<MultToAddShare> mult_to_add_share_round_2,
                std::vector<BeaverTripleVector<uint64_t>> beaver_vector_round_3,
                std::vector<BeaverTripleMatrix<uint64_t>> beaver_matrix_round_4,
                size_t num_iterations)
      : beaver_matrix_round_1_(std::move(beaver_matrix_round_1)),
        mult_to_add_share_round_2_(std::move(mult_to_add_share_round_2)),
        beaver_vector_round_3_(std::move(beaver_vector_round_3)),
        beaver_matrix_round_4_(std::move(beaver_matrix_round_4)),
        counter_(num_iterations) {}

  // Store preprocessed data for each round of the gradient descent.
  std::vector<BeaverTripleMatrix<uint64_t>> beaver_matrix_round_1_;
  std::vector<MultToAddShare> mult_to_add_share_round_2_;
  std::vector<BeaverTripleVector<uint64_t>> beaver_vector_round_3_;
  std::vector<BeaverTripleMatrix<uint64_t>> beaver_matrix_round_4_;

  // This counter keeps track of the available resources.
  // It is set to the actual number of rounds the preprocessed data can provide.
  int counter_;
};

// Store the internal states used for generating the messages and to compute
// and store intermediate results.

// Store StateRound1 after the first round.
// Store shares [X-A] and [Theta-B]. These shares are used in the computation
// of secure matrix multiplication [X*Theta] in the next round.
// ([A], [B]) is part of the Beaver triple matrix ([A], [B], [C = A*B]).
struct StateRound1 {
  std::vector<uint64_t> share_x_minus_a;
  std::vector<uint64_t> share_theta_minus_b;
};

// Store StateRound2 after the second round.
// During the second round, the StateRound1 and the other party's round 1
// message are used to compute share [U] = [X*Theta] where * represents matrix
// multiplication mod modulus. Share [u_i] together with preprocessed data is
// used to generate the second message that will be used to compute the
// secure exponentiation [v_i] = exp([u_i]).
struct StateRound2 {
  std::vector<uint64_t> share_mult;
};

// Store StateRound3 after the third round.
// During the third round, the StateRound2 and the other party's round 2
// message are used to compute share [v_i] = exp([u_i]) where u_i is the dot
// product between x_i and Theta. Share [v_i] together with preprocessed data is
// used to generate the third message that will be used to compute
// [W] = [Delta_t*V] where * represent pointwise multiplication mod modulus.
// ([A], [B]) is part of the Beaver triple vector
// ([A_i], [B_i], [C_i = A_i*B_i]).
struct StateRound3 {
  std::vector<uint64_t> share_delta_t_minus_a;
  std::vector<uint64_t> share_v_minus_b;
};

// Store StateRound4 after the forth round.
// During the forth round, the StateRound3 and the other party's round 3
// message are used to compute share [W] = [Delta_t*V] where W is the pointwise
// multiplication between Delta_t and V mod modulus. Share [W] together with
// the preprocessed data is used to generate the forth message that will be used
// to compute [Z] = [X*S] where * represents matrix multiplication mod modulus
// and [S] = [Y - W].
// ([A], [B]) is part of the Beaver triple matrix ([A], [B], [C = A*B]).
struct StateRound4 {
  std::vector<uint64_t> share_s_minus_a;
  std::vector<uint64_t> share_x_minus_b;
};

}  // namespace poisson_regression
}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_GRADIENT_DESCENT_UTILS_H_
