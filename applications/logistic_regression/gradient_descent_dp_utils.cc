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

#include "applications/logistic_regression/gradient_descent_dp_utils.h"

#include "absl/strings/str_cat.h"

namespace private_join_and_compute {
namespace logistic_regression_dp {

StatusOr<LogRegDPShareProvider> LogRegDPShareProvider::Create(
    std::vector<applications::SigmoidPrecomputedValue> sigmoid,
    std::vector<uint64_t> beaver_matrix_a,
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
    beaver_matrix_b_c,
    std::vector<std::vector<uint64_t>> noise,
    size_t num_examples, size_t num_features, size_t num_iterations) {
  if (num_iterations == 0) {
    return InvalidArgumentError(
    "Preprocessed data must be available for a "
    "positive number of iterations.");
    }
  // All share vectors must have the same number of entries.
  if (sigmoid.size() != num_iterations ||
    noise.size() != num_iterations ||
    beaver_matrix_b_c.size() != num_iterations) {
    return InvalidArgumentError("Invalid input size.");
  }
  // The shares and noise must have correct shape.
  if (beaver_matrix_a.size() != (num_examples * num_features)) {
    return InvalidArgumentError(
    "Invalid input size: a provided "
    "share has a wrong shape");
  }
  for (size_t idx = 0; idx < num_iterations; idx++) {
    if (noise[idx].size() != num_features ||
    beaver_matrix_b_c[idx].first.size() != num_examples ||
    beaver_matrix_b_c[idx].second.size() != num_features) {
      return InvalidArgumentError(
          absl::StrCat("Invalid input size: a provided "
                       "share has the wrong shape at index ",
                       idx));
    }
  }
  return LogRegDPShareProvider(
      std::move(sigmoid), std::move(beaver_matrix_a),
      std::move(beaver_matrix_b_c), std::move(noise), num_iterations);
}

StatusOr<LogRegDPShareProvider> LogRegDPShareProvider::CreateNewMic(
    std::vector<applications::SigmoidPrecomputedValueNewMic> sigmoid,
    std::vector<uint64_t> beaver_matrix_a,
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
    beaver_matrix_b_c,
    std::vector<std::vector<uint64_t>> noise,
    size_t num_examples, size_t num_features, size_t num_iterations) {
  if (num_iterations == 0) {
    return InvalidArgumentError(
        "Preprocessed data must be available for a "
        "positive number of iterations.");
  }
  // All share vectors must have the same number of entries.
  if (sigmoid.size() != num_iterations ||
      noise.size() != num_iterations ||
      beaver_matrix_b_c.size() != num_iterations) {
    return InvalidArgumentError("Invalid input size.");
  }
  // The shares and noise must have correct shape.
  if (beaver_matrix_a.size() != (num_examples * num_features)) {
    return InvalidArgumentError(
        "Invalid input size: a provided "
        "share has a wrong shape");
  }
  for (size_t idx = 0; idx < num_iterations; idx++) {
    if (noise[idx].size() != num_features ||
        beaver_matrix_b_c[idx].first.size() != num_examples ||
        beaver_matrix_b_c[idx].second.size() != num_features) {
      return InvalidArgumentError(
          absl::StrCat("Invalid input size: a provided "
                       "share has the wrong shape at index ",
                       idx));
    }
  }
  return LogRegDPShareProvider(
      std::move(sigmoid), std::move(beaver_matrix_a),
      std::move(beaver_matrix_b_c), std::move(noise), num_iterations);
}

StatusOr<applications::SigmoidPrecomputedValue>
LogRegDPShareProvider::GetSigmoidPrecomputedValue() const {
  if (counter_ == 0) {
    return InvalidArgumentError("Preprocess data is empty.");
  }
  return sigmoid_.back();
}

applications::SigmoidPrecomputedValueNewMic&
LogRegDPShareProvider::GetSigmoidPrecomputedValueNewMic() {
  return  sigmoid_new_mic_.back();
}

StatusOr<std::vector<uint64_t>>
LogRegDPShareProvider::GetXTransposeDMatrixA() const {
  return beaver_matrix_a_;
}

StatusOr<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
LogRegDPShareProvider::GetXTransposeDMatrixBandC() const {
  if (counter_ == 0) {
    return InvalidArgumentError("Preprocessed data is empty.");
  }
  return beaver_matrix_b_c_.back();
}

StatusOr<std::vector<uint64_t>>
LogRegDPShareProvider::GetNoise() const {
  if (counter_ == 0) {
    return InvalidArgumentError("Preprocessed data is empty.");
  }
  return noise_.back();
}

Status LogRegDPShareProvider::Shrink() {
  if (counter_ == 0) {
    return InvalidArgumentError("Preprocessed data is empty.");
  }
  counter_--;
  if (!sigmoid_.empty()) {
    sigmoid_.pop_back();
  } else {
    sigmoid_new_mic_.pop_back();
  }
  beaver_matrix_b_c_.pop_back();
  noise_.pop_back();
  return OkStatus();
}

}  // namespace logistic_regression_dp
}  // namespace private_join_and_compute

