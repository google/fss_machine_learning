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

#include "applications/logistic_regression/gradient_descent_utils.h"

#include "absl/strings/str_cat.h"

namespace private_join_and_compute {
namespace logistic_regression {

StatusOr<LogRegShareProvider> LogRegShareProvider::Create(
    std::vector<uint64_t> beaver_matrix_a_round_1,
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
        beaver_matrix_b_c_round_1,
    std::vector<applications::SigmoidPrecomputedValue> sigmoid,
    std::vector<uint64_t> beaver_matrix_a_round_3,
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
        beaver_matrix_b_c_round_3,
    size_t num_examples, size_t num_features, size_t num_iterations) {
  if (num_iterations == 0) {
    return InvalidArgumentError(
        "Preprocessed data must be available for a "
        "positive number of iterations.");
  }
  // All share vectors must have the same number of entries.
  // Commenting as this is not true for minibatch
  /*
  if (beaver_matrix_b_c_round_1.size() != num_iterations ||
      sigmoid.size() != num_iterations ||
      beaver_matrix_b_c_round_3.size() != num_iterations) {
    return InvalidArgumentError("Invalid input size.");
  }*/
  // The shares must have correct shape.
  if (beaver_matrix_a_round_1.size() != (num_examples * num_features) ||
      beaver_matrix_a_round_3.size() != (num_examples * num_features)) {
    return InvalidArgumentError(
        "Invalid input size: a provided "
        "share has a wrong shape");
  }
  // Commenting as this is not true for minibatch
  /*
  for (size_t idx = 0; idx < num_iterations; idx++) {
    if (beaver_matrix_b_c_round_1[idx].first.size() != num_features ||   // B
        beaver_matrix_b_c_round_1[idx].second.size() != num_examples ||  // C
        beaver_matrix_b_c_round_3[idx].first.size() != num_examples ||
        beaver_matrix_b_c_round_3[idx].second.size() != num_features) {
      return InvalidArgumentError(
          absl::StrCat("Invalid input size: a provided "
                       "share has the wrong shape at index ",
                       idx));
    }
  }*/
  return LogRegShareProvider(
      std::move(beaver_matrix_a_round_1), std::move(beaver_matrix_b_c_round_1),
      std::move(sigmoid), std::move(beaver_matrix_a_round_3),
      std::move(beaver_matrix_b_c_round_3), num_iterations);
}

StatusOr<LogRegShareProvider> LogRegShareProvider::CreateNewMic(
    std::vector<uint64_t> beaver_matrix_a_round_1,
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
    beaver_matrix_b_c_round_1,
    std::vector<applications::SigmoidPrecomputedValueNewMic> sigmoid,
    std::vector<uint64_t> beaver_matrix_a_round_3,
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
    beaver_matrix_b_c_round_3,
    size_t num_examples, size_t num_features, size_t num_iterations) {
  if (num_iterations == 0) {
    return InvalidArgumentError(
        "Preprocessed data must be available for a "
        "positive number of iterations.");
  }
  // All share vectors must have the same number of entries.
  // Commenting out as different for minibatch
  //if (beaver_matrix_b_c_round_1.size() != num_iterations ||
  //    sigmoid.size() != num_iterations ||
  //    beaver_matrix_b_c_round_3.size() != num_iterations) {
  //  return InvalidArgumentError("Invalid input size.");
  //}
  // The shares must have correct shape.
  if (beaver_matrix_a_round_1.size() != (num_examples * num_features) ||
      beaver_matrix_a_round_3.size() != (num_examples * num_features)) {
    return InvalidArgumentError(
        "Invalid input size: a provided "
        "share has a wrong shape");
  }
  // Commenting out as different for minibatch
  //for (size_t idx = 0; idx < num_iterations; idx++) {
  //  if (beaver_matrix_b_c_round_1[idx].first.size() != num_features ||   // B
  //      beaver_matrix_b_c_round_1[idx].second.size() != num_examples ||  // C
  //      beaver_matrix_b_c_round_3[idx].first.size() != num_examples ||
  //      beaver_matrix_b_c_round_3[idx].second.size() != num_features) {
  //    return InvalidArgumentError(
  //       absl::StrCat("Invalid input size: a provided "
  //                     "share has the wrong shape at index ",
  //                     idx));
  //  }
  //}
  return LogRegShareProvider(
      std::move(beaver_matrix_a_round_1), std::move(beaver_matrix_b_c_round_1),
      std::move(sigmoid), std::move(beaver_matrix_a_round_3),
      std::move(beaver_matrix_b_c_round_3), num_iterations);
}

StatusOr<std::vector<uint64_t>> LogRegShareProvider::GetRoundOneBeaverMatrixA()
    const {
  return beaver_matrix_a_round_1_;
}

StatusOr<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
LogRegShareProvider::GetRoundOneBeaverMatrixBandC() const {
  if (counter_ == 0) {
    return InvalidArgumentError("Preprocessed data is empty.");
  }
  return beaver_matrix_b_c_round_1_.back();
}

StatusOr<applications::SigmoidPrecomputedValue>
LogRegShareProvider::GetSigmoidPrecomputedValue() const {
  if (counter_ == 0) {
    return InvalidArgumentError("Preprocess data is empty.");
  }
  return sigmoid_.back();
}

applications::SigmoidPrecomputedValueNewMic&
LogRegShareProvider::GetSigmoidPrecomputedValueNewMic() {
  return  sigmoid_new_mic_.back();
}

StatusOr<std::vector<uint64_t>>
LogRegShareProvider::GetRoundThreeBeaverMatrixA() const {
  return beaver_matrix_a_round_3_;
}

StatusOr<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
LogRegShareProvider::GetRoundThreeBeaverMatrixBandC() const {
  if (counter_ == 0) {
    return InvalidArgumentError("Preprocessed data is empty.");
  }
  return beaver_matrix_b_c_round_3_.back();
}

Status LogRegShareProvider::Shrink() {
  if (counter_ == 0) {
    return InvalidArgumentError("Preprocessed data is empty.");
  }
  counter_--;
  beaver_matrix_b_c_round_1_.pop_back();
  if (!sigmoid_.empty()) {
    sigmoid_.pop_back();
  } else {
    sigmoid_new_mic_.pop_back();
  }
  beaver_matrix_b_c_round_3_.pop_back();
  return OkStatus();
}

}  // namespace logistic_regression
}  // namespace private_join_and_compute
