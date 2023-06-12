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

#include "poisson_regression/gradient_descent_utils.h"

#include "absl/strings/str_cat.h"

namespace private_join_and_compute {
namespace poisson_regression {

StatusOr<ShareProvider> ShareProvider::Create(
    std::vector<BeaverTripleMatrix<uint64_t>> beaver_matrix_round_1,
    std::vector<MultToAddShare> mult_to_add_share_round_2,
    std::vector<BeaverTripleVector<uint64_t>> beaver_vector_round_3,
    std::vector<BeaverTripleMatrix<uint64_t>> beaver_matrix_round_4,
    size_t num_features, size_t feature_length, size_t num_iterations) {
  if (num_iterations == 0) {
    return InvalidArgumentError(
        "Preprocessed data must be available for a "
        "positive number of iterations.");
  }
  // All share vectors must have the same number of entries.
  if (beaver_matrix_round_1.size() != num_iterations ||
      mult_to_add_share_round_2.size() != num_iterations ||
      beaver_vector_round_3.size() != num_iterations ||
      beaver_matrix_round_4.size() != num_iterations) {
    return InvalidArgumentError("Invalid input size.");
  }
  // The shares must have correct shape.
  for (size_t idx = 0; idx < num_iterations; idx++) {
    if (beaver_matrix_round_1[idx].GetA().size() !=
            num_features * feature_length ||
        beaver_matrix_round_1[idx].GetB().size() != feature_length ||
        beaver_matrix_round_1[idx].GetC().size() != num_features ||
        mult_to_add_share_round_2[idx].first.size() != num_features ||
        mult_to_add_share_round_2[idx].second.size() != num_features ||
        beaver_vector_round_3[idx].GetA().size() != num_features ||
        beaver_vector_round_3[idx].GetB().size() != num_features ||
        beaver_vector_round_3[idx].GetC().size() != num_features ||
        beaver_matrix_round_4[idx].GetA().size() != num_features ||
        beaver_matrix_round_4[idx].GetB().size() !=
            num_features * feature_length ||
        beaver_matrix_round_4[idx].GetC().size() != feature_length) {
      return InvalidArgumentError(
          absl::StrCat("Invalid input size: a provided "
                       "share has the wrong shape at index ",
                       idx));
    }
  }
  return ShareProvider(std::move(beaver_matrix_round_1),
                       std::move(mult_to_add_share_round_2),
                       std::move(beaver_vector_round_3),
                       std::move(beaver_matrix_round_4), num_iterations);
}

StatusOr<BeaverTripleMatrix<uint64_t>> ShareProvider::GetRoundOneBeaverMatrix()
    const {
  if (counter_ == 0) {
    return InvalidArgumentError("Preprocess data is empty.");
  }
  return beaver_matrix_round_1_.back();
}

StatusOr<MultToAddShare> ShareProvider::GetRoundTwoMultToAdd() const {
  if (counter_ == 0) {
    return InvalidArgumentError("Preprocess data is empty.");
  }
  return mult_to_add_share_round_2_.back();
}

StatusOr<BeaverTripleVector<uint64_t>> ShareProvider::
    GetRoundThreeBeaverVector() const {
  if (counter_ == 0) {
    return InvalidArgumentError("Preprocess data is empty.");
  }
  return beaver_vector_round_3_.back();
}

StatusOr<BeaverTripleMatrix<uint64_t>> ShareProvider::GetRoundFourBeaverMatrix()
    const {
  if (counter_ == 0) {
    return InvalidArgumentError("Preprocess data is empty.");
  }
  return beaver_matrix_round_4_.back();
}

Status ShareProvider::Shrink() {
  if (counter_ == 0) {
    return InvalidArgumentError("Preprocess data is empty.");
  }
  counter_--;
  beaver_matrix_round_1_.pop_back();
  mult_to_add_share_round_2_.pop_back();
  beaver_vector_round_3_.pop_back();
  beaver_matrix_round_4_.pop_back();
  return OkStatus();
}

}  // namespace poisson_regression
}  // namespace private_join_and_compute
