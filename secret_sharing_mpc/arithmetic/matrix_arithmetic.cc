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

#include "secret_sharing_mpc/arithmetic/matrix_arithmetic.h"

namespace private_join_and_compute {

StatusOr<std::vector<uint64_t>> Transpose(const std::vector<uint64_t>& mat,
                                          size_t dim1, size_t dim2) {
  // Verify that dim1, dim2 correspond to the matrix mat size.
  if (mat.size() != (dim1 * dim2)) {
    return InvalidArgumentError("Transpose: invalid dimensions.");
  }

  std::vector<uint64_t> mat_transpose(dim2 * dim1);
  for (size_t idx_i = 0; idx_i < dim1; idx_i++) {
    for (size_t idx_j = 0; idx_j < dim2; idx_j++) {
      size_t mat_idx = idx_i * dim2 + idx_j;
      size_t mat_transpose_idx = idx_j * dim1 + idx_i;
      mat_transpose[mat_transpose_idx] = mat[mat_idx];
    }
  }
  return mat_transpose;
}

}  // namespace private_join_and_compute
