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

#include "secret_sharing_mpc/arithmetic/fixed_point_arithmetic.h"

#include <utility>
#include <vector>

#include "poisson_regression/fixed_point_element.h"

namespace private_join_and_compute {

StatusOr<std::vector<FixedPointElement>> TruncMatrixMulFP(
    const std::vector<FixedPointElement>& matrix_a,
    const std::vector<FixedPointElement>& matrix_b, size_t dim1, size_t dim2,
    size_t dim3, std::unique_ptr<FixedPointElementFactory>& fp_factory_) {
  if (matrix_a.empty() || matrix_b.empty()) {
    return InvalidArgumentError("TruncMatrixMulFP: input must not be empty.");
  }
  if (matrix_a.size() != dim1 * dim2 || matrix_b.size() != dim2 * dim3) {
    return InvalidArgumentError("TruncMatrixMulFP: invalid matrix dimension.");
  }

  // Store the output of the multiplication. The value matrix_c(rdx,cdx) is
  // stored at matrix_c[rdx*dim3 + cdx].
  // matrix_c(rdx,cdx) =
  // sum_{idx = 0...dim2 - 1} matrix_a(rdx, idx)*matrix_b(idx, cdx) mod modulus.
  // matrix_a(rdx, idx) is stored at matrix_a[rdx*dim2 + idx].
  // matrix_b(idx, cdx) is stored at matrix_b[idx*dim3 + cdx].
  ASSIGN_OR_RETURN(FixedPointElement zero,
                   fp_factory_->CreateFixedPointElementFromInt(0));
  std::vector<FixedPointElement> matrix_c(dim1 * dim3, zero);
  for (size_t rdx = 0; rdx < dim1; rdx++) {
    for (size_t cdx = 0; cdx < dim3; cdx++) {
      FixedPointElement sum = zero;
      for (size_t idx = 0; idx < dim2; idx++) {
        ASSIGN_OR_RETURN(
            FixedPointElement element_prod,
            matrix_a[rdx * dim2 + idx].TruncMulFP(matrix_b[idx * dim3 + cdx]));
        ASSIGN_OR_RETURN(sum, sum.ModAdd(element_prod));
      }
      matrix_c[rdx * dim3 + cdx] = sum;
    }
  }
  return std::move(matrix_c);
}

}  // namespace private_join_and_compute
