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

#ifndef PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_ARITHMETIC_FIXED_POINT_ARITHMETIC_H_
#define PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_ARITHMETIC_FIXED_POINT_ARITHMETIC_H_

#include <memory>
#include <vector>

#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"

namespace private_join_and_compute {

// Compute matrix multiplication C = A * B mod modulus where A, B have
// dimensions (dim1 x dim2) and (dim2 x dim3) respectively.
// Each element product includes an additional term 2^num_fractional_bits,
// which needs to be divided. Thus, for each of dim2 element multiplications,
// the result needs to be truncated which introduces
// an error of at most 2^-lf (lf = number of fractional bits)
// per multiplication.
StatusOr<std::vector<FixedPointElement>> TruncMatrixMulFP(
    const std::vector<FixedPointElement>& matrix_a,
    const std::vector<FixedPointElement>& matrix_b, size_t dim1, size_t dim2,
    size_t dim3, std::unique_ptr<FixedPointElementFactory>& fp_factory_);

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_ARITHMETIC_FIXED_POINT_ARITHMETIC_H_
