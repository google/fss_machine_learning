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

#ifndef PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_ARITHMETIC_MATRIX_ARITHMETIC_H_
#define PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_ARITHMETIC_MATRIX_ARITHMETIC_H_

#include <vector>

#include "private_join_and_compute/util/status.inc"

namespace private_join_and_compute {

// Compute matrix transpose.
// Input: Matrix mat (dim1 * dim2), flattened into 1d vector
// Output: mat.transpose (dim2 * dim1)
StatusOr<std::vector<uint64_t>> Transpose(const std::vector<uint64_t>& mat,
                                          size_t dim1, size_t dim2);

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_ARITHMETIC_MATRIX_ARITHMETIC_H_
