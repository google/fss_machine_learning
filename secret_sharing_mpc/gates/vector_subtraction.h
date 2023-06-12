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

#ifndef PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_VECTOR_SUBTRACTION_H_
#define PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_VECTOR_SUBTRACTION_H_

#include <vector>

#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status.inc"

namespace private_join_and_compute {

// The function in this file subtracts two vectors (also flattened matrices).
// Two players P_0, P_1
// Input: [A], [B] (shares of vectors A, B)
// Output: [C] = [A] - [B]
// The vectors A and B must have matching dimensions.

// More specifically, P_0 holds A_0 and B_0 while P_1 holds A_1 and B_1.
// For all i:
// P_0 computes: C_0[i] = A_0[i] - B_0[i]
// P_1 computes: C_1[i] = A_1[i] - B_1[i]
// C[i] = C_0[i] - C_1[i]

// The function first checks if the input vectors are not empty and have the
// same size. If the check fails, it returns INVALID_ARGUMENT.
StatusOr<std::vector<uint64_t>> VectorSubtract(
    const std::vector<uint64_t>& vector_a,
    const std::vector<uint64_t>& vector_b, uint64_t modulus);

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_VECTOR_SUBTRACTION_H_
