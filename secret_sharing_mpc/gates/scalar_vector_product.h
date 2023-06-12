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

#ifndef PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_SCALAR_VECTOR_PRODUCT_H_
#define PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_SCALAR_VECTOR_PRODUCT_H_

#include <memory>
#include <vector>

#include "poisson_regression/fixed_point_element_factory.h"

namespace private_join_and_compute {

// The functions in this file (one for each party) multiply all elements
// of a secret-shared vector B (also flattened matrix) by a public scalar a.
// Two players P_0, P_1
// Input: a (scalar), [B] = [b0 b1 ... bn] (secret shares of vector B)
// Output: [aB] = [ab0 ab1 ... abn]

// More specifically, in scalar vector product:
// P_0 & P_1 both hold public a
// P_0 holds B_0 while P_1 holds B_1.
// For all i, the parties want to compute and output:
// P_0 computes: [a * B_0[i]]
// P_1 computes: [a * B_1[i]]

// Due to truncation as a result of multiplication in fixed point,
// Each element of the result can have error of at most 2^-lf
// where lf is the number of fractional bits
// The function first checks if the input vector B is not empty.
// If the check fails, it returns INVALID_ARGUMENT.
// There are two functions, one per party.
// This is because the secure scalar vector product operation is not symmetric.
// P_0 locally computes:
// floor((a * B_0 - a * modulus) / 2^(num_fractional_bits)) % modulus
// P_1 locally computes:
// floor((a * B_1) / 2^(num_fractional_bits)) % modulus

StatusOr<std::vector<uint64_t>> ScalarVectorProductPartyZero(
    double scalar_a, const std::vector<uint64_t>& vector_b,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_,
    uint64_t modulus);

StatusOr<std::vector<uint64_t>> ScalarVectorProductPartyOne(
    double scalar_a, const std::vector<uint64_t>& vector_b,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_,
    uint64_t modulus);

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_SECRET_SHARING_MPC_GATES_SCALAR_VECTOR_PRODUCT_H_
