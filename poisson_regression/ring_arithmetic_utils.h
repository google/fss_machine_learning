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

#ifndef PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_RING_ARITHMETIC_UTILS_H_
#define PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_RING_ARITHMETIC_UTILS_H_

#include <algorithm>
#include <string>
#include <vector>

#include "private_join_and_compute/util/status.inc"
#include "absl/numeric/int128.h"

namespace private_join_and_compute {

// If a, b are not in (0, modulus - 1), perform a mod reduction first.
// Case 1: a + b < modulus -> return (a + b).
// Case 2: a + b >= modulus -> a >= (modulus - b).
// In this case, the return value is (a + b - modulus), which is equivalent to
// (a - (modulus - b)).
inline uint64_t ModAdd(uint64_t a, uint64_t b, uint64_t modulus) {
  if (a >= modulus) {
    a %= modulus;
  }

  if (b >= modulus) {
    b %= modulus;
  }

  if (a >= (modulus - b)) {
    return (a - (modulus - b));
  } else {
    return (a + b);
  }
}

// If a, b are not in (0, modulus - 1), perform a mod reduction first.
// Case 1: a >= b -> return (a - b).
// Case 2: a < b -> b - a > 0. In this case, the return value is
// (a - b + modulus), which is equivalent to (modulus - (b - a)).
inline uint64_t ModSub(uint64_t a, uint64_t b, uint64_t modulus) {
  if (a >= modulus) {
    a %= modulus;
  }

  if (b >= modulus) {
    b %= modulus;
  }

  if (a >= b) {
    return (a - b);
  } else {
    return (modulus - (b - a));
  }
}

// Unoptimized multiplication: convert input into uint128, then perform
// modulo operation and convert the output back to uint64_t.
inline uint64_t ModMul(uint64_t a, uint64_t b, uint64_t modulus) {
  return absl::Uint128Low64((absl::uint128(a) * absl::uint128(b))
                             % absl::uint128(modulus));
}

// Batched version for ModAdd, ModSub, and ModMul.
// The functions first check if the input vectors are not empty and have the
// same size. If the check fails, they return INVALID_ARGUMENT.
StatusOr<std::vector<uint64_t>> BatchedModAdd(
    const std::vector<uint64_t>& vector_a,
    const std::vector<uint64_t>& vector_b, uint64_t modulus);

StatusOr<std::vector<uint64_t>> BatchedModSub(
    const std::vector<uint64_t>& vector_a,
    const std::vector<uint64_t>& vector_b, uint64_t modulus);

StatusOr<std::vector<uint64_t>> BatchedModMul(
    const std::vector<uint64_t>& vector_a,
    const std::vector<uint64_t>& vector_b, uint64_t modulus);

// Compute a^exp mod modulus.
// Return INVALID_ARGUMENT if a = exp = 0 (0^0 is not defined).
uint64_t ModExp(uint64_t a, uint64_t exp, uint64_t modulus);

// Compute b = a^{-1} mod modulus assuming that the modulus is prime.
StatusOr<uint64_t> ModInv(uint64_t a, uint64_t modulus);

// Compute modulo inverse in batch setting assuming that the modulus is prime.
StatusOr<std::vector<uint64_t>> BatchedModInv(
    const std::vector<uint64_t>& input, uint64_t modulus);

// Compute matrix multiplication A*B mod modulus where A, B have dimension
// (dim1 x dim2) and (dim2 x dim3) respectively. This function uses standard
// matrix multiplication.
// Return INVALID_ARGUMENT if any of the matrix is empty, or the dimensions
// are not valid for the multiplication.
StatusOr<std::vector<uint64_t>> ModMatrixMul(
    const std::vector<uint64_t>& matrix_a,
    const std::vector<uint64_t>& matrix_b,
    size_t dim1, size_t dim2, size_t dim3, uint64_t modulus);
}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_RING_ARITHMETIC_UTILS_H_
