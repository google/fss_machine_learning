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

#include "poisson_regression/ring_arithmetic_utils.h"

#include "private_join_and_compute/util/status.inc"

namespace private_join_and_compute {

StatusOr<std::vector<uint64_t>> BatchedModAdd(
    const std::vector<uint64_t>& vector_a,
    const std::vector<uint64_t>& vector_b, uint64_t modulus) {
  if (vector_a.empty() || vector_b.empty()) {
    return InvalidArgumentError("BatchedModAdd: input must not be empty.");
  }
  if (vector_a.size() != vector_b.size()) {
    return InvalidArgumentError("BatchedModAdd: input must have "
                                "the same length.");
  }
  std::vector<uint64_t> res(vector_a.size());
  for (size_t idx = 0; idx < vector_a.size(); idx++) {
    res[idx] = ModAdd(vector_a[idx], vector_b[idx], modulus);
  }
  return res;
}

StatusOr<std::vector<uint64_t>> BatchedModSub(
    const std::vector<uint64_t>& vector_a,
    const std::vector<uint64_t>& vector_b, uint64_t modulus) {
  if (vector_a.empty() || vector_b.empty()) {
    return InvalidArgumentError("BatchedModSub: input must not be empty.");
  }
  if (vector_a.size() != vector_b.size()) {
    return InvalidArgumentError("BatchedModSub: input must have "
                                "the same length.");
  }
  std::vector<uint64_t> res(vector_a.size());
  for (size_t idx = 0; idx < vector_a.size(); idx++) {
    res[idx] = ModSub(vector_a[idx], vector_b[idx], modulus);
  }
  return res;
}

StatusOr<std::vector<uint64_t>> BatchedModMul(
    const std::vector<uint64_t>& vector_a,
    const std::vector<uint64_t>& vector_b, uint64_t modulus) {
  if (vector_a.empty() || vector_b.empty()) {
    return InvalidArgumentError("BatchedModMul: input must not be empty.");
  }
  if (vector_a.size() != vector_b.size()) {
    return InvalidArgumentError("BatchedModMul: input must have "
                                "the same length.");
  }
  std::vector<uint64_t> res(vector_a.size());
  for (size_t idx = 0; idx < vector_a.size(); idx++) {
    res[idx] = ModMul(vector_a[idx], vector_b[idx], modulus);
  }
  return res;
}

// Compute a^exp mod modulus.
// NOTE: 0^0 = 1.
uint64_t ModExp(uint64_t a, uint64_t exp, uint64_t modulus) {
  uint64_t res = 1;
  while (exp > 0) {
    if (exp & 1) {
      res = ModMul(res , a, modulus);
    }
    exp >>= 1;
    a = ModMul(a, a, modulus);
  }
  return res;
}

// Compute b = a^{-1} mod modulus assuming that the modulus is prime.
// If modulus is a prime, the inverse always exists and can be computed by using
// little Fermat's theorem: a^{-1} = a^{modulus - 2} mod modulus.
// If modulus is not a prime, This method may fail to find the inverse.
// Thus, after computing a^{modulus - 2}, it is necessary to verify that
// a*a^{modulus - 2} = 1.
StatusOr<uint64_t> ModInv(uint64_t a, uint64_t modulus) {
  if (0 == a) {
    return InvalidArgumentError("0 does not have inverse.");
  } else {
    auto inverse = ModExp(a, modulus - 2, modulus);
    if (ModMul(inverse, a, modulus) != 1) {
      return InvalidArgumentError("ModInv: cannot find inverse. This method "
                                  "works for prime modulus only.");
    } else {
      return inverse;
    }
  }
}

namespace {
// Helper functions to compute batched mod inverse.
inline size_t GetLeftChildIndex(size_t index) {
  return 2*index + 1;
}

inline size_t GetRightChildIndex(size_t index) {
  return 2*index + 2;
}
}  // namespace

// Compute modulo inverse in batch setting assuming that the modulus is prime.
// The batched version that computes mod inverse of N values requires
// O(3N) ModMul operations and 1 actually ModInv operations.
// This is much faster than computing N ModInv operations, which requires
// O(N*log(modulus)) ModMul operations. The saving is O(log(modulus)) when
// N is large.
StatusOr<std::vector<uint64_t>> BatchedModInv(
    const std::vector<uint64_t>& input, uint64_t modulus) {
  if (input.empty()) {
    return InvalidArgumentError("Input length must be a positive number.");
  }

  // If any of the input elements is 0, return INVALID_ARGUMENT.
  for (size_t idx = 0; idx < input.size(); idx++) {
    if (0 == input[idx]) {
      return InvalidArgumentError("0 does not have inverse.");
    }
  }

  // Handle the special case input.size() == 1. When input.size() == 1,
  // logn = log2(input.size()) = 0, thus (1 << logn) - 1 < 0.
  if (1 == input.size()) {
    auto inverse = ModInv(input[0], modulus);
    if (!inverse.ok()) {
      return InvalidArgumentError("BatchedModInv: cannot find inverses.");
    }
    return std::vector<uint64_t>{inverse.value()};
  }

  // When input.size() > 1.
  // heap_array: storage for a heap, where there are N leaf nodes.
  // The value of the leaf nodes is initialized with the value of input.
  // Inner nodes are initialized with 1.
  int logn = std::ceil(std::log2(input.size()));
  std::vector<uint64_t> heap_array((1 << logn) - 1, 1);
  heap_array.insert(heap_array.end(), input.begin(), input.end());
  // If the input size is odd, pad 1 at the end. This is to make all the leaf
  // nodes to have siblings, and all parent nodes to have 2 children.
  if (input.size() % 2 == 1) {
    heap_array.push_back(1);
  }

  // Multiply up the tree. Starting from the layer just above the leaf layer.
  // The value of a node v:
  // value(v) = value(left child)*value(right chile) mod modulus.
  // At the end of this step, the root (heap_array[0]) has value
  // input[0]*...*input[N-1] mod modulus.
  // We only need to perform the computation for the parent nodes that are
  // ancestors of at least one input node. At depth d, if the largest index of
  // the an input's ancestor node is p_d, then we just need to compute value(v)
  // for v from 2^d - 1 to p_d.
  // Just above leaf layer (depth = logn - 1), largest input's ancestor index
  // is (heap_array.size() - 2)/2.
  std::vector<int> largest_parent_indices;
  int largest_parent_index = (heap_array.size() - 2)/2;
  int current_depth = logn - 1;
  while (current_depth >= 0) {
    largest_parent_indices.push_back(largest_parent_index);
    int smallest_parent_index = (1 << current_depth) - 1;
    for (int idx = smallest_parent_index; idx <= largest_parent_index; idx++) {
      heap_array[idx] = ModMul(heap_array[GetLeftChildIndex(idx)],
                               heap_array[GetRightChildIndex(idx)], modulus);
    }
    // Update the values for the layer above this current layer.
    largest_parent_index = (largest_parent_index - 1)/2;
    current_depth -= 1;
  }

  // Compute the inverse of the value at the root. If the inverse does not
  // exist, return INTERNAL_ERROR.
  auto rootInverse = ModInv(heap_array[0], modulus);
  if (!rootInverse.ok()) {
    return InternalError("BatchedModInv: cannot find inverses.");
  }
  heap_array[0] = rootInverse.value();

  // Propagate the inverse down the tree.
  // As v = value(v.left_chile)*value(v.right_child) mod modulus.
  // With v^{-1} mod modulus known, we can compute:
  // (v.left_child}^{-1} mod modulus = v^{-1}*v.right_child mod modulus.
  // (v.right_child}^{-1} mod modulus = v^{-1}*v.left_child mod modulus.
  // We start from the root heap_array[0]. Here we compute inverse of the left
  // and right child of the root. Then we propagate the inverses all the way to
  // the leaf layer. We only perform computation on input's ancestor nodes.
  std::reverse(largest_parent_indices.begin(), largest_parent_indices.end());
  for (int depth = 0; depth < logn; depth++) {
    // Iterate through input's ancestor nodes at this depth.
    for (int idx = (1 << depth) - 1;
      idx <= largest_parent_indices[depth]; idx++) {
      // Store left child.
      uint64_t left_child = heap_array[GetLeftChildIndex(idx)];
      // Compute inverse for the left child and store it at the same node.
      heap_array[GetLeftChildIndex(idx)] =
          ModMul(heap_array[idx], heap_array[GetRightChildIndex(idx)], modulus);
      // Compute inverse for the right child and store it at the same node.
      heap_array[GetRightChildIndex(idx)] =
          ModMul(heap_array[idx], left_child, modulus);
    }
  }

  // Return the leaf nodes containing the inverse of the input.
  return std::vector<uint64_t>(
      heap_array.begin() + (1 << logn) - 1,
      heap_array.begin() + (1 << logn) - 1 + input.size());
}

// Compute matrix multiplication C = A*B mod modulus where A, B have dimension
// (dim1 x dim2) and (dim2 x dim3) respectively.
StatusOr<std::vector<uint64_t>> ModMatrixMul(
    const std::vector<uint64_t>& matrix_a,
    const std::vector<uint64_t>& matrix_b,
    size_t dim1, size_t dim2, size_t dim3, uint64_t modulus) {
  if (matrix_a.empty() || matrix_b.empty()) {
    return InvalidArgumentError("ModMatrixMul: input must not be empty.");
  }
  if (matrix_a.size() != dim1*dim2 || matrix_b.size() != dim2*dim3) {
    return InvalidArgumentError("ModMatrixMul: invalid matrix dimension.");
  }

  // Store the output of the multiplication. The value matrix_c(rdx,cdx) is
  // stored at matrix_c[rdx*dim3 + cdx].
  // matrix_c(rdx,cdx) =
  // sum_{idx = 0...dim2 - 1} matrix_a(rdx, idx)*matrix_b(idx, cdx) mod modulus.
  // matrix_a(rdx, idx) is stored at matrix_a[rdx*dim2 + idx].
  // matrix_b(idx, cdx) is stored at matrix_b[idx*dim3 + cdx].
  std::vector<uint64_t> matrix_c(dim1*dim3);
  for (size_t rdx = 0; rdx < dim1; rdx++) {
    for (size_t cdx = 0; cdx < dim3; cdx++) {
      uint64_t sum = 0;
      for (size_t idx = 0; idx < dim2; idx++) {
        sum = ModAdd(
            ModMul(matrix_a[rdx*dim2 + idx], matrix_b[idx*dim3 + cdx], modulus),
            sum, modulus);
      }
      matrix_c[rdx*dim3 + cdx] = sum;
    }
  }
  return std::move(matrix_c);
}

}  // namespace private_join_and_compute
