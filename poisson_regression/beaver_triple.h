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

#ifndef PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_BEAVER_TRIPLE_H_
#define PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_BEAVER_TRIPLE_H_

#include <cstddef>
#include <tuple>
#include <vector>

#include "private_join_and_compute/util/status.inc"

namespace private_join_and_compute {

// The following classes define the data structure for Beaver triple.
// This is a template class. The underlying ModularInt should implement
// arithmetic in a ring Z_N.
// We have two different kinds of Beaver triples.
// (a) Regular triples: C[i] = A[i]*B[i] (mod N) where A, B, and C are vectors
// and the operator * represent regular multiplication in Z_N.
// (b) Vectorized triples: C = A*B where A, B, and C are matrices and the
// operator * represents matrix multiplication.

// Data structure to hold regular Beaver triples.
// Two players hold shares of A, B, and C, where A,B, and C are vectors and
// C[i] = A[i]*B[i]. Each triple (A[i], B[i], C[i]) is used for one
// multiplication [z] = F_mult([x], [y]) = [xy].
template<typename ModularInt>
class BeaverTripleVector {
 public:
  // Create Beaver triple vector.
  // The function takes three vectors A, B, and C of the same length as input.
  // NOTE: Beaver triple vector of size 0 is allowed.
  // Returns INVALID_ARGUMENT if the input vectors have different length.
  static StatusOr<BeaverTripleVector> Create(
      std::vector<ModularInt> va,
      std::vector<ModularInt> vb,
      std::vector<ModularInt> vc) {
    // Checking that all vectors have the same length.
    if (va.size() != vb.size() || vb.size() != vc.size()) {
      return InvalidArgumentError("va, vb, and vc must have the same size.");
    } else {
      return BeaverTripleVector(std::move(va), std::move(vb), std::move(vc));
    }
  }

  // Default copy/move constructors.
  BeaverTripleVector(const BeaverTripleVector&) = default;
  BeaverTripleVector(BeaverTripleVector&&) = default;

  // Default copy/move assignments.
  BeaverTripleVector& operator=(const BeaverTripleVector&) = default;
  BeaverTripleVector& operator=(BeaverTripleVector&&) = default;

  // Destructor.
  ~BeaverTripleVector() = default;

  // Access the triple at a certain index.
  // Return INTERNAL_ERROR if the index is out of bound.
  // Else, return OK_STATUS.
  Status GetTripleAt (
      size_t index,
      ModularInt& a,
      ModularInt& b,
      ModularInt& c) const {
    if (index >= dim_) {
      return InternalError("GetTriple: Out of bound error");
    } else {
      a = va_[index];
      b = vb_[index];
      c = vc_[index];
      return OkStatus();
    }
  }

  // Get method to access the Beaver triple.
  const std::vector<ModularInt>& GetA() const {return va_;}
  const std::vector<ModularInt>& GetB() const {return vb_;}
  const std::vector<ModularInt>& GetC() const {return vc_;}

 private:
  BeaverTripleVector(std::vector<ModularInt> va,
                     std::vector<ModularInt> vb,
                     std::vector<ModularInt> vc)
      : va_(std::move(va)), vb_(std::move(vb)), vc_(std::move(vc)) {
        dim_ = va_.size();
      }

  // Locally store the shares of the Beaver triples.
  std::vector<ModularInt> va_;
  std::vector<ModularInt> vb_;
  std::vector<ModularInt> vc_;

  // The length of va_, vb_, and vc_.
  size_t dim_;
};

// Data structure to store shares of the vectorized Beaver triple matrices.
// In two party setting, the players interact to generate random matrices
// A, B, and C such that C = A*B (mod ModularInt). Each party holds an additive
// share of the above matrices. (A, B, C) is used for one matrix multiplication
// [Z] = F_mult([X], [Y]) = [X*Y].
template<typename ModularInt>
class BeaverTripleMatrix {
 public:
  // Factory to create Beaver Triple Matrix.
  // The function takes three matrices A, B, and C, together with three size
  // values as inputs.
  // The function verifies that input satisfies the size constraints:
  // #col of A = #row of B, #col of B = #col of C, and  #row of A = #row of C.
  // NOTE: It is ambiguous when one of the matrices has dimension (0 x col) or
  // (row x 0). To simplify the problem, we don't consider A(0 x m), B(m x p),
  // C(0 x p) as a valid input.
  // Return INVALID_ARGUMENT if the constraints are not met.
  static StatusOr<BeaverTripleMatrix> Create(
      std::vector<ModularInt> ma,
      std::vector<ModularInt> mb,
      std::vector<ModularInt> mc,
      size_t dim1,
      size_t dim2,
      size_t dim3) {
    // Check if the input is invalid:
    // dim1 == 0 OR dim2 == 0 OR dim3 == 0 OR
    // #row x #col of A != dim1*dim2 OR
    // #row x #col of B != dim2*dim3 OR
    // #row x #col of C != dim1*dim3.
    if (dim1 == 0 || dim2 == 0 || dim3 == 0) {
      return InvalidArgumentError("Matrix dimension must not be zero.");
    } else if (dim1*dim2 != ma.size() || dim2*dim3 != mb.size() ||
        dim1*dim3 != mc.size()) {
      return InvalidArgumentError("Invalid size for matrices (A, or B, or C).");
    } else {
      return BeaverTripleMatrix(std::move(ma), std::move(mb), std::move(mc),
                                dim1, dim2, dim3);
    }
  }

  // Default copy/move constructors.
  BeaverTripleMatrix(const BeaverTripleMatrix&) = default;
  BeaverTripleMatrix(BeaverTripleMatrix&&) = default;

  // Default copy/move assignments.
  BeaverTripleMatrix& operator=(const BeaverTripleMatrix&) = default;
  BeaverTripleMatrix& operator=(BeaverTripleMatrix&&) = default;

  // Destructor.
  ~BeaverTripleMatrix() = default;

  // Get method to access the Beaver triple.
  const std::vector<ModularInt>& GetA() const {return ma_;}
  const std::vector<ModularInt>& GetB() const {return mb_;}
  const std::vector<ModularInt>& GetC() const {return mc_;}
  std::tuple<size_t, size_t, size_t> GetDimensions() const {
    return std::make_tuple(dim1_, dim2_, dim3_);
  }

 private:
  // Constructor.
  BeaverTripleMatrix(std::vector<ModularInt> ma,
                     std::vector<ModularInt> mb,
                     std::vector<ModularInt> mc,
                     size_t dim1,
                     size_t dim2,
                     size_t dim3)
      : ma_(std::move(ma)), mb_(std::move(mb)), mc_(std::move(mc)),
      dim1_(dim1), dim2_(dim2), dim3_(dim3) {}

  // Locally store the shares of the Beaver triples matrices.
  // We represent the matrix as a 1D vector.
  // Value for location (i,j) of matrix ma_ is stored at ma_[i*dim2 + j].
  std::vector<ModularInt> ma_;
  std::vector<ModularInt> mb_;
  std::vector<ModularInt> mc_;

  // Fields that hold the size of the matrices A, B, and C.
  // #row x #col of A_: dim1 x dim2
  // #row x #col of B_: dim2 x dim3
  // #row x #col of C_: dim1 x dim3
  size_t dim1_, dim2_, dim3_;
};

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_BEAVER_TRIPLE_H_
