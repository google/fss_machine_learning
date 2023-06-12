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

#ifndef PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_FIXED_POINT_ELEMENT_H_
#define PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_FIXED_POINT_ELEMENT_H_

#include "poisson_regression/fixed_point_element_factory.h"
#include "private_join_and_compute/util/status.inc"

namespace private_join_and_compute {

class FixedPointElementFactory;

// Class for representing fixed-point numbers.
// The factory class FixedPointElementFactory should be used to create
// FixedPointElements.
class FixedPointElement {
 public:
  // Returns the value of the FixedPointElement
  uint64_t ExportToUint64() const;

  // Returns the underlying fixed-point number represented by the
  // FixedPointElement.
  double ExportToDouble() const;

  // Returns a FixedPointElement whose value is (this + fpe) in ring R.
  // Assumes that fpe is in the same ring as this FixedPointElement.
  // Undefined behaviour if the two FixedPointElements are not in the same ring.
  // Returns an INTERNAL error code if the operation fails.
  StatusOr<FixedPointElement> ModAdd(const FixedPointElement& fpe) const;

  // Returns a FixedPointElement whose value is (this - fpe) in ring R.
  // Assumes that fpe is in the same ring as this FixedPointElement.
  // Undefined behaviour if the two FixedPointElements are not in the same ring.
  // Returns an INTERNAL error code if the operation fails.
  StatusOr<FixedPointElement> ModSub(const FixedPointElement& fpe) const;

  // Reuturns a FixedPointElement whose value is (this * fpe) in ring R.
  // Assumes that fpe is in the same ring as this FixedPointElement.
  // Undefined behaviour if the two FixedPointElements are not in the same ring.
  // Returns an INTERNAL error code if the operation fails.
  StatusOr<FixedPointElement> ModMul(const FixedPointElement& fpe) const;

  // Returns a FixedPointElement whose value is
  // (uint64_representation(this) * uint64_represtation(fpe)) /
  // 2^{num_fractional_bits} where / denotes integer division
  // in the ring.
  // The TruncMul function performs (truncated) uint multiplication of the
  // underlying values. This means that a special operation is not performed
  // when the underlying values represent negative fixed-point numbers, as
  // opposed to what is done in TruncMulFP.
  //
  // Assumes that fpe is in the same ring as this FixedPointElement.
  // Undefined behaviour if the two FixedPointElements are not in the same ring.
  // Returns an INTERNAL error code if the operation fails.
  StatusOr<FixedPointElement> TruncMul(const FixedPointElement& fpe) const;

  // Returns a FixedPointElement whose value is as follows:
  // Let this_abs and fpe_abs be uint64 representations of the
  // FixedPointElements representing the absolute values of this and fpe.
  // If both this FixedPointElement and fpe represent positive numbers or both
  // represent negative numbers, then this function returns
  // res = (this_abs * fpe_abs) / 2^{num_fractional_bits} where / denotes
  // integer division in the ring.
  // Otherwise, it returns res.Negate().
  // This corresponds to returning the FixedPointElement corresponding to the
  // actual multiplication of the two underlying fixed-point numbers.
  //
  // Assumes that fpe is in the same ring as this FixedPointElement.
  // Undefined behaviour if the two FixedPointElements are not in the same ring.
  // Returns an INTERNAL error code if the operation fails.
  StatusOr<FixedPointElement> TruncMulFP(const FixedPointElement& fpe) const;

  // Returns a FixedPointElement whose value is (this) / (2^{exp})
  // where / denotes integer division.
  // Requirements: (0 <= exp <= num_ring_bits)
  // Returns an INVALID_ARGUMENT error code if the requirements are not
  // satisfied.
  // Returns an INTERNAL error code if the operation fails.
  StatusOr<FixedPointElement> PowerOfTwoDiv(const int64_t exp) const;

  // Returns a FixedPointElement whose value is as follows:
  // Let this_abs be uint64 representations of the
  // FixedPointElements representing the absolute values of this.
  // If fpe is a non-negative fixed point number, return res = this_abs/2^{exp},
  // where / denotes integer division. Else, return res.Negate().
  // Requirements: (0 <= exp <= num_ring_bits)
  // Returns an INVALID_ARGUMENT error code if the requirements are not
  // satisfied.
  // Returns an INTERNAL error code if the operation fails.
  StatusOr<FixedPointElement> PowerOfTwoDivFP(const int64_t exp) const;

  // Returns the additive inverse in R.
  // Returns an INTERNAL error code if the operation fails.
  StatusOr<FixedPointElement> Negate() const;

  // Returns the parameters associated with the FixedPointElement.
  FixedPointElementFactory::Params GetElementParams() const;

 private:
  friend class FixedPointElementFactory;

  FixedPointElement(const FixedPointElementFactory* fpe_factory,
                    const uint64_t value);

  const FixedPointElementFactory* fpe_factory_;
  uint64_t value_;
};

inline StatusOr<FixedPointElement> operator-(const FixedPointElement& fpe) {
  return fpe.Negate();
}

inline StatusOr<FixedPointElement> operator+(const FixedPointElement& fpe1,
                                             const FixedPointElement& fpe2) {
  return fpe1.ModAdd(fpe2);
}

inline StatusOr<FixedPointElement> operator-(const FixedPointElement& fpe1,
                                             const FixedPointElement& fpe2) {
  return fpe1.ModSub(fpe2);
}

inline StatusOr<FixedPointElement> operator*(const FixedPointElement& fpe1,
                                             const FixedPointElement& fpe2) {
  return fpe1.ModMul(fpe2);
}

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_FIXED_POINT_ELEMENT_H_
