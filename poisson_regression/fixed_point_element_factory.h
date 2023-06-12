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

#ifndef PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_FIXED_POINT_ELEMENT_FACTORY_H_
#define PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_FIXED_POINT_ELEMENT_FACTORY_H_

#include "private_join_and_compute/util/status.inc"

namespace private_join_and_compute {

class FixedPointElement;

// Class for generating fixed-point numbers represented as unsigned integers
// in the ring R = \Z_{2^{num_ring_bits}}. The maximum supported ring is
// \Z_{2^63}.
// The bottom half of the ring represents positive numbers and the top half of
// the ring represents negative numbers.
// Requirements: R should be large enough for all operations.
class FixedPointElementFactory {
 public:
  // Constructs a new FixedPointElementFactory that produces
  // FixedPointElements in the ring R = \Z_{2^{num_ring_bits}} with
  // num_fractional_bits number of binary fractional bits.
  // Requirements: (0 <= num_ring_bits < 64)
  // and (0 <= num_fractional_bits < num_ring_bits)
  // Returns an INVALID_ARGUMENT error code if the requirements are not
  // satisfied.
  static StatusOr<FixedPointElementFactory> Create(
      const int64_t num_fractional_bits,
      const int64_t num_ring_bits);

  // Returns a FixedPointElement created from an integer value
  // Requirements: (-2^{num_ring_bits - 1 - num_fractional_bits} < value)
  // and (value < 2^{num_ring_bits - 1 - num_fractional_bits})
  // Returns an INVALID_ARGUMENT error code if the requirements are not
  // satisfied.
  StatusOr<FixedPointElement> CreateFixedPointElementFromInt(
      const int64_t value) const;

  // Returns a FixedPointElement created from a double value which will be
  // truncated to num_fractional_bits binary fractional bits.
  // Requirements: (-2^{num_ring_bits - 1 - num_fractional_bits} < value)
  // and (value < 2^{num_ring_bits - 1 - num_fractional_bits})
  // Returns an INVALID_ARGUMENT error code if the requirements are not
  // satisfied.
  StatusOr<FixedPointElement> CreateFixedPointElementFromDouble(
       const double value) const;

  // Constructs a new FixedPointElement from a fixed-point number represented
  // as an unsigned integer, i.e., already multiplied by 2^{num_frac_bits}.
  // Requirements: (value < 2^{num_ring_bits})
  // Returns an INVALID_ARGUMENT error code if the requirements are not
  // satisfied.
  StatusOr<FixedPointElement> ImportFixedPointElementFromUint64(
      const uint64_t value) const;

  // Returns true if value fits in the ring R.
  // i.e. if value < 2^{num_ring_bits}.
  bool IsUint64InRing(const uint64_t value) const;

  // The parameters for FixedPointElements created with this factory.
  // fractional_multiplier = 2^{num_fractional_bits}
  // integer_ring_multiplier = 2^{num_ring_bits - num_fractional_bits}
  // primary_ring_modulus = 2^{num_ring_bits}
  struct Params {
    uint8_t num_fractional_bits;
    uint8_t num_ring_bits;
    uint64_t fractional_multiplier;
    uint64_t integer_ring_modulus;
    uint64_t primary_ring_modulus;
  };

  inline const Params& GetParams() { return fpe_params_; }

  // Returns true if the two factories have the same parameters.
  bool IsSameFactoryAs(FixedPointElementFactory& factory);

 private:
  friend class FixedPointElement;

  explicit FixedPointElementFactory(const Params& fpe_params);

  Params fpe_params_;
};

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_FIXED_POINT_ELEMENT_FACTORY_H_
