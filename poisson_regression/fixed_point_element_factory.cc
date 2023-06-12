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

#include "poisson_regression/fixed_point_element_factory.h"

#include "poisson_regression/fixed_point_element.h"
#include "private_join_and_compute/util/status.inc"
#include "absl/strings/str_cat.h"

namespace private_join_and_compute {

FixedPointElementFactory::FixedPointElementFactory(
    const Params& fpe_params)
    : fpe_params_(fpe_params) {}

StatusOr<FixedPointElementFactory> FixedPointElementFactory::Create(
      const int64_t num_fractional_bits,
      const int64_t num_ring_bits) {
  if (num_ring_bits < 0 || num_ring_bits > 63) {
      return InvalidArgumentError(
          absl::StrCat("FixedPointElementFactory::Create: Invalid num_ring_bits"
                       ": num_ring_bits must be >=0 and < 64. Given: ",
                       num_ring_bits));
  }
  if (num_fractional_bits < 0 || num_fractional_bits >= num_ring_bits) {
       return InvalidArgumentError(
           absl::StrCat("FixedPointElementFactory::Create: Invalid "
                        "num_fractional_bits: num_fractional_bits must be >=0"
                        "and < num_ring_bits. Given: ", num_fractional_bits));
  }
  Params fpe_params;
  fpe_params.num_fractional_bits = static_cast<uint8_t>(num_fractional_bits);
  fpe_params.num_ring_bits = static_cast<uint8_t>(num_ring_bits);
  fpe_params.fractional_multiplier = 1UL << num_fractional_bits;
  fpe_params.integer_ring_modulus = 1UL <<
      (num_ring_bits - num_fractional_bits);
  fpe_params.primary_ring_modulus = 1UL << num_ring_bits;
  return FixedPointElementFactory(fpe_params);
}

StatusOr<FixedPointElement> FixedPointElementFactory
    ::CreateFixedPointElementFromInt(
      const int64_t value) const {
  return CreateFixedPointElementFromDouble(static_cast<double>(value));
}

StatusOr<FixedPointElement> FixedPointElementFactory
    ::CreateFixedPointElementFromDouble(
       const double value) const {
  int64_t value_bound = static_cast<int64_t>(
      fpe_params_.integer_ring_modulus / 2);
  if (value <= -value_bound || value >= value_bound) {
    return InvalidArgumentError(
           absl::StrCat("FixedPointElementFactory::CreateFixedPointElement: "
                        "Invalid value: abs(value) must be "
                        "< 2^{num_ring_bits - 1 - num_fractional_bits. Given: ",
                        value));
  }

  bool is_negative = value < 0;

  uint64_t positive_fpe_value;

  if (is_negative) {
    positive_fpe_value = (-value) * fpe_params_.fractional_multiplier;
  } else {
    positive_fpe_value = value * fpe_params_.fractional_multiplier;
  }

  FixedPointElement positive_fpe = FixedPointElement(this, positive_fpe_value);
  if (is_negative) {
    return positive_fpe.Negate();
  } else {
    return positive_fpe;
  }
}

StatusOr<FixedPointElement> FixedPointElementFactory
    ::ImportFixedPointElementFromUint64(
      const uint64_t value) const {
  if (!IsUint64InRing(value)) {
    return InvalidArgumentError(
        absl::StrCat("FixedPointElementFactory::"
                     "CreateFixedPointElementFromUint64: Invalid value: "
                     "value must be < 2^{num_ring_bits}. Given: ", value));
  }
  return FixedPointElement(this, value);
}

bool FixedPointElementFactory::IsUint64InRing(const uint64_t value) const {
  return (value < fpe_params_.primary_ring_modulus);
}

bool FixedPointElementFactory::IsSameFactoryAs(
    FixedPointElementFactory& factory) {
  if (fpe_params_.num_fractional_bits != factory.fpe_params_.num_fractional_bits
      || fpe_params_.num_ring_bits != factory.fpe_params_.num_ring_bits
      || fpe_params_.fractional_multiplier
        != factory.fpe_params_.fractional_multiplier
      || fpe_params_.integer_ring_modulus
        != factory.fpe_params_.integer_ring_modulus
      || fpe_params_.primary_ring_modulus
        != factory.fpe_params_.primary_ring_modulus) {
    return false;
  }
  return true;
}

}  // namespace private_join_and_compute
