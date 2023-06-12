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

#include "poisson_regression/fixed_point_element.h"

#include "poisson_regression/fixed_point_element_factory.h"
#include "private_join_and_compute/util/status.inc"
#include "absl/numeric/int128.h"
#include "absl/strings/str_cat.h"

namespace private_join_and_compute {
namespace {
bool IsFixedPointNegative(const FixedPointElement& fpe) {
  if (fpe.ExportToUint64() >= fpe.GetElementParams().primary_ring_modulus/2) {
    return true;
  } else {
    return false;
  }
}
}  // namespace

FixedPointElement::FixedPointElement(
    const FixedPointElementFactory* fpe_factory,
    const uint64_t value)
    : fpe_factory_(fpe_factory), value_(value) {}

uint64_t FixedPointElement::ExportToUint64() const {
  return value_;
}

double FixedPointElement::ExportToDouble() const {
  if (IsFixedPointNegative(*this)) {
    uint64_t abs_val = fpe_factory_->fpe_params_.primary_ring_modulus - value_;
    return (-1.0 * abs_val) / fpe_factory_->fpe_params_.fractional_multiplier;
  } else {
    return (1.0 * value_) / fpe_factory_->fpe_params_.fractional_multiplier;
  }
}

StatusOr<FixedPointElement> FixedPointElement::ModAdd(
    const FixedPointElement& fpe) const {
  uint64_t result = (value_ + fpe.value_)
      % fpe_factory_->fpe_params_.primary_ring_modulus;
  return FixedPointElement(fpe_factory_, result);
}

StatusOr<FixedPointElement> FixedPointElement::ModSub(
    const FixedPointElement& fpe) const {
  ASSIGN_OR_RETURN(FixedPointElement negated, fpe.Negate());
  return ModAdd(negated);
}

StatusOr<FixedPointElement> FixedPointElement::ModMul(
    const FixedPointElement& fpe) const {
  absl::uint128 mutliplied = static_cast<absl::uint128>(value_)
      * static_cast<absl::uint128>(fpe.value_);
  uint64_t result = static_cast<uint64_t>(
        mutliplied % fpe_factory_->fpe_params_.primary_ring_modulus);
  return FixedPointElement(fpe_factory_, result);
}

StatusOr<FixedPointElement> FixedPointElement::TruncMul(
    const FixedPointElement& fpe) const {
  absl::uint128 mutliplied = static_cast<absl::uint128>(value_)
      * static_cast<absl::uint128>(fpe.value_);
  absl::uint128 truncated = mutliplied
      / fpe_factory_->fpe_params_.fractional_multiplier;
  uint64_t result = static_cast<uint64_t>(
      truncated % fpe_factory_->fpe_params_.primary_ring_modulus);
  return FixedPointElement(fpe_factory_, result);
}


StatusOr<FixedPointElement> FixedPointElement::TruncMulFP(
    const FixedPointElement& fpe) const {
  bool this_fpe_negative = IsFixedPointNegative(*this);
  bool fpe_negative = IsFixedPointNegative(fpe);

  if (this_fpe_negative && fpe_negative) {
    // Both negative. Result should be positive.
    ASSIGN_OR_RETURN(auto this_fpe_abs, this->Negate());
    ASSIGN_OR_RETURN(auto fpe_abs, fpe.Negate());
    return this_fpe_abs.TruncMul(fpe_abs);
  } else if (this_fpe_negative) {
    // Only this is negative. Result should be negative.
    ASSIGN_OR_RETURN(auto this_fpe_abs, this->Negate());
    ASSIGN_OR_RETURN(auto abs_result, this_fpe_abs.TruncMul(fpe));
    return abs_result.Negate();
  } else if (fpe_negative) {
    // Only fpe is negative. Result should be negative.
    ASSIGN_OR_RETURN(auto fpe_abs, fpe.Negate());
    ASSIGN_OR_RETURN(auto abs_result, this->TruncMul(fpe_abs));
    return abs_result.Negate();
  } else {
    // Both positive. Result should be positive.
    return this->TruncMul(fpe);
  }
}

StatusOr<FixedPointElement> FixedPointElement::PowerOfTwoDiv(
    const int64_t exp) const {
  if (exp > fpe_factory_->fpe_params_.num_ring_bits) {
    return InvalidArgumentError(
        absl::StrCat("FixedPointElement::PowerOfTwoDiv: Invalid exponent exp: "
                        "exp must be <= num_ring_bits. Given:  ", exp));
  }
  uint64_t result = value_ >> exp;
  return FixedPointElement(fpe_factory_, result);
}

StatusOr<FixedPointElement> FixedPointElement::PowerOfTwoDivFP(
    const int64_t exp) const {
  if (exp > fpe_factory_->fpe_params_.num_ring_bits) {
    return InvalidArgumentError(
        absl::StrCat("FixedPointElement::PowerOfTwoDiv: Invalid exponent exp: "
                        "exp must be <= num_ring_bits. Given:  ", exp));
  }
  if (IsFixedPointNegative(*this)) {
    return FixedPointElement(
        fpe_factory_,
        fpe_factory_->fpe_params_.primary_ring_modulus -
        ((fpe_factory_->fpe_params_.primary_ring_modulus - value_) >> exp));
  }
  uint64_t result = value_ >> exp;
  return FixedPointElement(fpe_factory_, result);
}

StatusOr<FixedPointElement> FixedPointElement::Negate() const {
  if (value_ == 0) return *this;
  uint64_t result = fpe_factory_->fpe_params_.primary_ring_modulus - value_;
  return FixedPointElement(fpe_factory_, result);
}

FixedPointElementFactory::Params FixedPointElement::GetElementParams() const {
  return fpe_factory_->fpe_params_;
}

}  // namespace private_join_and_compute
