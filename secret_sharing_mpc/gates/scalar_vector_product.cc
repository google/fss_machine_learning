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
#include "secret_sharing_mpc/gates/scalar_vector_product.h"

#include <cstdint>

#include "poisson_regression/fixed_point_element.h"
#include "absl/numeric/int128.h"

namespace private_join_and_compute {

StatusOr<std::vector<uint64_t>> ScalarVectorProductPartyZero(
    double scalar_a, const std::vector<uint64_t>& vector_b,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_,
    uint64_t modulus) {
  if (vector_b.empty()) {
    return InvalidArgumentError(
        "ScalarVectorProduct: input vector must not be empty.");
  }

  bool scalar_negative = scalar_a < 0.0;

  // Do scalar-vector product with the absolute value of scalar_a
  // If negative, the output will be negated at the end
  if (scalar_negative) {
     scalar_a *= (-1);
  }

  // Represent as scalar_a * 2^{l_f}.
  ASSIGN_OR_RETURN(FixedPointElement fpe_a,
                   fp_factory_->CreateFixedPointElementFromDouble(scalar_a));
  uint64_t ring_a = fpe_a.ExportToUint64();

  std::vector<uint64_t> res(vector_b.size());
  for (size_t idx = 0; idx < vector_b.size(); idx++) {
    // floor((a * b_0 - a * modulus) / 2^(num_fractional_bits)) % modulus
    res[idx] = absl::Uint128Low64(
        (((absl::int128(ring_a) * absl::int128(vector_b[idx])) -
          (absl::int128(ring_a) * absl::int128(modulus))) >>
        fp_factory_->GetParams().num_fractional_bits) %
        absl::uint128(modulus));
  }

  // Negate the result if scalar was negative
  if (scalar_negative) {
    for (size_t idx = 0; idx < vector_b.size(); idx++) {
      res[idx] = modulus - res[idx];
    }
  }
  return res;
}

StatusOr<std::vector<uint64_t>> ScalarVectorProductPartyOne(
    double scalar_a, const std::vector<uint64_t>& vector_b,
    std::unique_ptr<FixedPointElementFactory>& fp_factory_, uint64_t modulus) {
  if (vector_b.empty()) {
    return InvalidArgumentError(
        "ScalarVectorProduct: input vector must not be empty.");
  }

  // Do scalar-vector product with the absolute value of scalar_a
  // If negative, the output will be negated at the end
  bool scalar_negative = scalar_a < 0.0;

  // Do scalar-vector product with the absolute value of scalar_a
  if (scalar_negative) {
    scalar_a *= (-1);
  }

  // Represent as scalar_a * 2^{l_f}.
  ASSIGN_OR_RETURN(FixedPointElement fpe_a,
                   fp_factory_->CreateFixedPointElementFromDouble(scalar_a));

  std::vector<uint64_t> res(vector_b.size());
  for (size_t idx = 0; idx < vector_b.size(); idx++) {
    // floor((a * b_1) / 2^(num_fractional_bits)) % modulus

    // Import current vector element from uint64_t
    ASSIGN_OR_RETURN(
        FixedPointElement fpe_b_idx,
        fp_factory_->ImportFixedPointElementFromUint64(vector_b[idx]));
    ASSIGN_OR_RETURN(FixedPointElement fpe_prod, fpe_a.TruncMul(fpe_b_idx));

    res[idx] = fpe_prod.ExportToUint64();
  }

  // Negate the result if scalar was negative
  if (scalar_negative) {
    for (size_t idx = 0; idx < vector_b.size(); idx++) {
      res[idx] = modulus - res[idx];
    }
  }

  return res;
}

}  // namespace private_join_and_compute
