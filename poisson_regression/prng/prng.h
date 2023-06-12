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

#ifndef PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_PRNG_PRNG_H_
#define PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_PRNG_PRNG_H_

#include "private_join_and_compute/util/status.inc"
#include "absl/numeric/int128.h"
#include "absl/strings/string_view.h"

namespace private_join_and_compute {

// An interface for a secure pseudo-random number generator.
class SecurePrng {
 public:
  virtual StatusOr<uint8_t> Rand8() = 0;
  virtual StatusOr<uint64_t> Rand64() = 0;
  virtual StatusOr<absl::uint128> Rand128() = 0;
  virtual ~SecurePrng() = default;
  static StatusOr<std::unique_ptr<SecurePrng>> Create(
      absl::string_view seed);
  static StatusOr<std::string> GenerateSeed();
  static int SeedLength();
};

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_PRNG_PRNG_H_
