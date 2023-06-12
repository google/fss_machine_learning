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

#ifndef PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_PRNG_BASIC_RNG_H_
#define PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_PRNG_BASIC_RNG_H_

#include "private_join_and_compute/crypto/openssl.inc"
#include "poisson_regression/prng/prng.h"
#include "absl/base/casts.h"
#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/strings/string_view.h"

namespace private_join_and_compute {

// Basic RNG class that uses RAND_bytes from OpenSSL to sample randomness.
// BasicRng does not require a seed internally.
class BasicRng : public SecurePrng {
 public:
  // Create a BasicRng object.
  // Returns an INTERNAL error code if the creation fails.
  static StatusOr<std::unique_ptr<BasicRng>> Create(absl::string_view seed) {
    return absl::make_unique<BasicRng>();
  }

  // Sample 8 bits of randomness using OpenSSL RAND_bytes.
  // Returns an INTERNAL error code if the sampling fails.
  inline absl::StatusOr<uint8_t> Rand8() override { return Rand<uint8_t>(); }

  // Sample 64 bits of randomness using OPENSSL RAND_bytes.
  // Returns an INTERNAL error code if the sampling fails.
  inline absl::StatusOr<uint64_t> Rand64() override { return Rand<uint64_t>(); }

  // Sample 128 bits of randomness using OPENSSL RAND_bytes.
  // Returns an INTERNAL error code if the sampling fails.
  inline absl::StatusOr<absl::uint128> Rand128() override {
    return Rand<absl::uint128>();
  }

  // BasicRng does not use seeds.
  static absl::StatusOr<std::string> GenerateSeed() { return std::string(); }
  static int SeedLength() { return 0; }

 private:
  template <typename T>
  absl::StatusOr<T> Rand() {
    std::array<uint8_t, sizeof(T)> rand;
    int success = RAND_bytes(rand.data(), rand.size());
    if (!success) {
      return absl::InternalError(
          "BasicRng::Rand - Failed to create randomness");
    }
    return absl::bit_cast<T>(rand);
  }
};

}  // namespace private_join_and_compute

#endif  // PRIVATE_JOIN_AND_COMPUTE_POISSON_REGRESSION_PRNG_BASIC_RNG_H_
