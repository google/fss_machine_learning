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

#include "fss_gates/equality.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "absl/numeric/int128.h"
#include "fss_gates/equality.pb.h"
#include "dcf/fss_gates/prng/basic_rng.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/internal/status_matchers.h"
#include "dpf/internal/value_type_helpers.h"
#include "dpf/status_macros.h"

namespace distributed_point_functions {
namespace fss_gates {
namespace {

using ::testing::Test;

TEST(EqualityTest, GenAndEvalSucceedsForSmallGroupWithOutputsInLargeGroup) {
  EqParameters eq_parameters;
  const int group_size = 16;

  // Setting input group to be Z_{2^16}
  eq_parameters.set_log_group_size(group_size);

  // Setting output group to be Z_{2^20}
  const uint64_t m = 20;
  const uint64_t M = 1ULL << m;
  eq_parameters.set_output_group_modulus(M);

  // Creating a Equality gate
  DPF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<EqualityGate> EqGate,
      EqualityGate::Create(eq_parameters));

  EqKey key_0, key_1;

  // Initializing the input and output masks uniformly at random;
  const absl::string_view kSampleSeed = absl::string_view();
  DPF_ASSERT_OK_AND_ASSIGN(
      auto rng, distributed_point_functions::BasicRng::Create(kSampleSeed));

  absl::uint128 N = absl::uint128(1) << eq_parameters.log_group_size();

  DPF_ASSERT_OK_AND_ASSIGN(absl::uint128 r_in0, rng->Rand64());
  DPF_ASSERT_OK_AND_ASSIGN(absl::uint128 r_in1, rng->Rand64());
  r_in0 = r_in0 % N;
  r_in1 = r_in1 % N;

  DPF_ASSERT_OK_AND_ASSIGN(absl::uint128 r_out, rng->Rand64());
  r_out = r_out % absl::uint128(M);

  // Generating Equality gate keys
  DPF_ASSERT_OK_AND_ASSIGN(std::tie(key_0, key_1), EqGate->Gen(r_in0, r_in1, r_out));

  // Inside this loop we will test the Evaluation of the Equality gate on
  // input values ranging between [0, 400)
  for (uint64_t x = 0; x < 400; x++) {
    for (uint64_t y = 0; y < 400; y++) {
      absl::uint128 res_0, res_1;

      // Evaluating Equality gate key_0 on masked input x + r_in0, y + r_in1
      DPF_ASSERT_OK_AND_ASSIGN(res_0, EqGate->Eval(key_0, (x + r_in0) % N, (y + r_in1) % N));

      // Evaluating Equality gate key_1 on masked input x + r_in0, y + r_in1
      DPF_ASSERT_OK_AND_ASSIGN(res_1, EqGate->Eval(key_1, (x + r_in0) % N, (y + r_in1) % N));

      // Reconstructing the actual output of the Equality gate by adding together
      // the secret shared output res_0 and res_1, and then subtracting out
      // the output mask r_out
      absl::uint128 result = (res_0 + res_1 - r_out) % absl::uint128(M);

      // Check result is correct
      EXPECT_EQ(result, (x == y));
    }
  }
}

}  // namespace
}  // namespace fss_gates
}  // namespace distributed_point_functions