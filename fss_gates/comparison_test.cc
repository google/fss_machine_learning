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

#include "fss_gates/comparison.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "absl/numeric/int128.h"
#include "fss_gates/comparison.pb.h"
#include "dcf/fss_gates/prng/basic_rng.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/internal/status_matchers.h"
#include "dpf/internal/value_type_helpers.h"
#include "dpf/status_macros.h"
#include <chrono>

namespace distributed_point_functions {
namespace fss_gates {
namespace {

using ::testing::Test;


TEST(ComparisonTest, GenAndEvalSucceedsForSmallGroup) {
  ASSERT_TRUE(true);

  CmpParameters cmp_parameters;
  const int group_size = 4;// must be 1 higher than 3   //16;

  // Setting input group to be Z_{2^16}
  cmp_parameters.set_log_group_size(group_size);

  // Creating a Comparison gate
  DPF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ComparisonGate> CmpGate,
      ComparisonGate::Create(cmp_parameters));

  CmpKey key_0, key_1;

  // Initializing the input and output masks uniformly at random;
  const absl::string_view kSampleSeed = absl::string_view();
  DPF_ASSERT_OK_AND_ASSIGN(
      auto rng, distributed_point_functions::BasicRng::Create(kSampleSeed));

  absl::uint128 N = absl::uint128(1) << cmp_parameters.log_group_size();

  DPF_ASSERT_OK_AND_ASSIGN(absl::uint128 r_in0, rng->Rand64());
  DPF_ASSERT_OK_AND_ASSIGN(absl::uint128 r_in1, rng->Rand64());
  r_in0 = r_in0 % N;
  r_in1 = r_in1 % N;

  DPF_ASSERT_OK_AND_ASSIGN(absl::uint128 r_out, rng->Rand64());
  r_out = r_out % absl::uint128(2);

  // Generating Comparison gate keys
  DPF_ASSERT_OK_AND_ASSIGN(std::tie(key_0, key_1), CmpGate->Gen(r_in0, r_in1, r_out));

  // Inside this loop we will test the Evaluation of the Comparison gate on
  // input values ranging between [0, 400)
  for (uint64_t x = 0; x < 8; x++) {
    for (uint64_t y = 0; y < 8; y++) {
      absl::uint128 res_0, res_1;

      // Evaluating Comparison gate key_0 on masked input x + r_in0, y + r_in1
      DPF_ASSERT_OK_AND_ASSIGN(res_0, CmpGate->Eval(0, key_0, (x + r_in0) % N, (y + r_in1) % N));

      // Evaluating Comparison gate key_1 on masked input x + r_in0, y + r_in1
      DPF_ASSERT_OK_AND_ASSIGN(res_1, CmpGate->Eval(1, key_1, (x + r_in0) % N, (y + r_in1) % N));

      // Reconstructing the actual output of the Equality gate by adding together
      // the secret shared output res_0 and res_1, and then subtracting out
      // the output mask r_out
      absl::uint128 result = res_0 ^ res_1 ^ r_out;

      // Check result is correct

      EXPECT_EQ(result, (x < y));
    }
  }
}

TEST(ComparisonTest, TemporaryBenchmarkTestGenAndEvalSucceedsForSmallGroup) {
  ASSERT_TRUE(true);

  CmpParameters cmp_parameters;
  const int group_size = 64;// must be 1 higher than 3   //16;

// Setting input group to be Z_{2^16}
  cmp_parameters.set_log_group_size(group_size);

// Creating a Comparison gate
  DPF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ComparisonGate> CmpGate,
      ComparisonGate::Create(cmp_parameters));

  CmpKey key_0, key_1;

// Initializing the input and output masks uniformly at random;
  const absl::string_view kSampleSeed = absl::string_view();
  DPF_ASSERT_OK_AND_ASSIGN(
      auto rng, distributed_point_functions::BasicRng::Create(kSampleSeed));

  absl::uint128 N = absl::uint128(1) << cmp_parameters.log_group_size();

  DPF_ASSERT_OK_AND_ASSIGN(absl::uint128 r_in0, rng->Rand64());
  DPF_ASSERT_OK_AND_ASSIGN(absl::uint128 r_in1, rng->Rand64());
  r_in0 = r_in0 % N;
  r_in1 = r_in1 % N;

  DPF_ASSERT_OK_AND_ASSIGN(absl::uint128 r_out, rng->Rand64());
  r_out = r_out % absl::uint128(2);

  // Generating Comparison gate keys
  DPF_ASSERT_OK_AND_ASSIGN(std::tie(key_0, key_1), CmpGate->Gen(r_in0, r_in1, r_out));

  size_t num_evals = 1000;

  // Just for benchmarking idea (batched version incorrect)
//  std::vector<CmpKey> keys_0 (num_evals, key_0), keys_1 (num_evals, key_1);
//  std::vector<uint64_t> junk_inputs_x (num_evals), junk_inputs_y (num_evals);
//
//  auto start = std::chrono::high_resolution_clock::now();
//
//  for (size_t idx = 0; idx < num_evals; idx++) {
//      junk_inputs_x[idx] ^= 0;
//      junk_inputs_y[idx] ^= 0;
//  }
//
//  DPF_ASSERT_OK_AND_ASSIGN(std::vector<absl::uint128> res_masked,
//  CmpGate->BatchEval(0,
//        keys_0,
//        junk_inputs_x,
//        junk_inputs_y));
//
//  for (size_t idx = 0; idx < num_evals; idx++) {
//    res_masked[idx] ^= 0;
//  }



  // Not Batched

  // Inside this loop we will test the Evaluation of the Comparison gate on
  // input values ranging between [0, 400)
  // just set so that we have 1000 iterations
  auto start = std::chrono::high_resolution_clock::now();
  for (uint64_t x = 0; x < num_evals; x++) {
    for (uint64_t y = 0; y < 1; y++) {
      absl::uint128 res_0, res_1;

      // Evaluating Comparison gate key_0 on masked input x + r_in0, y + r_in1
      DPF_ASSERT_OK_AND_ASSIGN(res_0, CmpGate->Eval(0, key_0, (x + r_in0) % N, (y + r_in1) % N));

       // Evaluating Comparison gate key_1 on masked input x + r_in0, y + r_in1
      DPF_ASSERT_OK_AND_ASSIGN(res_1, CmpGate->Eval(1, key_1, (x + r_in0) % N, (y + r_in1) % N));

      // Reconstructing the actual output of the Equality gate by adding together
      // the secret shared output res_0 and res_1, and then subtracting out
      // the output mask r_out
      absl::uint128 result = res_0 ^ res_1 ^ r_out;

      // Check result is correct

      EXPECT_EQ(result, (x < y));
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double end_to_end_time = (std::chrono::duration_cast<std::chrono::microseconds>(
      end - start).count()) / 1e6;;

  std::cerr <<  "End to End time (excluding preprocessing) total (s) = " << end_to_end_time <<std::endl;

}

}  // namespace
}  // namespace fss_gates
}  // namespace distributed_point_functions
