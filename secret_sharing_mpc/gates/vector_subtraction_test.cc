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

#include "secret_sharing_mpc/gates/vector_subtraction.h"

#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace private_join_and_compute {
namespace {
using ::testing::Test;

// kModulus = 2^{64} - 1
const uint64_t kModulus = 18446744073709551615UL;

class VectorSubtractionTest : public Test {};

// In this file, test that secure vector subtraction works correctly.
// I.e. for vectors X, Y, we get Z = X - Y
// Note that this operation is done locally by each party.
// Hence, there is no test pertaining to preprocessing.

// End-to-end test for secure vector subtraction gate.
// Each party has input share of vectors [X], [Y] (matrix flattened into vector)
// The output is a share of vector [X - Y]
TEST_F(VectorSubtractionTest, SecureVectorSubtractionSucceeds) {
  // vector_a = {2^{63}, 5, 20}
  // vector_b = {2^{63}, 25, 5}
  // Testing 3 different cases: =, <, >
  std::vector<uint64_t> vector_a{9223372036854775808UL, 5, 20};
  std::vector<uint64_t> vector_b{9223372036854775808UL, 25, 5};
  ASSERT_OK_AND_ASSIGN(auto vector_c,
                       VectorSubtract(vector_a, vector_b, kModulus));

  // Expected result: {0, -20, 15}
  std::vector<uint64_t> vector_d{0, 18446744073709551595UL, 15};

  EXPECT_EQ(vector_c, vector_d);
}

}  // namespace
}  // namespace private_join_and_compute
