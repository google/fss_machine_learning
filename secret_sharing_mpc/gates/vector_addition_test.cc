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

#include "secret_sharing_mpc/gates/vector_addition.h"

#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace private_join_and_compute {
namespace {
using ::testing::Test;

// kPrimeModulus = 2^{31} - 1;
const uint64_t kPrimeModulus = 2147483647;

class VectorAdditionTest : public Test {};

// In this file, test that secure vector addition works correctly.
// I.e. for vectors X, Y, we get Z = X + Y
// Note that this operation is done locally by each player.
// Hence, there is no test pertaining to preprocessing.

// End-to-end test for secure vector addition gate.
// Each party has input share of vectors [X], [Y] (matrix flattened into vector)
// The output is a share of vector [X + Y]
TEST_F(VectorAdditionTest, SecureMatrixAdditionSucceeds) {
  std::vector<uint64_t> vector_a{3 * kPrimeModulus + 5, 3 * kPrimeModulus - 15,
                                 3 * kPrimeModulus - 15};
  std::vector<uint64_t> vector_b{2 * kPrimeModulus + 15, 5 * kPrimeModulus + 10,
                                 3 * kPrimeModulus + 15};
  ASSERT_OK_AND_ASSIGN(auto vector_c,
                       VectorAdd(vector_a, vector_b, kPrimeModulus));

  // Expected result: {20, modulus - 5, 0}
  std::vector<uint64_t> vector_d{20, kPrimeModulus - 5, 0};

  EXPECT_EQ(vector_c, vector_d);
}

}  // namespace
}  // namespace private_join_and_compute
