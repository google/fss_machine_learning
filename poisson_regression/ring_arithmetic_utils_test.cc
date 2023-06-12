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

#include "poisson_regression/ring_arithmetic_utils.h"

#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace private_join_and_compute {
namespace {
using ::testing::HasSubstr;
using testing::StatusIs;
using ::testing::Test;

// kModulus = 2^{64} - 1
const uint64_t kModulus = 18446744073709551615UL;

// kPrimeModulus = 2^{31} - 1;
const uint64_t kPrimeModulus = 2147483647;

class RingArithmeticTest : public Test {
};

TEST_F(RingArithmeticTest, TestModAdd) {
  // vector_a = {5, 2^{63}}
  // vector_b = {9, 2^{63}}
  // Test 2 cases: non mod reduction add, mod reduction add.
  std::vector<uint64_t> vector_a{5, 9223372036854775808UL};
  std::vector<uint64_t> vector_b{9, 9223372036854775808UL};
  std::vector<uint64_t> vector_c(vector_a.size());

  for (size_t idx = 0; idx < vector_c.size(); idx++) {
    vector_c[idx] = ModAdd(vector_a[idx], vector_b[idx], kModulus);
  }

  // Expected result: {14, 1}
  std::vector<uint64_t> vector_d{14, 1};

  EXPECT_EQ(vector_c, vector_d);
}

TEST_F(RingArithmeticTest, TestModAddWithInputNotInCorrectRange) {
  // vector_a = {3*modulus + 5, 3*modulus - 15, 3*modulus - 15}
  // vector_b = {2*modulus + 15, 5*modulus + 10, 3*modulus + 15}
  // Test 3 cases:
  // (a % modulus) + (b % modulus) < modulus.
  // (a % modulus) + (b % modulus) = modulus.
  // (a % modulus) + (b % modulus) > modulus.
  std::vector<uint64_t> vector_a{3*kPrimeModulus + 5, 3*kPrimeModulus - 15,
                               3*kPrimeModulus - 15};
  std::vector<uint64_t> vector_b{2*kPrimeModulus + 15, 5*kPrimeModulus + 10,
                               3*kPrimeModulus + 15};
  std::vector<uint64_t> vector_c(vector_a.size());

  for (size_t idx = 0; idx < vector_c.size(); idx++) {
    vector_c[idx] = ModAdd(vector_a[idx], vector_b[idx], kPrimeModulus);
  }

  // Expected result: {20, modulus - 5, 0}
  std::vector<uint64_t> vector_d{20, kPrimeModulus - 5, 0};

  EXPECT_EQ(vector_c, vector_d);
}

TEST_F(RingArithmeticTest, TestModSub) {
  // vector_a = {2^{63}, 5, 20}
  // vector_b = {2^{63}, 25, 5}
  // Testing 3 different cases: =, <, >
  std::vector<uint64_t> vector_a{9223372036854775808UL, 5, 20};
  std::vector<uint64_t> vector_b{9223372036854775808UL, 25, 5};
  std::vector<uint64_t> vector_c(vector_a.size());

  for (size_t idx = 0; idx < vector_c.size(); idx++) {
    vector_c[idx] = ModSub(vector_a[idx], vector_b[idx], kModulus);
  }

  // Expected result: {0, -20, 15}
  std::vector<uint64_t> vector_d{0, 18446744073709551595UL, 15};

  EXPECT_EQ(vector_c, vector_d);
}

TEST_F(RingArithmeticTest, TestModSubWithInputNotInCorrectRange) {
  // vector_a = {3*modulus + 5, 3*modulus + 10, 2*modulus + 30}
  // vector_b = {2*modulus + 15, 5*modulus + 10, 3*modulus + 15}
  // Test 3 cases:
  // (a % modulus) < (b % modulus).
  // (a % modulus) = (b % modulus).
  // (a % modulus) > (b % modulus).
  std::vector<uint64_t> vector_a{3*kPrimeModulus + 5, 3*kPrimeModulus + 10,
                               3*kPrimeModulus + 30};
  std::vector<uint64_t> vector_b{2*kPrimeModulus + 15, 5*kPrimeModulus + 10,
                               3*kPrimeModulus + 15};
  std::vector<uint64_t> vector_c(vector_a.size());

  for (size_t idx = 0; idx < vector_c.size(); idx++) {
    vector_c[idx] = ModSub(vector_a[idx], vector_b[idx], kPrimeModulus);
  }

  // Expected result: {modulus - 10, 0, 15}
  std::vector<uint64_t> vector_d{kPrimeModulus - 10, 0, 15};

  EXPECT_EQ(vector_c, vector_d);
}

TEST_F(RingArithmeticTest, TestModMul) {
  // vector_a = {5, 20}
  // vector_b = {15, 2^{63}}
  // Testing 2 cases: overflow/non-overflow after multiplication.
  std::vector<uint64_t> vector_a{5, 20};
  std::vector<uint64_t> vector_b{15, 9223372036854775808UL};
  std::vector<uint64_t> vector_c(vector_a.size());

  for (size_t idx = 0; idx < vector_c.size(); idx++) {
    vector_c[idx] = ModMul(vector_a[idx], vector_b[idx], kModulus);
  }

  // Expected result: {75, 10}
  std::vector<uint64_t> vector_d{75, 10};

  EXPECT_EQ(vector_c, vector_d);
}

// Test cases for batched operations where one of the inputs is empty.
TEST_F(RingArithmeticTest, TestBatchedModOperationsEmptyInputFails) {
  std::vector<uint64_t> empty_vector(0);
  auto output_add = BatchedModAdd(empty_vector, empty_vector, kPrimeModulus);
  auto output_sub = BatchedModSub(empty_vector, empty_vector, kPrimeModulus);
  auto output_mul = BatchedModMul(empty_vector, empty_vector, kPrimeModulus);
  EXPECT_THAT(output_add,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("BatchedModAdd: input must not be empty.")));
  EXPECT_THAT(output_sub,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("BatchedModSub: input must not be empty.")));
  EXPECT_THAT(output_mul,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("BatchedModMul: input must not be empty.")));
}

// Test cases for batched operations where input's size is invalid.
TEST_F(RingArithmeticTest, TestBatchedModOperationsInputSizeMismatchFails) {
  std::vector<uint64_t> vector_a{1, 2, 3};
  std::vector<uint64_t> vector_b{1, 2, 3, 4};
  auto output_add = BatchedModAdd(vector_a, vector_b, kPrimeModulus);
  auto output_sub = BatchedModSub(vector_a, vector_b, kPrimeModulus);
  auto output_mul = BatchedModMul(vector_a, vector_b, kPrimeModulus);
  EXPECT_THAT(output_add, StatusIs(StatusCode::kInvalidArgument,
                                   HasSubstr("BatchedModAdd: input must have "
                                             "the same length.")));
  EXPECT_THAT(output_sub, StatusIs(StatusCode::kInvalidArgument,
                                   HasSubstr("BatchedModSub: input must have "
                                             "the same length.")));
  EXPECT_THAT(output_mul, StatusIs(StatusCode::kInvalidArgument,
                                   HasSubstr("BatchedModMul: input must have "
                                             "the same length.")));
}

TEST_F(RingArithmeticTest, TestBatchedModAddSucceeds) {
  std::vector<uint64_t> vector_a{3*kPrimeModulus + 5, 3*kPrimeModulus - 15,
                               3*kPrimeModulus - 15};
  std::vector<uint64_t> vector_b{2*kPrimeModulus + 15, 5*kPrimeModulus + 10,
                               3*kPrimeModulus + 15};
  ASSERT_OK_AND_ASSIGN(auto vector_c,
                       BatchedModAdd(vector_a, vector_b, kPrimeModulus));

  // Expected result: {20, modulus - 5, 0}
  std::vector<uint64_t> vector_d{20, kPrimeModulus - 5, 0};

  EXPECT_EQ(vector_c, vector_d);
}

TEST_F(RingArithmeticTest, TestBatchedModSubSucceeds) {
  // vector_a = {2^{63}, 5, 20}
  // vector_b = {2^{63}, 25, 5}
  // Testing 3 different cases: =, <, >
  std::vector<uint64_t> vector_a{9223372036854775808UL, 5, 20};
  std::vector<uint64_t> vector_b{9223372036854775808UL, 25, 5};
  ASSERT_OK_AND_ASSIGN(auto vector_c,
                       BatchedModSub(vector_a, vector_b, kModulus));

  // Expected result: {0, -20, 15}
  std::vector<uint64_t> vector_d{0, 18446744073709551595UL, 15};

  EXPECT_EQ(vector_c, vector_d);
}

TEST_F(RingArithmeticTest, TestBatchedModMulSucceeds) {
  std::vector<uint64_t> vector_a{5, 20};
  std::vector<uint64_t> vector_b{15, 9223372036854775808UL};
  ASSERT_OK_AND_ASSIGN(auto vector_c,
                       BatchedModMul(vector_a, vector_b, kModulus));

  // Expected result: {75, 10}
  std::vector<uint64_t> vector_d{75, 10};

  EXPECT_EQ(vector_c, vector_d);
}

TEST_F(RingArithmeticTest, TestModExpZeroToZero) {
  auto exp = ModExp(0, 0, kPrimeModulus);
  EXPECT_EQ(exp, 1);
}

TEST_F(RingArithmeticTest, TestModExp) {
  auto exp1 = ModExp(0, 15, kPrimeModulus);
  auto exp2 = ModExp(15, 0, kPrimeModulus);
  auto exp3 = ModExp(30, kPrimeModulus - 1, kPrimeModulus);
  auto exp4 = ModExp(30, 40, kPrimeModulus);
  EXPECT_EQ(exp1, 0);
  EXPECT_EQ(exp2, 1);
  EXPECT_EQ(exp3, 1);
  EXPECT_EQ(exp4, 1815815085);
}

TEST_F(RingArithmeticTest, TestModInvInvalidArgument) {
  auto inverse = ModInv(0, kPrimeModulus);
  EXPECT_THAT(inverse, StatusIs(StatusCode::kInvalidArgument,
                                HasSubstr("0 does not have inverse.")));
}

// If the function does not return an inverse, return INTERNAL_ERROR.
TEST_F(RingArithmeticTest, TestModInvDoesNotExist) {
  uint64_t non_prime_modulus = 16;
  auto inverse = ModInv(2, non_prime_modulus);
  EXPECT_THAT(inverse, StatusIs(StatusCode::kInvalidArgument,
                                HasSubstr("ModInv: cannot find inverse.")));
}

TEST_F(RingArithmeticTest, TestValidModInv) {
  ASSERT_OK_AND_ASSIGN(auto inverse1, ModInv(1, kPrimeModulus));
  ASSERT_OK_AND_ASSIGN(auto inverse2, ModInv(2, kPrimeModulus));
  ASSERT_OK_AND_ASSIGN(auto inverse3, ModInv(kPrimeModulus - 1, kPrimeModulus));
  ASSERT_OK_AND_ASSIGN(auto inverse4, ModInv(21341254, kPrimeModulus));
  EXPECT_EQ(inverse1, 1);
  EXPECT_EQ(inverse2, (kPrimeModulus + 1)/2);
  EXPECT_EQ(inverse3, kPrimeModulus - 1);
  EXPECT_EQ(ModMul(21341254, inverse4, kPrimeModulus), 1);
}

// Test batched mod inverse with invalid input.
TEST_F(RingArithmeticTest, TestBatchedModInvInvalidArgument) {
  const std::vector<uint64_t> input{1, 2, 3, 0, 5, 6};
  auto inverses = BatchedModInv(input, kPrimeModulus);
  EXPECT_THAT(inverses, StatusIs(StatusCode::kInvalidArgument,
                                 HasSubstr("0 does not have inverse.")));
}

// Test batched mod inverse where one of the input values does not have inverse.
TEST_F(RingArithmeticTest, TestBatchedModInvCannotFindInvserse) {
  uint64_t non_prime_modulus = 16;
  const std::vector<uint64_t> input{1, 2, 3, 4, 5, 6};
  auto inverses = BatchedModInv(input, non_prime_modulus);
  EXPECT_THAT(inverses,
              StatusIs(StatusCode::kInternal,
                       HasSubstr("BatchedModInv: cannot find inverses.")));
}

// Test batched mod inverse with input of size 1.
TEST_F(RingArithmeticTest, TestBatchedModInvSucceedsOnInputSizeOne) {
  std::vector<uint64_t> input{23};
  ASSERT_OK_AND_ASSIGN(auto inverses, BatchedModInv(input, kPrimeModulus));
  EXPECT_EQ(inverses.size(), input.size());
  for (size_t idx = 0; idx < input.size(); idx++) {
    EXPECT_EQ(ModMul(inverses[idx], input[idx], kPrimeModulus), 1);
  }
}

// Test batched mod inverse with input of size a power of 2.
TEST_F(RingArithmeticTest, TestBatchedModInvSucceedsOnPowerOfTwoInputSize) {
  std::vector<uint64_t> input{1, 2, 3, 4, 5, 6, 7, 8};
  ASSERT_OK_AND_ASSIGN(auto inverses, BatchedModInv(input, kPrimeModulus));
  EXPECT_EQ(inverses.size(), input.size());
  for (size_t idx = 0; idx < input.size(); idx++) {
    EXPECT_EQ(ModMul(inverses[idx], input[idx], kPrimeModulus), 1);
  }
}

// Test batched mod inverse with input of size not a power of 2.
TEST_F(RingArithmeticTest, TestBatchedModInvSucceedsOnNonPowerOfTwoInputSize) {
  // Test odd size input.
  std::vector<uint64_t> input{1, 2, 3, 4, 5};
  ASSERT_OK_AND_ASSIGN(auto inverses, BatchedModInv(input, kPrimeModulus));
  EXPECT_EQ(inverses.size(), input.size());
  for (size_t idx = 0; idx < input.size(); idx++) {
    EXPECT_EQ(ModMul(inverses[idx], input[idx], kPrimeModulus), 1);
  }
  // Test even size input.
  input.push_back(6);
  ASSERT_OK_AND_ASSIGN(auto inverses2, BatchedModInv(input, kPrimeModulus));
  EXPECT_EQ(inverses2.size(), input.size());
  for (size_t idx = 0; idx < input.size(); idx++) {
    EXPECT_EQ(ModMul(inverses2[idx], input[idx], kPrimeModulus), 1);
  }
}

TEST_F(RingArithmeticTest, ModMatrixMulEmptyInputFails) {
  std::vector<uint64_t> empty_vector;
  std::vector<uint64_t> vector_a{1};
  auto vector_c = ModMatrixMul(empty_vector, vector_a, 1, 1, 1, kPrimeModulus);
  EXPECT_THAT(vector_c,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("ModMatrixMul: input must not be empty.")));
}

// vector_b.size() = 4 != 2*1.
TEST_F(RingArithmeticTest, ModMatrixMulInvalidDimensionFails) {
  std::vector<uint64_t> vector_a{1, 2};
  std::vector<uint64_t> vector_b{1, 2, 3, 4};
  auto vector_c = ModMatrixMul(vector_a, vector_b, 1, 2, 1, kPrimeModulus);
  EXPECT_THAT(vector_c,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("ModMatrixMul: invalid matrix dimension.")));
}

// Test [1 2]*[1 2 | 3 4] = [7 10].
TEST_F(RingArithmeticTest, ModMatrixMulSmallInputSucceeds) {
  std::vector<uint64_t> vector_a{1, 2};
  std::vector<uint64_t> vector_b{1, 2, 3, 4};
  std::vector<uint64_t> expected_output{7, 10};
  size_t dim1 = 1;
  size_t dim2 = 2;
  size_t dim3 = 2;
  ASSERT_OK_AND_ASSIGN(auto vector_c,
                       ModMatrixMul(vector_a, vector_b,
                                    dim1, dim2, dim3, kPrimeModulus));
  EXPECT_EQ(vector_c.size(), dim1*dim3);
  EXPECT_EQ(vector_c, expected_output);
}

}  // namespace
}  // namespace private_join_and_compute
