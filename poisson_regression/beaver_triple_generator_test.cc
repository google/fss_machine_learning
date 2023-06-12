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


#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/prng/basic_rng.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status.inc"
#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace private_join_and_compute {
namespace {

using ::testing::HasSubstr;
using testing::StatusIs;
using ::testing::Test;

// kModulus = 2^{64} - 1
const uint64_t kModulus = 18446744073709551615UL;

// kPrimeModulus = 2^{63} - 25;
const uint64_t kPrimeModulus = (1ULL << 63) - 25;

const absl::string_view kPrngSeed = "0123456789abcdef0123456789abcdef";

class BeaverGeneratorWithPrngTest : public Test {
 protected:
  void SetUp() override {
    modulus_ = kModulus;
  }

  StatusOr<std::unique_ptr<BasicRng>> MakePrng() {
    auto prng = BasicRng::Create(kPrngSeed);
    if (!prng.ok()) {
      return InternalError("Prng fails to be initialized.");
    }
    return prng;
  }

  uint64_t modulus_;
};

// Test the function that generates random vector of ModularInt from a
// random seed. The input is valid in this case.
TEST_F(BeaverGeneratorWithPrngTest, SmallVectorTest) {
  ASSERT_OK_AND_ASSIGN(auto prng, this->MakePrng());
  size_t length = 10;
  auto random_vector = SampleVectorFromPrng(length, this->modulus_, prng.get());
  EXPECT_OK(random_vector);
  EXPECT_EQ(random_vector.value().size(), length);
}

// Test the function that generates random vector of ModularInt from a
// random seed. The input is invalid in this case.
TEST_F(BeaverGeneratorWithPrngTest, EmptyLengthVectorFailsTest) {
  ASSERT_OK_AND_ASSIGN(auto prng, this->MakePrng());
  auto random_vector = SampleVectorFromPrng(0, this->modulus_, prng.get());
  EXPECT_THAT(
      random_vector,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("length of vector must be a positive integer")));
}

// Test the function that generates random Beaver triple vector share from a
// random seed. The input is invalid in this case.
TEST_F(BeaverGeneratorWithPrngTest, EmptyBeaverVectorShareFailsTest) {
  size_t dim = 0;
  auto beaver_triple_vector_shares = internal::SampleBeaverVectorShareWithPrng(
      dim, this->modulus_);

  EXPECT_THAT(
      beaver_triple_vector_shares,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("length of vector must be a positive integer")));
}

// Test the function that generates random Beaver triple vector from a
// PRNG. The input is valid in this case.
// The output of the function is share of Beaver triple vector (A, B, C)
// split between two parties. Also, C[idx] = A[idx]*B[idx].
TEST_F(BeaverGeneratorWithPrngTest, SmallBeaverVectorShareTest) {
  size_t dim = 10;
  // Generate share of Beaver triple vector for each party.
  ASSERT_OK_AND_ASSIGN(auto beaver_triple_vector_shares,
                       internal::SampleBeaverVectorShareWithPrng(
                           dim, this->modulus_));
  auto beaver_triple_vector_0 = beaver_triple_vector_shares.first;
  auto beaver_triple_vector_1 = beaver_triple_vector_shares.second;

  std::vector<uint64_t> vector_a_0 = beaver_triple_vector_0.GetA();
  std::vector<uint64_t> vector_b_0 = beaver_triple_vector_0.GetB();
  std::vector<uint64_t> vector_c_0 = beaver_triple_vector_0.GetC();
  std::vector<uint64_t> vector_a_1 = beaver_triple_vector_1.GetA();
  std::vector<uint64_t> vector_b_1 = beaver_triple_vector_1.GetB();
  std::vector<uint64_t> vector_c_1 = beaver_triple_vector_1.GetC();

  EXPECT_EQ(vector_a_0.size(), dim);
  EXPECT_EQ(vector_a_1.size(), dim);
  EXPECT_EQ(vector_b_0.size(), dim);
  EXPECT_EQ(vector_b_1.size(), dim);
  EXPECT_EQ(vector_c_0.size(), dim);
  EXPECT_EQ(vector_c_1.size(), dim);

  std::vector<uint64_t> vector_a, vector_b, vector_c;
  vector_a.reserve(dim);
  vector_b.reserve(dim);
  vector_c.reserve(dim);

  // Reconstruct the vector A, B, C.
  for (size_t idx = 0; idx < dim; idx++) {
    vector_a.push_back(
        ModAdd(vector_a_0[idx], vector_a_1[idx], this->modulus_));
    vector_b.push_back(
        ModAdd(vector_b_0[idx], vector_b_1[idx], this->modulus_));
    vector_c.push_back(
        ModAdd(vector_c_0[idx], vector_c_1[idx], this->modulus_));
  }

  uint64_t zero = 0;
  for (size_t idx = 0; idx < 10; idx++) {
    // Verify that C[idx] == A[idx] * B[idx] mod modulus.
    EXPECT_EQ(vector_c[idx],
              ModMul(vector_a[idx], vector_b[idx], this->modulus_));
    // Verify that C is not a zero vector.
    EXPECT_FALSE(vector_c[idx] == zero);
  }
}

// Test the function that generates random Beaver triple matrix from a
// PRNG. The input is valid in this case.
// The output of the function is share of Beaver triple matrix (A, B, C) split
// between two parties. Also, C = A*B where * represents matrix multiplication
// mod modulus.
TEST_F(BeaverGeneratorWithPrngTest, SmallBeaverMatrixTest) {
  size_t dim1 = 3;
  size_t dim2 = 4;
  size_t dim3 = 5;

  // Generate share of Beaver triple matrix for each party.
  ASSERT_OK_AND_ASSIGN(auto beaver_triple_matrix_shares,
                       internal::SampleBeaverMatrixShareWithPrng(
                           dim1, dim2, dim3, this->modulus_));

  auto beaver_triple_matrix_0 = beaver_triple_matrix_shares.first;
  auto beaver_triple_matrix_1 = beaver_triple_matrix_shares.second;

  std::vector<uint64_t> vector_a_0 = beaver_triple_matrix_0.GetA();
  std::vector<uint64_t> vector_b_0 = beaver_triple_matrix_0.GetB();
  std::vector<uint64_t> vector_c_0 = beaver_triple_matrix_0.GetC();

  std::vector<uint64_t> vector_a_1 = beaver_triple_matrix_1.GetA();
  std::vector<uint64_t> vector_b_1 = beaver_triple_matrix_1.GetB();
  std::vector<uint64_t> vector_c_1 = beaver_triple_matrix_1.GetC();

  // Check if the sizes are correct.
  EXPECT_EQ(vector_a_0.size(), dim1*dim2);
  EXPECT_EQ(vector_a_1.size(), dim1*dim2);
  EXPECT_EQ(vector_b_0.size(), dim2*dim3);
  EXPECT_EQ(vector_b_1.size(), dim2*dim3);
  EXPECT_EQ(vector_c_0.size(), dim1*dim3);
  EXPECT_EQ(vector_c_1.size(), dim1*dim3);

  std::vector<uint64_t> vector_a, vector_b, vector_c;
  vector_a.reserve(dim1*dim2);
  vector_b.reserve(dim2*dim3);
  vector_c.reserve(dim1*dim3);

  // Reconstruct A, B, C from the shares.
  for (size_t idx = 0; idx < dim1 * dim2; idx++) {
    vector_a.push_back(
        ModAdd(vector_a_0[idx], vector_a_1[idx], this->modulus_));
  }
  for (size_t idx = 0; idx < dim2 * dim3; idx++) {
    vector_b.push_back(
        ModAdd(vector_b_0[idx], vector_b_1[idx], this->modulus_));
  }
  for (size_t idx = 0; idx < dim1 * dim3; idx++) {
    vector_c.push_back(
        ModAdd(vector_c_0[idx], vector_c_1[idx], this->modulus_));
  }

  std::vector<uint64_t> vector_c2;
  std::vector<uint64_t> zeroVec;
  vector_c2.resize(dim1*dim3);
  zeroVec.resize(dim1*dim3);

  // Compute C2 = A*B mod modulus directly.
  for (size_t rdx = 0; rdx < dim1; rdx++) {
    for (size_t cdx = 0; cdx < dim3; cdx++) {
      uint64_t sum = 0;
      zeroVec.push_back(sum);
      for (size_t kdx = 0; kdx < dim2; kdx++) {
        sum = ModAdd(sum,
                     ModMul(vector_a[rdx*dim2 + kdx],
                            vector_b[kdx*dim3 + cdx], this->modulus_),
                     this->modulus_);
      }
      vector_c2[rdx*dim3 + cdx] = sum;
    }
  }

  // Check if the function computes the output correctly.
  // Add extra check to make sure that the output is not a vector of zeros.
  EXPECT_EQ(vector_c, vector_c2);
  EXPECT_FALSE(vector_c == zeroVec);
}

TEST_F(BeaverGeneratorWithPrngTest, MultToAddShareEmptyDimensionFailsTest) {
  size_t length = 0;
  auto mult_to_add_shares = internal::SampleMultToAddSharesWithPrng(
      length, kPrimeModulus);

  EXPECT_THAT(mult_to_add_shares,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("length must be a positive integer.")));
}

// Test the SampleMultToAddSharesWithPrng function when the number of shares is
// a power of 2.
TEST_F(BeaverGeneratorWithPrngTest,
       MultToAddShareSucceedsOnPowerOfTwoInputSize) {
  size_t length = 16;
  ASSERT_OK_AND_ASSIGN(auto mult_to_add_shares,
                       internal::SampleMultToAddSharesWithPrng(
                           length, kPrimeModulus));

  auto vector_alpha_0 = mult_to_add_shares.first.first;
  auto vector_beta_0 = mult_to_add_shares.first.second;
  auto vector_alpha_1 = mult_to_add_shares.second.first;
  auto vector_beta_1 = mult_to_add_shares.second.second;

  for (size_t idx = 0; idx < length; idx++) {
    EXPECT_EQ(ModAdd(
        ModMul(vector_alpha_0[idx], vector_alpha_1[idx], kPrimeModulus),
        ModMul(vector_beta_0[idx], vector_beta_1[idx], kPrimeModulus),
        kPrimeModulus), 1);
  }
}

// Test the SampleMultToAddSharesWithPrng function when the number of shares is
// not a power of 2.
TEST_F(BeaverGeneratorWithPrngTest,
       MultToAddShareSucceedsOnNonPowerOfTwoInputSize) {
  size_t length = 10;
  ASSERT_OK_AND_ASSIGN(auto mult_to_add_shares,
                       internal::SampleMultToAddSharesWithPrng(
                           length, kPrimeModulus));

  auto vector_alpha_0 = mult_to_add_shares.first.first;
  auto vector_beta_0 = mult_to_add_shares.first.second;
  auto vector_alpha_1 = mult_to_add_shares.second.first;
  auto vector_beta_1 = mult_to_add_shares.second.second;

  for (size_t idx = 0; idx < length; idx++) {
    EXPECT_EQ(ModAdd(
        ModMul(vector_alpha_0[idx], vector_alpha_1[idx], kPrimeModulus),
        ModMul(vector_beta_0[idx], vector_beta_1[idx], kPrimeModulus),
        kPrimeModulus), 1);
  }
}

}  // namespace
}  // namespace private_join_and_compute
