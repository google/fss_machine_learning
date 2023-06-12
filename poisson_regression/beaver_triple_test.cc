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

#include "poisson_regression/beaver_triple.h"

#include "private_join_and_compute/util/status.inc"
#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace private_join_and_compute {
namespace {

using ::testing::HasSubstr;
using testing::StatusIs;
using ::testing::Test;

class BeaverTripleTest : public Test {
};

// Test: Create a Beaver Triple Vector from empty vector.
// Expect: Beaver triple vector created successfully.
TEST_F(BeaverTripleTest, TestEmptyBeaverTripleVector) {
  const std::vector<uint64_t> kEmptyVector;
  ASSERT_OK_AND_ASSIGN(BeaverTripleVector<uint64_t>  btv,
                       BeaverTripleVector<uint64_t>::Create(
                           kEmptyVector, kEmptyVector, kEmptyVector));
  EXPECT_EQ(btv.GetA(), kEmptyVector);
  EXPECT_EQ(btv.GetB(), kEmptyVector);
  EXPECT_EQ(btv.GetC(), kEmptyVector);
}

// Test: Create a Beaver Triple Vector from three small vectors.
// Expect: Beaver triple vector created successfully and the data
// is initialized correctly.
TEST_F(BeaverTripleTest, TestSmallBeaverTripleVector) {
  const std::vector<uint64_t> kA{1, 2, 3, 4, 5, 6};
  const std::vector<uint64_t> kB{2, 4, 2, 5, 6, 2};
  const std::vector<uint64_t> kC{4, 1, 0, 3, 8, 3};

  ASSERT_OK_AND_ASSIGN(BeaverTripleVector<uint64_t> btv,
                       BeaverTripleVector<uint64_t>::Create(kA, kB, kC));
  EXPECT_EQ(btv.GetA(), kA);
  EXPECT_EQ(btv.GetB(), kB);
  EXPECT_EQ(btv.GetC(), kC);
}

// Test: Create a Beaver Triple Vector using invalid input.
// Case 1: A.size() = B.size() != C.size().
// Case 2: A.size() = C.size() != B.size().
// Case 3: A.size() != B.size() = C.size().
// Expect: All invocations return INVALID_ARGUMENT status.
TEST_F(BeaverTripleTest, TestInvalidBeaverTripleVector) {
  const std::vector<uint64_t> kEmptyVector;
  const std::vector<uint64_t> kA{1, 2, 3, 4, 5, 6};
  const std::vector<uint64_t> kB{2, 4, 2, 5, 6, 2};
  const std::vector<uint64_t> kC{4, 1, 0, 3, 8, 3};

  auto btv1 = BeaverTripleVector<uint64_t>::Create(kA, kB, kEmptyVector);
  auto btv2 = BeaverTripleVector<uint64_t>::Create(kA, kEmptyVector, kC);
  auto btv3 = BeaverTripleVector<uint64_t>::Create(kEmptyVector, kB, kC);

  EXPECT_THAT(btv1, StatusIs(StatusCode::kInvalidArgument,
                             HasSubstr("must have the same size")));
  EXPECT_THAT(btv2, StatusIs(StatusCode::kInvalidArgument,
                             HasSubstr("must have the same size")));
  EXPECT_THAT(btv3, StatusIs(StatusCode::kInvalidArgument,
                             HasSubstr("must have the same size")));
}

// Test: GetTripleAt with valid index.
// Expect: The function returns status OK and the accessed triple is correct.
TEST_F(BeaverTripleTest, TestGetTripleAtGoodIndex) {
  const std::vector<uint64_t> kA{1, 2, 3, 4, 5, 6};
  const std::vector<uint64_t> kB{2, 4, 2, 5, 6, 2};
  const std::vector<uint64_t> kC{4, 1, 0, 3, 8, 3};

  const size_t kGoodIndex1 = 0;
  const size_t kGoodIndex2 = kA.size() - 1;

  ASSERT_OK_AND_ASSIGN(BeaverTripleVector<uint64_t> btv,
                       BeaverTripleVector<uint64_t>::Create(kA, kB, kC));
  uint64_t a, b, c;
  EXPECT_EQ((btv.GetTripleAt(kGoodIndex1, a, b, c)).code(), StatusCode::kOk);
  EXPECT_EQ(a, kA[kGoodIndex1]);
  EXPECT_EQ(b, kB[kGoodIndex1]);
  EXPECT_EQ(c, kC[kGoodIndex1]);

  EXPECT_EQ((btv.GetTripleAt(kGoodIndex2, a, b, c)).code(), StatusCode::kOk);
  EXPECT_EQ(a, kA[kGoodIndex2]);
  EXPECT_EQ(b, kB[kGoodIndex2]);
  EXPECT_EQ(c, kC[kGoodIndex2]);
}

// Test: GetTripleAt invalid index.
// Case 1: Non-empty Beaver Triple Vector -> index -1 and dim.
// Case 2: Empty Beaver Triple Vector -> index -1, 0, 1
// (all indices are invalid).
// Expect: The function returns INTERNAL status code.
TEST_F(BeaverTripleTest, TestGetTripleAtBadIndex) {
  const std::vector<uint64_t> kEmptyVector;
  const std::vector<uint64_t> kA1{1, 2, 3, 4, 5, 6};
  const std::vector<uint64_t> kB1{2, 4, 2, 5, 6, 2};
  const std::vector<uint64_t> kC1{4, 1, 0, 3, 8, 3};

  const size_t kBadIndex1 = -1;
  const size_t kBadIndex2 = kA1.size();

  ASSERT_OK_AND_ASSIGN(BeaverTripleVector<uint64_t> btv1,
                       BeaverTripleVector<uint64_t>::Create(kA1, kB1, kC1));
  uint64_t a, b, c;
  EXPECT_THAT(btv1.GetTripleAt(kBadIndex1, a, b, c),
              StatusIs(StatusCode::kInternal,
                       HasSubstr("GetTriple: Out of bound error")));
  EXPECT_THAT(btv1.GetTripleAt(kBadIndex2, a, b, c),
              StatusIs(StatusCode::kInternal,
                       HasSubstr("GetTriple: Out of bound error")));

  ASSERT_OK_AND_ASSIGN(BeaverTripleVector<uint64_t> btv2,
                       BeaverTripleVector<uint64_t>::Create(
                           kEmptyVector, kEmptyVector, kEmptyVector));
  EXPECT_THAT(btv2.GetTripleAt(-1, a, b, c),
              StatusIs(StatusCode::kInternal,
                       HasSubstr("GetTriple: Out of bound error")));
  EXPECT_THAT(btv2.GetTripleAt(0, a, b, c),
              StatusIs(StatusCode::kInternal,
                       HasSubstr("GetTriple: Out of bound error")));
  EXPECT_THAT(btv2.GetTripleAt(1, a, b, c),
              StatusIs(StatusCode::kInternal,
                       HasSubstr("GetTriple: Out of bound error")));
}

// Test: Create Beaver Triple Matrix with A(2, 3), B(3, 4), C(2, 4) and the
// function GetDimensions().
// This equivalents to dim1 = 2, dim2 = 3, dim3 = 4.
// Expect: The function creates a Beaver Triple Matrix  successfully
// and the data is correctly initialized. The function GetDimensions return
// correct values.
TEST_F(BeaverTripleTest, TestSmallBeaverTripleMatrix) {
  const std::vector<uint64_t> kA{3, 5, 6, 2, 4, 3};
  const std::vector<uint64_t> kB{7, 3, 8, 1, 3, 0, 2, 5, 3, 6, 2, 7};
  const std::vector<uint64_t> kC{2, 4, 9, 12, 4, 2, 7, 9};

  const size_t kDim1 = 2;
  const size_t kDim2 = 3;
  const size_t kDim3 = 4;

  ASSERT_OK_AND_ASSIGN(BeaverTripleMatrix<uint64_t> btm,
                       BeaverTripleMatrix<uint64_t>::Create(
                           kA, kB, kC, kDim1, kDim2, kDim3));
  EXPECT_EQ(btm.GetA(), kA);
  EXPECT_EQ(btm.GetB(), kB);
  EXPECT_EQ(btm.GetC(), kC);

  auto dimensions = btm.GetDimensions();
  auto dim1 = std::get<0>(dimensions);
  auto dim2 = std::get<1>(dimensions);
  auto dim3 = std::get<2>(dimensions);
  EXPECT_EQ(kA.size(), dim1*dim2);
  EXPECT_EQ(kB.size(), dim2*dim3);
  EXPECT_EQ(kC.size(), dim1*dim3);
}

// Test: Create Beaver Triple Matrix with one empty matrix.
// Expect: The function returns INVALID_ARGUMENT status code.
TEST_F(BeaverTripleTest, TestInvalidEmptyBeaverTripleMatrix) {
  const std::vector<uint64_t> kEmptyVector;
  const std::vector<uint64_t> kA{3, 5, 6, 2, 4, 3};
  const size_t kDim1 = 0;
  const size_t kDim2 = 2;
  const size_t kDim3 = 3;
  auto btm = BeaverTripleMatrix<uint64_t>::Create(
      kEmptyVector, kA, kEmptyVector, kDim1, kDim2, kDim3);
  EXPECT_THAT(btm, StatusIs(StatusCode::kInvalidArgument,
                            HasSubstr("Matrix dimension must not be zero")));
}

// Test: Create Beaver Triple Matrix. The input matrices and the dimensions
// do not meet the input's constraints.
// Case 1:
// A.size() = kDim1 * kDim2 AND B.size() = kDim2 * kDim3.
// However: C.size() = 6 != kDim1 * kDim3 = 2 * 4.
// Expect: Test returns INVALID_ARGUMENT status code.
TEST_F(BeaverTripleTest, TestInvalidBeaverTripleMatrix1) {
  const std::vector<uint64_t> kA{3, 5, 6, 2, 4, 3};
  const std::vector<uint64_t> kB{7, 3, 8, 1, 3, 0, 2, 5, 3, 6, 2, 7};
  const std::vector<uint64_t> kC{2, 4, 9, 12, 4, 2};

  const size_t kDim1 = 2;
  const size_t kDim2 = 3;
  const size_t kDim3 = 4;

  auto btm = BeaverTripleMatrix<uint64_t>::Create(
      kA, kB, kC, kDim1, kDim2, kDim3);

  EXPECT_THAT(btm, StatusIs(StatusCode::kInvalidArgument,
                            HasSubstr("Invalid size for matrices")));
}

// Test: Create Beaver Triple Matrix. The input matrices and the dimensions
// do not meet the input's constraints.
// Case 2:
// B.size() = kDim2 * kDim3 AND C.size() = kDim1 * kDim3.
// However: A.size() = 6 != dim1 * dim2 = 3 * 3.
// Expect: Test returns INVALID_ARGUMENT status code.
TEST_F(BeaverTripleTest, TestInvalidBeaverTripleMatrix2) {
  const std::vector<uint64_t> kA{3, 5, 6, 2, 4, 3};
  const std::vector<uint64_t> kB{7, 3, 8, 1, 3, 0, 2, 5, 3, 6, 2, 7};
  const std::vector<uint64_t> kC{2, 4, 9, 12, 4, 2, 7, 9, 0, 1, 5, 6};

  const size_t kDim1 = 3;
  const size_t kDim2 = 3;
  const size_t kDim3 = 4;

  auto btm = BeaverTripleMatrix<uint64_t>::Create(
      kA, kB, kC, kDim1, kDim2, kDim3);

  EXPECT_THAT(btm, StatusIs(StatusCode::kInvalidArgument,
                            HasSubstr("Invalid size for matrices")));
}

// Test: Create Beaver Triple Matrix. The input matrices and the dimensions
// do not meet the input's constraints.
// Case 3:
// C.size() = kDim1 * kDim3 AND A.size() = kDim1 * kDim2.
// However: B.size() = 6 != kDim2 * kDim3 = 2 * 4
// Expect: Test returns INVALID_ARGUMENT status code.
TEST_F(BeaverTripleTest, TestInvalidBeaverTripleMatrix3) {
  const std::vector<uint64_t> kA{3, 5, 6, 2, 4, 3};
  const std::vector<uint64_t> kB{2, 4, 9, 12, 4, 2};
  const std::vector<uint64_t> kC{7, 3, 8, 1, 3, 0, 2, 5, 3, 6, 2, 7};

  const size_t kDim1 = 3;
  const size_t kDim2 = 2;
  const size_t kDim3 = 4;

  auto btm = BeaverTripleMatrix<uint64_t>::Create(
      kA, kB, kC, kDim1, kDim2, kDim3);

  EXPECT_THAT(btm, StatusIs(StatusCode::kInvalidArgument,
                            HasSubstr("Invalid size for matrices")));
}

}  // namespace
}  // namespace private_join_and_compute
