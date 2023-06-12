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

#include "secret_sharing_mpc/arithmetic/matrix_arithmetic.h"

#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace private_join_and_compute {
namespace {

using ::testing::HasSubstr;
using testing::StatusIs;
using ::testing::Test;

class MatrixArithmeticTest : public Test {};

// Test that matrix transpose works correctly.
// For matrix mat, we get mat.transpose

TEST_F(MatrixArithmeticTest, MatrixTransposeInvalidDimensionsFails) {
  size_t dim1 = 2;
  size_t dim2 = 3;
  std::vector<uint64_t> mat{1, 2, 3, 4, 5};
  auto transposed_mat = Transpose(mat, dim1, dim2);
  EXPECT_THAT(transposed_mat,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("Transpose: invalid dimensions.")));
}

TEST_F(MatrixArithmeticTest, MatrixTransposeSucceeds) {
  size_t dim1 = 3;
  size_t dim2 = 2;
  std::vector<uint64_t> mat{1, 2, 3, 4, 5, 6};
  std::vector<uint64_t> expected_transpose{1, 3, 5, 2, 4, 6};
  ASSERT_OK_AND_ASSIGN(auto computed_transpose, Transpose(mat, dim1, dim2));
  for (size_t idx = 0; idx < dim1 * dim2; idx++) {
    EXPECT_EQ(computed_transpose[idx], expected_transpose[idx]);
  }
}

}  // namespace
}  // namespace private_join_and_compute
